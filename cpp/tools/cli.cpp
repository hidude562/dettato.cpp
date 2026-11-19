//   dettato_cli MODEL.gguf IN.wav [IN2.wav ...] -o OUTDIR [--device auto|cpu|N] [--threads N] [--no-drums] [--list-devices]
//
// Writes OUTDIR/<name>.mid and OUTDIR/<name>.json (tracks, note counts, tuning, drum model, timing), like the Python CLI.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "dettato/dettato.hpp"
#include "io/midi.hpp"
#include "io/wav.hpp"

namespace fs = std::filesystem;

static std::string jsonEscape(const std::string& s)
{
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
        }
        o += c;
    }
    return o;
}

int main(int argc, char** argv)
{
    std::vector<std::string> inputs;
    std::string model, out_dir = ".", device = "auto";
    int threads = 0;
    bool drums = true;
    bool fast = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (a == "--no-drums") {
            drums = false;
        } else if (a == "--fast") {
            fast = true;
        } else if (a == "--list-devices") {
            const auto d = dtt::availableDevices();
            for (size_t k = 0; k < d.size(); k++) {
                std::printf("%2zu  %-7s %s%s\n", k, d[k].backend.c_str(), d[k].name.c_str(), d[k].integrated ? " (integrated)" : "");
            }
            return 0;
        } else if (model.empty()) {
            model = a;
        } else {
            inputs.push_back(a);
        }
    }
    if (model.empty() || inputs.empty()) {
        std::fprintf(stderr, "usage: %s MODEL.gguf IN.wav [...] -o OUTDIR [--device auto|cpu|N] [--threads N] [--no-drums] [--fast]\n", argv[0]);
        return 2;
    }
    dtt::LoadOptions lo;
    lo.n_threads = threads;
    if (device == "cpu") {
        lo.device = dtt::availableDevices().size() - 1;
    } else if (device != "auto") {
        lo.device = std::stoul(device);
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto pipe = dtt::Pipeline::load(model, lo);
    if (!pipe) {
        std::fprintf(stderr, "error: %s\n", dtt::describe(pipe.error()));
        return 1;
    }
    std::fprintf(stderr, "loaded %s on %s (%s) in %.2f s\n", model.c_str(), pipe->device().name.c_str(), pipe->device().backend.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    fs::create_directories(out_dir);
    int failures = 0;
    for (const std::string& in : inputs) {
        try {
            const dtt::Audio a = dtt::readWav(in);
            dtt::TranscribeOptions to;
            to.drums = drums;
            to.quality = fast ? dtt::Quality::Fast : dtt::Quality::Best;
            const auto t = std::chrono::steady_clock::now();
            auto r = pipe->transcribe(a.samples, a.rate, to);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            if (!r) {
                std::fprintf(stderr, "%s: %s %s\n", in.c_str(), dtt::describe(r.error()), pipe->lastErrorMessage().c_str());
                failures++;
                continue;
            }
            std::vector<dtt::MidiTrack> mt(r->tracks.size());
            for (size_t k = 0; k < r->tracks.size(); k++) {
                mt[k].name = r->tracks[k].name;
                mt[k].program = r->tracks[k].program;
                mt[k].is_drum = r->tracks[k].is_drum;
            }
            for (const auto& n : r->notes) {
                mt[n.track].notes.push_back({n.onset, n.offset, n.pitch, n.velocity});
            }
            const std::string stem = fs::path(in).stem().string();
            dtt::saveMidi((fs::path(out_dir) / (stem + ".mid")).string(), mt);
            std::ofstream js(fs::path(out_dir) / (stem + ".json"));
            js << "{\"source\": \"" << jsonEscape(in) << "\", \"seconds\": " << r->seconds << ", \"run_seconds\": " << secs
               << ", \"tuning_cents\": " << r->tuning_cents << ", \"drums\": \"" << r->drum_model << "\", \"notes\": {\"transcribed\": "
               << r->notes_transcribed << ", \"dropped\": " << r->notes_dropped << ", \"inserted\": " << r->notes_inserted
               << "}, \"tracks\": [";
            for (size_t k = 0; k < mt.size(); k++) {
                js << (k ? ", " : "") << "{\"name\": \"" << jsonEscape(mt[k].name) << "\", \"program\": " << mt[k].program
                   << ", \"is_drum\": " << (mt[k].is_drum ? "true" : "false") << ", \"notes\": " << mt[k].notes.size() << "}";
            }
            js << "]}\n";
            std::fprintf(stderr, "%s: %.1f s audio in %.1f s (%.1fx real time), %zu tracks, %zu notes\n", in.c_str(), r->seconds, secs,
                         r->seconds / secs, r->tracks.size(), r->notes.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", in.c_str(), e.what());
            failures++;
        }
    }
    return failures ? 1 : 0;
}
