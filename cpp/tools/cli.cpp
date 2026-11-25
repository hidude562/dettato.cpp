//   dettato_cli MODEL.gguf IN.wav [IN2.wav ...] -o OUTDIR [--device auto|cpu|N] [--threads N] [--no-drums] [--only NAME[,NAME...]] [--list-devices]
//
// --only keeps the tracks whose instrument name contains one of the names (case-insensitive), e.g. --only drums.
//
// Writes OUTDIR/<name>.mid and OUTDIR/<name>.json (tracks, note counts, tuning, drum model, timing), like the Python CLI.

#include <algorithm>
#include <cctype>
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

static std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

int main(int argc, char** argv)
{
    std::vector<std::string> only; // lower-case name fragments; empty keeps every track
    std::vector<std::string> inputs;
    std::string model, out_dir = ".", device = "auto";
    int threads = 0;
    bool drums = true;
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
        } else if (a == "--only") {
            if (i + 1 >= argc || std::string(argv[i + 1]).starts_with("--")) {
                std::fprintf(stderr, "--only requires a comma-separated list of instrument names\n");
                return 2;
            }
            const size_t previous = only.size();
            std::string list = lower(argv[++i]);
            for (size_t b = 0; b <= list.size();) {
                const size_t e = std::min(list.find(',', b), list.size());
                size_t x = b, y = e;
                while (x < y && std::isspace(static_cast<unsigned char>(list[x]))) {
                    x++;
                }
                while (y > x && std::isspace(static_cast<unsigned char>(list[y - 1]))) {
                    y--;
                }
                if (y > x) {
                    only.push_back(list.substr(x, y - x));
                }
                b = e + 1;
            }
            if (only.size() == previous) {
                std::fprintf(stderr, "--only requires at least one nonempty instrument name\n");
                return 2;
            }
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
        std::fprintf(stderr, "usage: %s MODEL.gguf IN.wav [...] -o OUTDIR [--device auto|cpu|N] [--threads N] [--no-drums] [--only NAME[,NAME...]]\n", argv[0]);
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
            // the drum model is not needed when no requested name can match the drum track
            to.drums = drums && (only.empty() || std::any_of(only.begin(), only.end(), [](const std::string& o) {
                                     return std::string("drums").find(o) != std::string::npos;
                                 }));
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
            if (!only.empty()) {
                std::string found;
                std::vector<dtt::MidiTrack> kept;
                for (auto& tr : mt) {
                    found += (found.empty() ? "" : ", ") + tr.name;
                    const std::string name = lower(tr.name);
                    if (std::any_of(only.begin(), only.end(), [&](const std::string& o) { return name.find(o) != std::string::npos; })) {
                        kept.push_back(std::move(tr));
                    }
                }
                if (kept.empty()) {
                    std::fprintf(stderr, "%s: no track matches --only; found: %s\n", in.c_str(), found.empty() ? "(none)" : found.c_str());
                    failures++;
                    continue;
                }
                mt = std::move(kept);
            }
            size_t n_notes = 0;
            for (const auto& tr : mt) {
                n_notes += tr.notes.size();
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
                         r->seconds / secs, mt.size(), n_notes);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", in.c_str(), e.what());
            failures++;
        }
    }
    return failures ? 1 : 0;
}
