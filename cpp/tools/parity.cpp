// Parity of the C++ transcriber against the PyTorch reference dumps (py/dump_refs.py).
//
//   dettato_parity MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--batch B] [--tol X]
//
// Exit status 0 when every compared tensor is within --tol relative L2 and the notes match.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "backend.hpp"
#include "common.hpp"
#include "model_file.hpp"
#include "npy.hpp"
#include "transcriber/transcriber.hpp"

using namespace dtt;

namespace
{
int g_fail = 0;
double g_tol = 1e-3;

struct Stat {
    double max_abs = 0, ref_max = 0, rel_l2 = 0;
};

Stat compare(const std::vector<float>& a, const std::vector<double>& ref)
{
    Stat s;
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = a[i] - ref[i];
        s.max_abs = std::max(s.max_abs, std::abs(d));
        s.ref_max = std::max(s.ref_max, std::abs(ref[i]));
        num += d * d;
        den += ref[i] * ref[i];
    }
    s.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    return s;
}

void report(const std::string& name, const std::vector<float>& ours, const npy::Array& ref, double tol = -1)
{
    if (tol < 0) {
        tol = g_tol;
    }
    if (ours.size() != ref.size()) {
        std::printf("  %-22s SIZE MISMATCH ours %zu ref %zu\n", name.c_str(), ours.size(), ref.size());
        g_fail++;
        return;
    }
    const Stat s = compare(ours, ref.data);
    const bool ok = s.rel_l2 <= tol;
    g_fail += ok ? 0 : 1;
    std::printf("  %-22s rel_l2 %.3e  max_abs %.3e  (ref max %.3e)  %s\n", name.c_str(), s.rel_l2, s.max_abs, s.ref_max, ok ? "ok" : "FAIL");
}

/** ours: channels-last [C, P, T] (C fastest) -> torch (C, T, P) row-major. */
std::vector<float> clToTorch(const std::vector<float>& x, int64_t C, int64_t P, int64_t T)
{
    std::vector<float> o(x.size());
    for (int64_t c = 0; c < C; c++)
        for (int64_t t = 0; t < T; t++)
            for (int64_t p = 0; p < P; p++)
                o[(c * T + t) * P + p] = x[c + C * (p + P * t)];
    return o;
}

std::string arg(int argc, char** argv, const char* key, const char* def)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

/** Accumulates the hidden tap over the whole song as Transcriber.run does, for comparison. */
struct FullSink : TapSink {
    int T, L, C;
    std::vector<float> buf;
    FullSink(int T_, int L_, int C_) : T(T_), L(L_), C(C_), buf(static_cast<size_t>(T_) * L_ * C_, 0.0f) {}
    void window(int tap, int g0, int a, int b, const float* w, const float* data, int C_, int L_, int Fw) override
    {
        if (tap != HIDDEN) {
            return;
        }
        for (int t = a; t < b; t++) {
            const int f = t - g0;
            const float wt = w[t - a];
            for (int l = 0; l < L; l++) {
                const float* src = data + static_cast<size_t>(C) * (l + static_cast<size_t>(L) * f);
                float* dst = buf.data() + (static_cast<size_t>(t) * L + l) * C;
                for (int c = 0; c < C; c++) {
                    dst[c] += src[c] * wt;
                }
            }
        }
    }
};
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--batch B] [--tol X]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], dir = argv[2];
    g_tol = std::stod(arg(argc, argv, "--tol", "1e-3"));
    try {
        Backend be(arg(argc, argv, "--device", "cpu"), std::stoi(arg(argc, argv, "--threads", "0")));
        std::printf("backend: %s\n", be.name().c_str());
        ModelFile mf(model, be.get());
        std::printf("weights: %.1f MB\n", mf.bytes() / 1e6);

        // ---- one window, stage by stage (batch 1)
        {
            Transcriber tr(mf, be, 1, LaneNet::Taps{true, false, true});
            LaneNet& net = tr.net();
            std::printf("window graph: %d nodes, compute buffer %.1f MB\n", net.nodes(), net.computeBytes() / 1e6);
            const auto x = npy::load(dir + "/win/x.npy").f32();
            const auto href = npy::load(dir + "/win/hcqt.npy");
            std::vector<float> hc(href.size());
            tr.windowHcqt(x.data(), hc.data());
            report("hcqt", hc, href);
            // the network on the reference HCQT, so its errors are its own
            const auto t0 = std::chrono::steady_clock::now();
            net.run(href.f32().data());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  (one window: %.1f ms)\n", ms);
            std::vector<float> v;
            std::vector<int64_t> ne;
            net.debug("stem", v, ne);
            report("stem", v, npy::load(dir + "/win/stem.npy"));
            for (const char* n : {"down", "layers_0", "layers_5", "layers_6", "layers_7", "layers_23", "up_norm", "refine_1"}) {
                net.debug(n, v, ne);
                report(n, clToTorch(v, ne[0], ne[1], ne[2]), npy::load(dir + "/win/" + n + ".npy"));
            }
            net.hidden(v);
            report("hidden", clToTorch(v, net.channels(), net.lanes(), net.framesOut()), npy::load(dir + "/win/hidden.npy"));
            net.logits(v);
            const char* heads[] = {"onset", "frame", "offset", "onset2", "frame2", "offset2"};
            for (int h = 0; h < N_HEADS; h++) {
                std::vector<float> o(static_cast<size_t>(net.framesOut()) * net.lanes());
                for (int t = 0; t < net.framesOut(); t++)
                    for (int l = 0; l < net.lanes(); l++)
                        o[static_cast<size_t>(t) * net.lanes() + l] = v[(static_cast<size_t>(t) * net.lanes() + l) * N_HEADS + h];
                report(std::string("logit_") + heads[h], o, npy::load(dir + "/win/logit_" + heads[h] + ".npy"));
            }
        }

        // ---- the whole fixture: maps, the hidden tap, notes
        {
            const int batch = std::stoi(arg(argc, argv, "--batch", "2"));
            Transcriber tr(mf, be, batch, LaneNet::Taps{true, false, false});
            const auto audio = npy::load(dir + "/audio16k.npy").f32();
            const auto lo = tr.layout(audio.size(), 0.25);
            FullSink sink(lo.T, tr.net().lanes(), tr.channels());
            const auto t0 = std::chrono::steady_clock::now();
            const Maps maps = tr.run(audio.data(), audio.size(), 0.25, &sink);
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("run: %.1f s audio, %zu windows, batch %d: %.2f s (%.2fx real time)\n", audio.size() / 16000.0,
                        lo.starts.size(), batch, s, audio.size() / 16000.0 / s);
            const char* heads[] = {"onset", "frame", "offset", "onset2", "frame2", "offset2"};
            for (int h = 0; h < N_HEADS; h++) {
                report(std::string("map_") + heads[h], maps.m[h], npy::load(dir + "/run/map_" + heads[h] + ".npy"));
            }
            // the reference accumulates in float16; ours in float32 (looser tolerance)
            std::vector<float> wsum(lo.T, 0.0f);
            const auto w = tr.taper(lo.Fw);
            for (long s0 : lo.starts) {
                const int g0 = tr.frameOffset(s0, lo.g_pad);
                for (int t = std::max(0, g0); t < std::min(lo.T, g0 + lo.Fw); t++) {
                    wsum[t] += w[t - g0];
                }
            }
            for (int t = 0; t < lo.T; t++)
                for (size_t i = 0; i < static_cast<size_t>(sink.L) * sink.C; i++)
                    sink.buf[static_cast<size_t>(t) * sink.L * sink.C + i] /= std::max(wsum[t], 1e-6f);
            report("run_hidden", sink.buf, npy::load(dir + "/run/hidden.npy"), std::max(g_tol, 5e-3));

            const auto notes = tr.notes(maps);
            const auto on = npy::load(dir + "/run/notes_onset.npy"), off = npy::load(dir + "/run/notes_offset.npy"),
                       pitch = npy::load(dir + "/run/notes_pitch.npy"), slot = npy::load(dir + "/run/notes_slot.npy"),
                       conf = npy::load(dir + "/run/notes_conf.npy");
            // exact matches: same pitch and slot, onset within 0.1 ms
            std::multimap<std::pair<int, int>, size_t> refidx;
            for (size_t i = 0; i < on.size(); i++) {
                refidx.emplace(std::make_pair(static_cast<int>(pitch.data[i]), static_cast<int>(slot.data[i])), i);
            }
            size_t match = 0;
            double d_on = 0, d_off = 0, d_conf = 0;
            std::vector<char> used(on.size(), 0);
            for (const auto& n : notes) {
                auto [lo_it, hi_it] = refidx.equal_range({n.pitch, n.slot});
                for (auto it = lo_it; it != hi_it; ++it) {
                    const size_t i = it->second;
                    if (!used[i] && std::abs(on.data[i] - n.onset) < 1e-4) {
                        used[i] = 1;
                        match++;
                        d_on = std::max(d_on, std::abs(on.data[i] - n.onset));
                        d_off = std::max(d_off, std::abs(off.data[i] - n.offset));
                        d_conf = std::max(d_conf, std::abs(conf.data[i] - n.conf));
                        break;
                    }
                }
            }
            const bool ok = match == notes.size() && match == on.size();
            g_fail += ok ? 0 : 1;
            std::printf("  notes: ours %zu, ref %zu, matched %zu  (max d_onset %.2e s, d_offset %.2e s, d_conf %.2e)  %s\n",
                        notes.size(), on.size(), match, d_on, d_off, d_conf, ok ? "ok" : "MISMATCH");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
