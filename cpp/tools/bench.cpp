// Transcriber throughput: dettato_bench MODEL.gguf AUDIO16K.npy [--device D] [--batch B] [--threads N] [--hop-frac F]
//                                                        [--taps none|hidden|both] [--repeat R]
// Times the CPU front end, the network and the readback separately, then a full Transcriber::run.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "backend.hpp"
#include "model_file.hpp"
#include "npy.hpp"
#include "transcriber/transcriber.hpp"

using namespace dtt;
using clk = std::chrono::steady_clock;

static std::string arg(int argc, char** argv, const char* key, const char* def)
{
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], key) == 0) {
            return argv[i + 1];
        }
    }
    return def;
}

static double ms(clk::time_point a)
{
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

struct NullSink : TapSink {
    void window(int, int, int, int, const float*, const float*, int, int, int) override {}
};

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf AUDIO16K.npy [--device D] [--batch B] [--threads N] [--hop-frac F] [--taps none|hidden|both]\n", argv[0]);
        return 2;
    }
    try {
        Backend be(arg(argc, argv, "--device", "auto"), std::stoi(arg(argc, argv, "--threads", "0")));
        const auto t_load = clk::now();
        ModelFile mf(argv[1], be.get());
        const std::string taps = arg(argc, argv, "--taps", "hidden");
        const int B = std::stoi(arg(argc, argv, "--batch", "4"));
        const double hop_frac = std::stod(arg(argc, argv, "--hop-frac", "0.5"));
        Transcriber tr(mf, be, B, LaneNet::Taps{taps != "none", taps == "both", false, arg(argc, argv, "--tap-f16", "0") == "1"});
        std::printf("backend %s | load %.0f ms | graph %d nodes, compute buffer %.0f MB | batch %d\n", be.name().c_str(), ms(t_load),
                    tr.net().nodes(), tr.net().computeBytes() / 1e6, B);
        const auto audio = npy::load(argv[2]).f32();
        const double secs = audio.size() / 16000.0;

        // phase timing on one batch of windows from the middle of the song
        const size_t W = 128000;
        const size_t per = static_cast<size_t>(tr.net().inChannels()) * (W / 160) * tr.net().bins();
        std::vector<float> xin(per * B), logits, hidden;
        const size_t s0 = audio.size() > W ? (audio.size() - W) / 2 : 0;
        for (int rep = 0; rep < 3; rep++) {
            auto t = clk::now();
            for (int k = 0; k < B; k++) {
                tr.windowHcqt(audio.data() + std::min(s0, audio.size() - std::min(audio.size(), W)), xin.data() + per * k);
            }
            const double t_h = ms(t);
            t = clk::now();
            tr.net().run(xin.data());
            const double t_n = ms(t);
            t = clk::now();
            tr.net().logits(logits);
            if (taps != "none") {
                if (tr.net().tapsF16()) {
                    std::vector<ggml_fp16_t> h16;
                    tr.net().hiddenF16(h16);
                } else {
                    tr.net().hidden(hidden);
                }
            }
            const double t_r = ms(t);
            std::printf("batch of %d windows: hcqt %.1f ms (CPU, 1 thread), network %.1f ms, readback %.1f ms -> %.1f ms/window\n", B, t_h,
                        t_n, t_r, (t_h + t_n + t_r) / B);
        }

        NullSink sink;
        const auto t = clk::now();
        const Maps maps = tr.run(audio.data(), audio.size(), hop_frac, taps != "none" ? &sink : nullptr);
        const double s = ms(t) / 1000.0;
        const auto lo = tr.layout(audio.size(), hop_frac);
        std::printf("run: %.1f s audio, %zu windows (hop_frac %.2f): %.2f s = %.1f ms/window, RTF %.2f, %zu notes\n", secs,
                    lo.starts.size(), hop_frac, s, 1000.0 * s / lo.starts.size(), secs / s, tr.notes(maps).size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
