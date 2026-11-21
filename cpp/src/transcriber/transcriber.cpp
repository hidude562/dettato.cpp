#include "transcriber/transcriber.hpp"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <future>
#include <numbers>
#include <thread>
#include <tuple>

#include "backend.hpp"
#include "common.hpp"
#include "model_file.hpp"

namespace dtt
{

Transcriber::Transcriber(const ModelFile& mf, const Backend& be, int batch, LaneNet::Taps taps)
    : mBe(be), mHcqt(mf, "tr"), mTaps(taps)
{
    mHop = 0.01 * 2; // model.hop_out = 0.01 * t_stride, bit for bit
    const int frames10 = static_cast<int>(std::lround(WIN_S * SR)) / Hcqt::HOP;
    mNet = std::make_unique<LaneNet>(mf, be, batch, frames10, taps);
    mParams.on_th = mf.f32("tr.th.on_th");
    mParams.off_th = mf.f32("tr.th.off_th");
    mParams.fr_th = mf.f32("tr.th.fr_th");
    mParams.fr_low_frames = static_cast<int>(mf.f32("tr.th.fr_low_frames"));
    mParams.on_nms = mf.has("tr.th.on_nms") ? static_cast<int>(mf.f32("tr.th.on_nms")) : 0;
    mV2Th = mf.has("tr.th.v2_th") ? mf.f32("tr.th.v2_th") : -1.0f;
}

Transcriber::~Transcriber() = default;

Transcriber::Layout Transcriber::layout(size_t n, double hop_frac) const
{
    Layout lo;
    lo.W = std::lround(WIN_S * SR);
    lo.Fw = static_cast<int>(std::nearbyint(WIN_S / mHop));
    lo.T = static_cast<int>(std::ceil(static_cast<double>(n) / SR / mHop));
    const long step = std::max(1L, static_cast<long>(std::nearbyint(lo.W * hop_frac)));
    lo.pad = lo.W / 2;
    for (long s = 0; s < static_cast<long>(n) + lo.pad; s += step) {
        lo.starts.push_back(s);
    }
    lo.g_pad = static_cast<int>(std::nearbyint(static_cast<double>(lo.pad) / SR / mHop));
    return lo;
}

int Transcriber::frameOffset(long s0, int g_pad) const
{
    return static_cast<int>(std::nearbyint(static_cast<double>(s0) / SR / mHop)) - g_pad;
}

std::vector<float> Transcriber::taper(int Fw) const
{
    // np.hanning(M) = 0.5 - 0.5 cos(2 pi n / (M - 1)), in float64, then the ends dropped and cast to float32
    const int M = Fw + 2;
    std::vector<float> w(Fw);
    for (int i = 0; i < Fw; i++) {
        w[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * (i + 1) / (M - 1)));
    }
    return w;
}

void Transcriber::windowHcqt(const float* x, float* out) const
{
    mHcqt.compute(x, static_cast<size_t>(std::lround(WIN_S * SR)), out);
}

Maps Transcriber::run(const float* audio, size_t n, double hop_frac, TapSink* sink, unsigned tap_mask,
                      const std::function<bool(float)>& progress)
{
    RunOptions o;
    o.hop_frac = hop_frac;
    o.sink = sink;
    o.tap_mask = tap_mask;
    o.progress = progress;
    return run(audio, n, o);
}

Maps Transcriber::run(const float* audio, size_t n, const RunOptions& o)
{
    TapSink* sink = o.sink;
    const auto& progress = o.progress;
    const bool want_h = sink != nullptr && (o.tap_mask & (1u << TapSink::HIDDEN));
    const bool want_s = sink != nullptr && (o.tap_mask & (1u << TapSink::STEM));
    DTT_CHECK(!want_h || mTaps.hidden, "the hidden tap was not enabled");
    DTT_CHECK(!want_s || mTaps.stem, "the stem tap was not enabled");
    const Layout lo = layout(n, o.hop_frac);
    const int B = mNet->batch(), L = mNet->lanes(), Fw = lo.Fw, C = mNet->channels();
    DTT_CHECK(Fw == mNet->framesOut(), "window frames %d != network frames %d", Fw, mNet->framesOut());
    const auto w = taper(Fw);
    Maps maps;
    maps.T = lo.T;
    maps.L = L;
    for (auto& m : maps.m) {
        m.assign(static_cast<size_t>(lo.T) * L, 0.0f);
    }
    std::vector<float> wsum(lo.T, 0.0f);
    // the second average, over the windows whose start is a multiple of sub_step samples
    std::vector<float> sub_wsum;
    if (o.sub_maps != nullptr) {
        DTT_CHECK(o.sub_step > 0, "sub_maps needs sub_step");
        o.sub_maps->T = lo.T;
        o.sub_maps->L = L;
        for (auto& m : o.sub_maps->m) {
            m.assign(static_cast<size_t>(lo.T) * L, 0.0f);
        }
        sub_wsum.assign(lo.T, 0.0f);
    }
    // the windows to run, in order (all of them unless o.only selects some)
    std::vector<long> starts;
    for (size_t i = 0; i < lo.starts.size(); i++) {
        if (o.only == nullptr || (i < o.only->size() && (*o.only)[i])) {
            starts.push_back(lo.starts[i]);
        }
    }

    const size_t W = static_cast<size_t>(lo.W);
    const size_t win_in = static_cast<size_t>(mNet->inChannels()) * (W / Hcqt::HOP) * mNet->bins();
    const size_t nwin = starts.size();
    const size_t nbatch = (nwin + B - 1) / B;
    const int H = mNet->heads();
    int finalized = 0;
    auto normalise = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            const float ws = std::max(wsum[t], 1e-6f);
            for (auto& m : maps.m) {
                for (int l = 0; l < L; l++) {
                    m[static_cast<size_t>(t) * L + l] /= ws;
                }
            }
        }
    };

    // Three stages overlap: the CPU front end of batch k+1 (one thread per window), the network on batch k, and the
    // accumulation of batch k-1 (maps, tap sink, final frames — in order). The input alternates between two buffers
    // (prepare(k+1) fills one while the network reads the other). The outputs need one set: batch k is read back only
    // after consume(k-1) has finished. The batch index is passed by value, never stored.
    struct Outputs {
        std::vector<float> logits, hidden, stem;
        std::vector<ggml_fp16_t> hidden16, stem16; // when the taps come back as f16 (converted in consume, off the critical path)
    };
    const bool f16 = mNet->tapsF16();
    std::vector<float> xins[2];
    Outputs s;
    auto first = [&](size_t batch) { return batch * B; };
    auto count = [&](size_t batch) { return static_cast<int>(std::min<size_t>(B, nwin - batch * B)); };
    auto prepare = [&](size_t batch, std::vector<float>& xin) {
        const size_t bi = first(batch);
        const int nb = count(batch);
        xin.resize(win_in * B);
        std::vector<std::thread> pool;
        for (int k = 0; k < B; k++) {
            pool.emplace_back([&, k] {
                std::vector<float> xw(W, 0.0f); // a short last batch keeps silent windows
                if (k < nb) {
                    // x_all = [pad zeros, audio, pad + W zeros]; the window is x_all[s0 : s0 + W]
                    const long s0 = starts[bi + k];
                    for (size_t i = 0; i < W; i++) {
                        const long j = s0 + static_cast<long>(i) - lo.pad;
                        xw[i] = (j >= 0 && j < static_cast<long>(n)) ? audio[j] : 0.0f;
                    }
                }
                mHcqt.compute(xw.data(), W, xin.data() + win_in * k);
            });
        }
        for (auto& t : pool) {
            t.join();
        }
    };
    auto consume = [&](size_t batch) {
        const size_t bi = first(batch);
        const int nb = count(batch);
        for (int k = 0; k < nb; k++) {
            const int g0 = frameOffset(starts[bi + k], lo.g_pad);
            const bool sub = o.sub_maps != nullptr && starts[bi + k] % o.sub_step == 0;
            const int a = std::max(0, g0), b = std::min(lo.T, g0 + Fw);
            if (b <= a) {
                continue;
            }
            const float* lg = s.logits.data() + static_cast<size_t>(k) * H * L * Fw;
            for (int t = a; t < b; t++) {
                const int f = t - g0;
                const float wt = w[f];
                for (int l = 0; l < L; l++) {
                    const float* z = lg + (static_cast<size_t>(f) * L + l) * H;
                    for (int h = 0; h < N_HEADS; h++) {
                        const float pr = 1.0f / (1.0f + std::exp(-z[h]));
                        maps.m[h][static_cast<size_t>(t) * L + l] += pr * wt;
                        if (sub) {
                            o.sub_maps->m[h][static_cast<size_t>(t) * L + l] += pr * wt;
                        }
                    }
                }
                wsum[t] += wt;
                if (sub) {
                    sub_wsum[t] += wt;
                }
            }
            if (sink != nullptr) {
                const size_t per = static_cast<size_t>(C) * L * Fw;
                for (int tap : {TapSink::HIDDEN, TapSink::STEM}) {
                    if (!(tap == TapSink::HIDDEN ? want_h : want_s)) {
                        continue;
                    }
                    std::vector<float>& f32 = tap == TapSink::HIDDEN ? s.hidden : s.stem;
                    const float* data = nullptr;
                    if (f16) {
                        const auto& h16 = tap == TapSink::HIDDEN ? s.hidden16 : s.stem16;
                        f32.resize(per);
                        ggml_fp16_to_fp32_row(h16.data() + per * k, f32.data(), static_cast<int64_t>(per));
                        data = f32.data();
                    } else {
                        data = f32.data() + per * k;
                    }
                    sink->window(tap, g0, a, b, w.data() + (a - g0), data, C, L, Fw);
                }
            }
        }
        // frames before the next window's first frame are final: normalise them and tell the sink
        const int final_to = bi + B < nwin ? std::clamp(frameOffset(starts[bi + B], lo.g_pad), finalized, lo.T) : lo.T;
        normalise(finalized, final_to);
        if (sink != nullptr && final_to > finalized) {
            sink->framesFinal(finalized, final_to, maps, wsum);
        }
        finalized = final_to;
    };

    std::future<void> prep, cons;
    auto drain = [&] {
        for (std::future<void>* f : {&prep, &cons}) {
            if (f->valid()) {
                try {
                    f->get();
                } catch (...) {
                }
            }
        }
    };
    if (nbatch > 0) {
        prep = std::async(std::launch::async, prepare, 0, std::ref(xins[0]));
    }
    try {
        for (size_t k = 0; k < nbatch; k++) {
            prep.get();
            if (k + 1 < nbatch) {
                prep = std::async(std::launch::async, prepare, k + 1, std::ref(xins[(k + 1) % 2]));
            }
            mNet->run(xins[k % 2].data());
            if (cons.valid()) {
                cons.get(); // batch k-1 accumulated: the outputs are free to overwrite
            }
            mNet->logits(s.logits);
            if (want_h) {
                f16 ? mNet->hiddenF16(s.hidden16) : mNet->hidden(s.hidden);
            }
            if (want_s) {
                f16 ? mNet->stemF16(s.stem16) : mNet->stem(s.stem);
            }
            cons = std::async(std::launch::async, consume, k);
            if (progress && !progress(static_cast<float>(std::min(first(k) + B, nwin)) / static_cast<float>(nwin))) {
                throw Cancelled();
            }
        }
        if (cons.valid()) {
            cons.get();
        }
    } catch (...) {
        drain();
        throw;
    }
    if (finalized < lo.T) {
        normalise(finalized, lo.T);
        if (sink != nullptr) {
            sink->framesFinal(finalized, lo.T, maps, wsum);
        }
    }
    if (o.sub_maps != nullptr) {
        for (int t = 0; t < lo.T; t++) {
            const float ws = std::max(sub_wsum[t], 1e-6f);
            for (auto& m : o.sub_maps->m) {
                for (int l = 0; l < L; l++) {
                    m[static_cast<size_t>(t) * L + l] /= ws;
                }
            }
        }
    }
    return maps;
}

std::vector<TranscribedNote> Transcriber::notes(const Maps& maps) const
{
    std::vector<TranscribedNote> rows;
    struct Voice {
        int slot;
        Head on, fr, off;
        float th;
    };
    for (const Voice v : {Voice{1, ONSET, FRAME, OFFSET, mParams.on_th}, Voice{2, ONSET2, FRAME2, OFFSET2, mV2Th}}) {
        if (v.th < 0.0f) {
            continue;
        }
        DecodeParams p = mParams;
        p.on_th = v.th;
        for (const DecodedNote& d : decodeVoice(maps[v.on], maps[v.fr], maps[v.off], maps.T, maps.L, mHop, p)) {
            const int t = static_cast<int>(std::min(std::max(std::nearbyint(d.onset / mHop), 0.0), static_cast<double>(maps.T - 1)));
            rows.push_back({d.onset, d.offset, d.pitch, maps.at(v.on, t, d.pitch - 21), v.slot});
        }
    }
    std::stable_sort(rows.begin(), rows.end(), [](const TranscribedNote& a, const TranscribedNote& b) {
        return std::tie(a.onset, a.offset, a.pitch) < std::tie(b.onset, b.offset, b.pitch);
    });
    return rows;
}

} // namespace dtt
