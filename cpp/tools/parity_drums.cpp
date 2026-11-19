// Parity of the C++ drum stage against the PyTorch reference dumps (py/dump_refs_drums.py).
//
//   dettato_parity_drums MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X] [--skip features,nets,e2e]
//                        [--cq-chunk FRAMES]   (the windowed cq branch above 1.5 windows; 0 = one graph)
//                        [--poll P]            (CPU: a persistent ggml threadpool with polling level P, an experiment)
//
// 1. features: with Python's 16 kHz signal injected (the three log-power blocks before the float16 rounding, then the
//    float16 features: every element equal or one float16 step away), and with our resampler (reported).
// 2. networks on the reference features: every tapped tensor of every model, the probabilities, the router.
// 3. hits: decode of (2); transcribe() with the injected 16 kHz signal (must be exact); transcribe() as shipped (our
//    resampler; agreement reported), with its time per part.
//
// Exit status 0 when every compared tensor is within --tol relative L2 and the injected-signal hits are exact.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <ggml-cpu.h>
#include <ggml.h>

#include "backend.hpp"
#include "common.hpp"
#include "drums/decode.hpp"
#include "drums/drums.hpp"
#include "drums/features.hpp"
#include "drums/nets.hpp"
#include "dsp/resample.hpp"
#include "model_file.hpp"
#include "npy.hpp"

using namespace dtt;

namespace
{
int g_fail = 0;
double g_tol = 1e-4;
bool g_gpu = false;

bool exists(const std::string& p)
{
    std::ifstream f(p);
    return f.good();
}

double relL2(const float* a, const double* ref, size_t n, double* max_abs = nullptr, double* ref_max = nullptr)
{
    double num = 0, den = 0, ma = 0, rm = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = a[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        ma = std::max(ma, std::abs(d));
        rm = std::max(rm, std::abs(ref[i]));
    }
    if (max_abs) {
        *max_abs = ma;
    }
    if (ref_max) {
        *ref_max = rm;
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

void report(const std::string& name, const std::vector<float>& ours, const npy::Array& ref, double tol = -1, size_t n = 0)
{
    if (tol < 0) {
        tol = g_tol;
    }
    if (n == 0) {
        n = ref.size();
        if (ours.size() != ref.size()) {
            std::printf("  %-18s SIZE MISMATCH ours %zu ref %zu\n", name.c_str(), ours.size(), ref.size());
            g_fail++;
            return;
        }
    }
    double ma, rm;
    const double r = relL2(ours.data(), ref.data.data(), n, &ma, &rm);
    const bool ok = r <= tol;
    g_fail += ok ? 0 : 1;
    std::printf("  %-18s rel_l2 %.3e  max_abs %.3e  (ref max %.3e)  %s\n", name.c_str(), r, ma, rm, ok ? "ok" : "FAIL");
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

/**
 * float16 features vs the reference's. Both are the float16 roundings of float32 values (ours32, ref32) that agree to
 * ~1e-7; an element differs when a float16 rounding boundary lies between the two. Checked: every differing element's
 * float32 values agree within the tolerance (relative, absolute below 1), so the flip is the rounding's, not a bug.
 * Reported: how many differ, how many by more than one float16 step (only near zero, where float16 is finer than the
 * float32 rounding error of a log).
 */
void reportF16(const std::string& name, const std::vector<float>& ours, const npy::Array& ref, const std::vector<float>* ours32,
               const npy::Array* ref32)
{
    if (ours.size() != ref.size()) {
        std::printf("  %-18s SIZE MISMATCH ours %zu ref %zu\n", name.c_str(), ours.size(), ref.size());
        g_fail++;
        return;
    }
    size_t diff = 0, far = 0, bad = 0;
    for (size_t i = 0; i < ours.size(); i++) {
        if (ours[i] == static_cast<float>(ref.data[i])) {
            continue;
        }
        diff++;
        const ggml_fp16_t a = ggml_fp32_to_fp16(ours[i]), b = ggml_fp32_to_fp16(static_cast<float>(ref.data[i]));
        // adjacent float16 values: their bit patterns (same sign) differ by one
        const int ia = (a & 0x8000) ? -(a & 0x7fff) : (a & 0x7fff), ib = (b & 0x8000) ? -(b & 0x7fff) : (b & 0x7fff);
        if (std::abs(ia - ib) > 1) {
            far++;
        }
        if (ours32 != nullptr) {
            const double o = (*ours32)[i], r = ref32->data[i];
            if (std::abs(o - r) > g_tol * std::max(1.0, std::abs(r))) {
                bad++;
            }
        }
    }
    double ma, rm;
    const double r = relL2(ours.data(), ref.data.data(), ours.size(), &ma, &rm);
    const bool ok = bad == 0;
    if (ours32 != nullptr) {
        g_fail += ok ? 0 : 1;
    }
    std::printf("  %-18s %zu of %zu differ (%.4f%%), %zu by more than one float16 step; rel_l2 %.3e max_abs %.3e", name.c_str(),
                diff, ours.size(), 100.0 * diff / std::max<size_t>(1, ours.size()), far, r, ma);
    if (ours32 != nullptr) {
        std::printf("; %zu with float32 values further apart than the tolerance  %s\n", bad, ok ? "ok" : "FAIL");
    } else {
        std::printf("  (reported)\n");
    }
}

/** The (T, 280) float32 features before the float16 rounding, from the three blocks. */
std::vector<float> joinBlocks(const std::vector<const std::vector<float>*>& b, const std::vector<int>& w, int T)
{
    int C = 0;
    for (int x : w) {
        C += x;
    }
    std::vector<float> o(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; t++) {
        int off = 0;
        for (size_t k = 0; k < b.size(); k++) {
            std::copy_n(b[k]->data() + static_cast<size_t>(t) * w[k], w[k], o.data() + static_cast<size_t>(t) * C + off);
            off += w[k];
        }
    }
    return o;
}

npy::Array joinRefBlocks(const std::string& dir, int T)
{
    npy::Array o;
    std::vector<npy::Array> b = {npy::load(dir + "/lo.npy"), npy::load(dir + "/hi.npy"), npy::load(dir + "/cq.npy")};
    int C = 0;
    for (auto& x : b) {
        C += static_cast<int>(x.shape[1]);
    }
    o.shape = {T, C};
    o.data.resize(static_cast<size_t>(T) * C);
    for (int t = 0; t < T; t++) {
        int off = 0;
        for (auto& x : b) {
            const int w = static_cast<int>(x.shape[1]);
            std::copy_n(x.data.data() + static_cast<size_t>(t) * w, w, o.data.data() + static_cast<size_t>(t) * C + off);
            off += w;
        }
    }
    return o;
}

struct RefHits {
    std::vector<double> on, off, vel;
    std::vector<int> pitch;
};

RefHits loadHits(const std::string& dir)
{
    RefHits h;
    const auto on = npy::load(dir + "/hits_onset.npy"), off = npy::load(dir + "/hits_offset.npy"),
               p = npy::load(dir + "/hits_pitch.npy"), v = npy::load(dir + "/hits_velocity.npy");
    h.on = on.data;
    h.off = off.data;
    h.vel = v.data;
    for (double x : p.data) {
        h.pitch.push_back(static_cast<int>(x));
    }
    return h;
}

/** Exact: same list, element for element (onset/offset bit for bit, velocity as float). */
bool compareHits(const std::string& name, const std::vector<DrumHit>& ours, const RefHits& ref, bool required)
{
    size_t exact = 0;
    long first_bad = -1;
    for (size_t i = 0; i < std::min(ours.size(), ref.on.size()); i++) {
        const auto& h = ours[i];
        if (h.onset == ref.on[i] && h.offset == ref.off[i] && h.pitch == ref.pitch[i] && h.velocity == static_cast<float>(ref.vel[i])) {
            exact++;
        } else if (first_bad < 0) {
            first_bad = static_cast<long>(i);
        }
    }
    // order-free agreement: (pitch, onset) pairs in common
    std::multimap<std::pair<int, long>, size_t> idx;
    for (size_t i = 0; i < ref.on.size(); i++) {
        idx.emplace(std::make_pair(ref.pitch[i], std::lround(ref.on[i] * 1000)), i);
    }
    size_t common = 0, vel_eq = 0;
    for (const auto& h : ours) {
        auto it = idx.find({h.pitch, std::lround(h.onset * 1000)});
        if (it != idx.end()) {
            common++;
            vel_eq += h.velocity == static_cast<float>(ref.vel[it->second]) ? 1 : 0;
            idx.erase(it);
        }
    }
    const bool ok = exact == ours.size() && exact == ref.on.size();
    const double agree = static_cast<double>(common) / std::max<size_t>(1, std::max(ours.size(), ref.on.size()));
    if (required) {
        g_fail += (g_gpu ? agree >= 0.98 : ok) ? 0 : 1;
    }
    std::printf("  %-18s ours %zu, ref %zu, identical in order %zu; (pitch, onset) in common %zu (%.2f%% of ref), same velocity %zu  %s\n",
                name.c_str(), ours.size(), ref.on.size(), exact, common, 100.0 * common / std::max<size_t>(1, ref.on.size()), vel_eq,
                ok ? "exact" : (required ? (g_gpu ? (agree >= 0.98 ? "ok (GPU: >= 98%)" : "FAIL (GPU: < 98%)") : "MISMATCH") : "(reported)"));
    if (!ok && first_bad >= 0) {
        const auto& h = ours[first_bad];
        std::printf("    first difference at %ld: ours (%.4f, %d, %.3f) ref (%.4f, %d, %.3f)\n", first_bad, h.onset, h.pitch, h.velocity,
                    ref.on[first_bad], ref.pitch[first_bad], ref.vel[first_bad]);
    }
    return ok;
}

/**
 * Hits from our own features (Python's 16 kHz signal injected) vs the reference. Our float16 features differ from the
 * reference's in ~0.04% of elements by one float16 step (see reportF16), which moves the networks' outputs by ~1e-6:
 * a velocity can then round to the other side of a 3rd-decimal boundary. Required: the same hits in the same order
 * (onset, offset, key); a velocity may differ only by 0.001 and only where the reference's own unrounded value lies
 * within 1e-5 of the rounding boundary.
 */
void compareHitsF16(const std::string& name, const std::vector<DrumHit>& ours, const std::vector<DrumDecoder::Source>& src,
                    const RefHits& ref, const std::string& dir)
{
    const auto kv = exists(dir + "/P_kit_vel.npy") ? npy::load(dir + "/P_kit_vel.npy") : npy::Array{};
    const auto pv = npy::load(dir + "/P_vel.npy"), pt = npy::load(dir + "/P_timpani.npy");
    bool same_hits = ours.size() == ref.on.size();
    size_t vel_flip = 0, vel_bad = 0;
    double worst = 0;
    for (size_t i = 0; same_hits && i < ours.size(); i++) {
        const auto& h = ours[i];
        if (h.onset != ref.on[i] || h.offset != ref.off[i] || h.pitch != ref.pitch[i]) {
            same_hits = false;
            std::printf("    hit %zu differs: ours (%.4f, %d) ref (%.4f, %d)\n", i, h.onset, h.pitch, ref.on[i], ref.pitch[i]);
            break;
        }
        if (h.velocity == static_cast<float>(ref.vel[i])) {
            continue;
        }
        vel_flip++;
        const auto& s = src[i];
        const npy::Array& m = s.map == 0 ? kv : (s.map == 1 ? pv : pt);
        double r = m.data[static_cast<size_t>(s.frame) * m.shape[1] + s.lane];
        if (s.map == 2) {
            r = std::min(std::max(r, 0.3), 1.0);
        }
        const double boundary = (std::floor(r * 1000.0 - 0.5) + 0.5) / 1000.0, near = std::min(std::abs(r - boundary), std::abs(r - boundary - 0.001));
        worst = std::max(worst, near);
        const bool ok = near <= 1e-5 && std::abs(h.velocity - ref.vel[i]) <= 0.0010001;
        vel_bad += ok ? 0 : 1;
        std::printf("    hit %zu (%.4f s, key %d): velocity ours %.3f ref %.3f; unrounded ours %.7f ref %.7f, %.1e from the rounding boundary  %s\n",
                    i, h.onset, h.pitch, h.velocity, ref.vel[i], s.raw, r, near, ok ? "(float16 features)" : "FAIL");
    }
    const bool ok = same_hits && vel_bad == 0;
    g_fail += ok ? 0 : 1;
    std::printf("  %-18s ours %zu, ref %zu: %s; %zu velocities rounded across a boundary  %s\n", name.c_str(), ours.size(), ref.on.size(),
                same_hits ? "the same hits in the same order" : "DIFFERENT HITS", vel_flip, ok ? "ok" : "FAIL");
}

void printTiming(const DrumTiming& t, double total_ms = -1)
{
    const double total = total_ms >= 0 ? total_ms : t.total;
    std::printf("    time: total %.0f ms = features %.0f + embed %.0f + trunk %.0f + cq %.0f + adapter %.0f + host GRU %.0f + decode %.0f"
                " (+ %.0f other); compute buffer %.0f MB\n",
                total, t.features, t.embed, t.trunk, t.cq, t.adapter, t.host_gru, t.decode,
                total - t.features - t.embed - t.trunk - t.cq - t.adapter - t.host_gru - t.decode, t.compute_bytes / 1e6);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X] [--skip features,nets,e2e]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], dir = argv[2];
    const std::string skip = arg(argc, argv, "--skip", "");
    try {
        Backend be(arg(argc, argv, "--device", "cpu"), std::stoi(arg(argc, argv, "--threads", "0")));
        // --poll P: a persistent CPU threadpool with polling level P (0 = workers sleep at barriers), instead of ggml's
        // default per-graph pool (an experiment for the shared Backend: how the CPU backend copes with busy cores)
        ggml_threadpool_t pool = nullptr;
        if (be.isCpu() && arg(argc, argv, "--poll", "")[0] != '\0') {
            ggml_threadpool_params tp = ggml_threadpool_params_default(be.threads());
            tp.poll = static_cast<uint32_t>(std::stoi(arg(argc, argv, "--poll", "0")));
            pool = ggml_threadpool_new(&tp);
            ggml_backend_cpu_set_threadpool(be.get(), pool);
            std::printf("persistent threadpool, poll %u\n", tp.poll);
        }
        // the CPU standard: 1e-4 and exact hits; on a GPU (f16 weights, f16 matmul inputs) 1e-2 and >= 98% of the hits
        g_gpu = !be.isCpu();
        g_tol = std::stod(arg(argc, argv, "--tol", g_gpu ? "1e-2" : "1e-4"));
        std::printf("backend: %s (tolerance %.0e%s)\n", be.name().c_str(), g_tol, g_gpu ? ", hits: >= 98% in common" : ", hits: exact");
        ModelFile mf(model, be.get());
        std::printf("weights: %.1f MB\n", mf.bytes() / 1e6);
        DrumTranscriber dt(mf, be);
        if (arg(argc, argv, "--cq-chunk", "")[0] != '\0') {
            dt.nets().setCqChunk(std::stoi(arg(argc, argv, "--cq-chunk", "0"))); // force / disable the windowed cq branch
        }
        std::printf("cq branch: %s\n", dt.nets().cqChunk() > 0 ? format("windows of %d frames above %d frames", dt.nets().cqChunk(), dt.nets().cqChunk() * 3 / 2).c_str() : "one graph");
        const auto x44 = npy::load(dir + "/x44.npy").f32();
        const auto x16ref = npy::load(dir + "/x16.npy").f32();
        const auto feats_ref = npy::load(dir + "/feats.npy");
        const int Tref = static_cast<int>(feats_ref.shape[0]);
        std::printf("fixture: %s, %.1f s, T %d\n", dir.c_str(), x44.size() / 44100.0, Tref);

        // ---- 1. features
        if (skip.find("features") == std::string::npos) {
            std::printf("features (Python's 16 kHz signal injected):\n");
            DrumFeatures::Debug dbg;
            int T = 0;
            auto t0 = std::chrono::steady_clock::now();
            const auto f = dt.features().compute(x44.data(), x44.size(), T, &x16ref, &dbg, be.threads());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const double gref = npy::load(dir + "/gain.npy").data[0];
            const double gd = std::abs(dbg.gain - gref) / gref;
            g_fail += gd <= 1e-15 ? 0 : 1;
            std::printf("  %-18s ours %.17g ref %.17g (rel %.2e)  %s\n", "gain", dbg.gain, gref, gd, gd <= 1e-15 ? "ok" : "FAIL");
            if (T != Tref) {
                std::printf("  T: ours %d ref %d  FAIL\n", T, Tref);
                g_fail++;
            }
            const char* names[] = {"lo", "hi", "cq"};
            const std::vector<float>* blocks[] = {&dbg.lo, &dbg.hi, &dbg.cq};
            for (int b = 0; b < 3; b++) {
                const auto ref = npy::load(dir + "/" + names[b] + ".npy"); // untrimmed: compare the first T frames
                report(names[b], *blocks[b], ref, -1, std::min(blocks[b]->size(), ref.size()));
            }
            if (T == Tref) {
                const auto o32 = joinBlocks({&dbg.lo, &dbg.hi, &dbg.cq}, {static_cast<int>(dbg.lo.size() / T), static_cast<int>(dbg.hi.size() / T),
                                                                           static_cast<int>(dbg.cq.size() / T)}, T);
                const auto r32 = joinRefBlocks(dir, T);
                reportF16("feats (f16)", f, feats_ref, &o32, &r32);
            }
            std::printf("    (%.0f ms)\n", ms);

            std::printf("features (our resampler):\n");
            DrumFeatures::Debug dbg2;
            int T2 = 0;
            const auto f2 = dt.features().compute(x44.data(), x44.size(), T2, nullptr, &dbg2, be.threads());
            if (dbg2.x16.size() == x16ref.size()) {
                double ma, rm;
                const double r = relL2(dbg2.x16.data(), npy::load(dir + "/x16.npy").data.data(), x16ref.size(), &ma, &rm);
                std::printf("  %-18s rel_l2 %.3e  max_abs %.3e  (ref max %.3e)  (reported: a soxr stand-in)\n", "x16", r, ma, rm);
            } else {
                std::printf("  x16 length: ours %zu ref %zu  FAIL\n", dbg2.x16.size(), x16ref.size());
                g_fail++;
            }
            for (int b = 0; b < 3; b++) {
                const std::vector<float>* bl[] = {&dbg2.lo, &dbg2.hi, &dbg2.cq};
                const auto ref = npy::load(dir + "/" + names[b] + ".npy");
                double ma, rm;
                const double r = relL2(bl[b]->data(), ref.data.data(), std::min(bl[b]->size(), ref.size()), &ma, &rm);
                std::printf("  %-18s rel_l2 %.3e  max_abs %.3e  (reported)\n", names[b], r, ma);
            }
            if (T2 == Tref) {
                reportF16("feats (f16)", f2, feats_ref, nullptr, nullptr);
            }
            std::printf("    (resample %.0f ms, STFT 512 %.0f ms, STFT 1411 %.0f ms, STFT 6144 %.0f ms; %d threads)\n", dbg2.resample_ms,
                        dbg2.lo_ms, dbg2.hi_ms, dbg2.cq_ms, be.threads());
        }

        // ---- 2. networks on the reference features
        DrumProbs P;
        if (skip.find("nets") == std::string::npos) {
            std::printf("networks (reference features):\n");
            DrumTaps taps;
            taps.stem = exists(dir + "/kit_stem.npy");
            DrumTiming tm;
            const auto fr = feats_ref.f32();
            const auto t0 = std::chrono::steady_clock::now();
            P = dt.nets().run(fr.data(), Tref, true, &taps, &tm);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::vector<std::string> order;
            for (const char* m : {"kit", "ex0", "ex1", "ex2"}) {
                for (const char* s : {"stem", "embed", "tb", "gru", "h", "z", "v", "cqin", "cq1", "cq", "a", "r1", "rnn", "att", "r",
                                      "router", "drum", "timpani"}) {
                    order.push_back(std::string(m) + "_" + s);
                }
            }
            for (const auto& name : order) {
                auto it = taps.t.find(name);
                if (it == taps.t.end()) {
                    continue;
                }
                if (!exists(dir + "/" + name + ".npy")) {
                    std::printf("  %-18s (no reference)\n", name.c_str());
                    continue;
                }
                const auto ref = npy::load(dir + "/" + name + ".npy");
                if (ref.size() == 1 && it->second.size() == 1) {
                    // a router logit: one sum of ~64 terms of order 1, which may land near zero: its error is absolute
                    // (relative to max(|ref|, 1)), as is its effect through the sigmoid
                    const double e = std::abs(it->second[0] - ref.data[0]) / std::max(1.0, std::abs(ref.data[0]));
                    const bool ok = e <= g_tol;
                    g_fail += ok ? 0 : 1;
                    std::printf("  %-18s ours %.7f ref %.7f  |err| / max(|ref|, 1) %.3e  %s\n", name.c_str(), it->second[0], ref.data[0], e,
                                ok ? "ok" : "FAIL");
                    continue;
                }
                report(name, it->second, ref);
            }
            for (auto& [k, v] : std::vector<std::pair<std::string, const std::vector<float>*>>{
                     {"kit", &P.kit}, {"kit_vel", &P.kit_vel}, {"drum", &P.drum}, {"timpani", &P.timpani}, {"vel", &P.vel}}) {
                report("P_" + k, *v, npy::load(dir + "/P_" + k + ".npy"));
            }
            const double rref = npy::load(dir + "/P_router.npy").data[0];
            const double rd = std::abs(P.router - rref);
            g_fail += rd <= g_tol ? 0 : 1;
            std::printf("  %-18s ours %.9f ref %.9f (abs %.2e)  %s\n", "P_router", P.router, rref, rd, rd <= g_tol ? "ok" : "FAIL");
            std::printf("    (on the reference features, every model)\n");
            printTiming(tm, ms);
        }

        // ---- 3. hits
        if (skip.find("e2e") == std::string::npos) {
            std::printf("hits:\n");
            const RefHits ref = loadHits(dir);
            if (!P.drum.empty()) {
                std::string m;
                const auto h = dt.decoder().decode(P, m);
                compareHits("decode(P)", h, ref, true);
            }
            {
                // transcribe() step by step with Python's 16 kHz signal, keeping where each hit came from
                int T = 0;
                const auto f = dt.features().compute(x44.data(), x44.size(), T, &x16ref, nullptr, be.threads());
                const DrumProbs Pf = dt.nets().run(f.data(), T, false, nullptr, nullptr);
                std::string m;
                std::vector<DrumDecoder::Source> src;
                const auto h = dt.decoder().decode(Pf, m, &src);
                if (g_gpu) {
                    compareHits("transcribe(x16 in)", h, ref, true);
                } else {
                    compareHitsF16("transcribe(x16 in)", h, src, ref, dir);
                }
                std::printf("    router %.9f -> %s\n", Pf.router, m.c_str());
            }
            {
                // shorter than 0.1 s: no hits, router None (NaN here), as the reference
                DrumInfo i0;
                const auto h0 = dt.transcribe(x44.data(), std::min<size_t>(x44.size(), 4409), &i0);
                const bool ok = h0.empty() && std::isnan(i0.router) && i0.model.empty();
                g_fail += ok ? 0 : 1;
                std::printf("  %-18s %zu hits, router %f, model '%s'  %s\n", "transcribe(<0.1 s)", h0.size(), i0.router, i0.model.c_str(),
                            ok ? "ok" : "FAIL");
            }
            {
                // the pipeline's overload: its own 16 kHz signal of the same audio (here dtt::resample of x44, unscaled)
                const auto a16 = resample(x44.data(), x44.size(), 44100, 16000, be.threads());
                DrumInfo i3;
                const auto t0 = std::chrono::steady_clock::now();
                const auto h3 = dt.transcribe(x44.data(), x44.size(), a16.data(), a16.size(), &i3);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                compareHits("transcribe(+16k)", h3, ref, false);
                std::printf("    router %.6f -> %s  (%.0f ms; the 16 kHz signal given)\n", i3.router, i3.model.c_str(), ms);
            }
            DrumTiming tm2;
            DrumInfo info2;
            const auto h2 = dt.transcribe(x44.data(), x44.size(), &info2, nullptr, nullptr, &tm2, false);
            compareHits("transcribe()", h2, ref, false);
            std::printf("    router %.6f -> %s\n", info2.router, info2.model.c_str());
            printTiming(tm2);
        }
        if (pool != nullptr) {
            ggml_backend_cpu_set_threadpool(be.get(), nullptr);
            ggml_threadpool_free(pool);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
