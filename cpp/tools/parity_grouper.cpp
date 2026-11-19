// Parity of the C++ grouper against the PyTorch reference dumps (py/dump_refs_grouper.py).
//
//   dettato_parity_grouper MODEL.gguf REFS_GROUPER/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X]
//                          [--stages cases,tuning,window,features,store,predict,decode,e2e,sensitivity] [--variant f16_exactpcen]
//
// Stage by stage on reference inputs (tensors: relative L2 <= --tol; decode: exact), then end to end from the audio and
// the reference input notes (track assignment: adjusted Rand index, names, note counts). Exit status 0 when every
// checked item passes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "npy.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "dsp/resample.hpp"
#include "grouper/decode_check.hpp"
#include "grouper/front.hpp"
#include "grouper/grouper.hpp"
#include "grouper/notenet.hpp"
#include "grouper/retune.hpp"
#include "model_file.hpp"
#include "transcriber/transcriber.hpp"

using namespace dtt;
namespace fs = std::filesystem;

namespace
{
int g_fail = 0;
double g_tol = 1e-4;

struct Stat {
    double max_abs = 0, ref_max = 0, rel_l2 = 0;
};

Stat compare(const float* a, const double* ref, size_t n)
{
    Stat s;
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = a[i] - ref[i];
        s.max_abs = std::max(s.max_abs, std::abs(d));
        s.ref_max = std::max(s.ref_max, std::abs(ref[i]));
        num += d * d;
        den += ref[i] * ref[i];
    }
    s.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    return s;
}

/** Compares the first n values (all if n == 0). tol < 0: report only. */
double report(const std::string& name, const std::vector<float>& ours, const npy::Array& ref, double tol, size_t n = 0)
{
    if (n == 0) {
        if (ours.size() != ref.size()) {
            std::printf("  %-26s SIZE MISMATCH ours %zu ref %zu\n", name.c_str(), ours.size(), ref.size());
            g_fail++;
            return INFINITY;
        }
        n = ref.size();
    }
    if (ours.size() < n || ref.size() < n) {
        std::printf("  %-26s SIZE MISMATCH ours %zu ref %zu (need %zu)\n", name.c_str(), ours.size(), ref.size(), n);
        g_fail++;
        return INFINITY;
    }
    const Stat s = compare(ours.data(), ref.data.data(), n);
    const char* verdict = tol < 0 ? "(report)" : (s.rel_l2 <= tol ? "ok" : "FAIL");
    if (tol >= 0 && s.rel_l2 > tol) {
        g_fail++;
    }
    std::printf("  %-26s rel_l2 %.3e  max_abs %.3e  (ref max %.3e)  %s\n", name.c_str(), s.rel_l2, s.max_abs, s.ref_max, verdict);
    return s.rel_l2;
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

bool exists(const std::string& p)
{
    return fs::exists(p);
}

/** ours channels-last [C, P, T] -> torch (C, T, P). */
std::vector<float> clToTorch(const std::vector<float>& x, int64_t C, int64_t P, int64_t T)
{
    std::vector<float> o(static_cast<size_t>(C * P * T));
    for (int64_t c = 0; c < C; c++)
        for (int64_t t = 0; t < T; t++)
            for (int64_t p = 0; p < P; p++)
                o[(c * T + t) * P + p] = x[c + C * (p + P * t)];
    return o;
}

/** Adjusted Rand index of two labelings (sklearn's adjusted_rand_score). */
double ari(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    const size_t n = a.size();
    if (n != b.size() || n == 0) {
        return NAN;
    }
    std::map<std::pair<int64_t, int64_t>, double> nij;
    std::map<int64_t, double> ai, bj;
    for (size_t i = 0; i < n; i++) {
        nij[{a[i], b[i]}] += 1;
        ai[a[i]] += 1;
        bj[b[i]] += 1;
    }
    auto c2 = [](double x) { return x * (x - 1) / 2; };
    double s = 0, sa = 0, sb = 0;
    for (auto& [k, v] : nij) {
        s += c2(v);
    }
    for (auto& [k, v] : ai) {
        sa += c2(v);
    }
    for (auto& [k, v] : bj) {
        sb += c2(v);
    }
    if ((ai.size() == bj.size() && (ai.size() == 1 || ai.size() == n))) {
        return 1.0;
    }
    const double expct = sa * sb / c2(static_cast<double>(n)), mx = (sa + sb) / 2;
    return mx == expct ? 1.0 : (s - expct) / (mx - expct);
}

std::vector<int64_t> i64(const npy::Array& a)
{
    return std::vector<int64_t>(a.data.begin(), a.data.end());
}

/** "name count, name count, ..." of tracks. */
std::string trackList(const std::vector<GroupTrack>& t)
{
    std::string s;
    for (const GroupTrack& x : t) {
        s += format("%s%s %zu", s.empty() ? "" : ", ", x.instrument.c_str(), x.notes.size());
    }
    return s;
}

std::string refTrackList(const std::string& dir, const std::vector<std::string>& names)
{
    const auto inst = npy::load(dir + "/inst.npy"), off = npy::load(dir + "/off.npy");
    std::string s;
    for (size_t k = 0; k < inst.size(); k++) {
        s += format("%s%s %lld", s.empty() ? "" : ", ", names[static_cast<size_t>(inst.data[k])].c_str(),
                    static_cast<long long>(off.data[k + 1] - off.data[k]));
    }
    return s;
}

/** Our tracks vs a reference tracks/ dir: same names in order, same notes (times and pitch; velocity reported apart). */
void compareTracks(const char* what, const std::vector<GroupTrack>& ours, const std::string& dir, const std::vector<std::string>& names,
                   bool must_match)
{
    const auto inst = npy::load(dir + "/inst.npy"), off = npy::load(dir + "/off.npy"), notes = npy::load(dir + "/notes.npy");
    bool same = ours.size() == inst.size();
    size_t ntp = 0, nvel = 0, ntot = 0;
    double dt = 0, dvel = 0, svel = 0;
    for (size_t k = 0; same && k < ours.size(); k++) {
        same = names[static_cast<size_t>(inst.data[k])] == ours[k].instrument &&
               ours[k].notes.size() == static_cast<size_t>(off.data[k + 1] - off.data[k]);
        for (size_t j = 0; same && j < ours[k].notes.size(); j++) {
            const double* r = notes.data.data() + (static_cast<size_t>(off.data[k]) + j) * 4;
            double d = 0;
            for (int c = 0; c < 3; c++) {
                d = std::max(d, std::abs(ours[k].notes[j][c] - r[c]));
            }
            dt = std::max(dt, d);
            ntp += d <= 1e-9;
            const double dv = std::abs(ours[k].notes[j][3] - r[3]);
            nvel += dv == 0;
            dvel = std::max(dvel, dv);
            svel += dv;
            ntot++;
        }
    }
    const bool ok = same && ntp == ntot && nvel == ntot;
    if (must_match && !ok) {
        g_fail++;
    }
    if (!same) {
        std::printf("  %-26s tracks differ (names / note counts)%s\n", what, must_match ? "  FAIL" : "");
        return;
    }
    std::printf("  %-26s %s: onset/offset/pitch identical %zu/%zu (max diff %.1e s), velocity identical %zu/%zu (mean |d| %.2f, "
                "max %.0f)%s\n", what, ok ? "identical tracks" : "same tracks, notes differ", ntp, ntot, dt, nvel, ntot,
                svel / std::max<size_t>(ntot, 1), dvel, must_match ? (ok ? "  ok" : "  FAIL") : "");
}

struct Fixture {
    std::string dir;
    double r = 1.0;
    std::vector<float> audio, y;
    std::vector<std::array<double, 3>> notes;
};

Fixture loadFixture(const std::string& dir)
{
    Fixture f;
    f.dir = dir;
    f.audio = npy::load(dir + "/audio16k.npy").f32();
    const auto tun = npy::load(dir + "/tuning.npy");
    f.r = tun.data[3];
    f.y = exists(dir + "/retuned.npy") ? npy::load(dir + "/retuned.npy").f32() : f.audio;
    const auto ni = npy::load(dir + "/notes_in.npy");
    for (size_t i = 0; i < ni.shape[0]; i++) {
        f.notes.push_back({ni.data[3 * i], ni.data[3 * i + 1], ni.data[3 * i + 2]});
    }
    return f;
}

GroupFeatures refFeatures(const std::string& fd, bool f32acc)
{
    GroupFeatures g;
    const auto meta = npy::load(fd + "/meta.npy"), feat = npy::load(fd + (f32acc ? "/feat_f32acc.npy" : "/feat.npy")),
               fsc = npy::load(fd + (f32acc ? "/fsc_f32acc.npy" : "/fsc.npy"));
    g.n = static_cast<int>(meta.shape[0]);
    g.meta = meta.f32();
    g.feat.assign(feat.data.begin(), feat.data.end());
    g.fsc = fsc.f32();
    return g;
}

double secs(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf REFS_GROUPER/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X] "
                             "[--stages cases,tuning,window,features,store,predict,decode,e2e,sensitivity] [--variant NAME]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], dir = argv[2];
    g_tol = std::stod(arg(argc, argv, "--tol", "1e-4"));
    const std::string stages_s = arg(argc, argv, "--stages", "cases,tuning,window,features,predict,decode,e2e");
    std::set<std::string> stages;
    for (size_t a = 0, b; a < stages_s.size(); a = b + 1) {
        b = stages_s.find(',', a);
        if (b == std::string::npos) {
            b = stages_s.size();
        }
        stages.insert(stages_s.substr(a, b - a));
    }
    const std::string variant = arg(argc, argv, "--variant", "f16_exactpcen");
    try {
        Backend be(arg(argc, argv, "--device", "cpu"), std::stoi(arg(argc, argv, "--threads", "0")));
        const bool gpu = !be.isCpu();
        std::printf("backend: %s\n", be.name().c_str());
        ModelFile mf(model, be.get());
        std::printf("weights: %.1f MB\n", mf.bytes() / 1e6);
        Grouper gr(mf, be);
        const auto& names = gr.names();
        const Fixture fx = loadFixture(dir);
        std::printf("fixture %s: %.1f s, %zu input notes, r %.6f\n", dir.c_str(), fx.audio.size() / 16000.0, fx.notes.size(), fx.r);

        // ---------------------------------------------------------------- the decode's scipy / numpy ports (random cases)
        const std::string cases = (fs::path(dir).parent_path() / "cases").string();
        if (stages.count("cases") && exists(cases)) {
            std::printf("[cases] %s\n", cases.c_str());
            g_fail += decode_check::checkClusterCases(cases);
        }

        // ---------------------------------------------------------------- tuning + resampler
        if (stages.count("tuning")) {
            std::printf("[tuning]\n");
            const auto tun = npy::load(dir + "/tuning.npy");
            double delta, strength, shift, r;
            const auto t0 = std::chrono::steady_clock::now();
            retune::ratio(fx.audio.data(), fx.audio.size(), delta, strength, shift, r);
            const double ms = secs(t0) * 1e3;
            const double dd = std::abs(delta - tun.data[0]), ds = std::abs(strength - tun.data[1]);
            const bool ok = dd <= 1e-6 && ds <= 1e-7 && (shift != 0) == (tun.data[2] != 0) && std::abs(r - tun.data[3]) <= 1e-9;
            g_fail += ok ? 0 : 1;
            std::printf("  cents %.9f (ref %.9f, diff %.1e), strength %.9f (ref %.9f, diff %.1e), shift %.4f, r %.9f  %s  (%.0f ms)\n",
                        delta, tun.data[0], dd, strength, tun.data[1], ds, shift, r, ok ? "ok" : "FAIL", ms);
            if (exists(dir + "/retuned.npy")) {
                const auto t1 = std::chrono::steady_clock::now();
                const auto y = resample(fx.audio.data(), fx.audio.size(), 16000.0, 16000.0 / tun.data[3], be.threads());
                const double rms = secs(t1) * 1e3;
                std::printf("  resampled %zu samples (ref %zu) in %.0f ms\n", y.size(), fx.y.size(), rms);
                if (y.size() == fx.y.size()) {
                    report("retuned audio vs soxr HQ", y, npy::load(dir + "/retuned.npy"), -1);
                } else {
                    g_fail++;
                    std::printf("  retuned length differs  FAIL\n");
                }
            }
        }

        // ---------------------------------------------------------------- one window, stage by stage
        if (stages.count("window") && exists(dir + "/win/X.npy")) {
            std::printf("[window] %s/win\n", dir.c_str());
            const std::string wd = dir + "/win";
            const GroupFront& fr = gr.front();
            NoteNet& net = gr.net();
            const auto audio = npy::load(wd + "/audio.npy").f32();
            const int Tf = static_cast<int>(audio.size() / GroupFront::HOP);
            const int C = fr.channels(), F = fr.bins(), NE = fr.extBins();
            std::vector<float> X(static_cast<size_t>(C) * Tf * F), P(static_cast<size_t>(NE) * Tf);
            auto t0 = std::chrono::steady_clock::now();
            fr.logHcqt(audio.data(), audio.size(), X.data(), P.data());
            const double t_hcqt = secs(t0);
            report("X_log (log HCQT)", X, npy::load(wd + "/X_log.npy"), g_tol);
            report("P (linear power)", P, npy::load(wd + "/P.npy"), g_tol);
            t0 = std::chrono::steady_clock::now();
            fr.pcen(X.data(), C, Tf, F);
            const double t_pcen = secs(t0);
            report("PCEN (exact conv)", X, npy::load(wd + "/X_pcen_exact.npy"), g_tol);
            GroupFront::standardise(X.data(), C, Tf, F);
            report("X (exact PCEN, standardised)", X, npy::load(wd + "/X_exact.npy"), g_tol);
            report("X vs reference (FFT PCEN)", X, npy::load(wd + "/X.npy"), -1);
            std::printf("  (front, one thread: log HCQT %.0f ms, PCEN %.0f ms)\n", t_hcqt * 1e3, t_pcen * 1e3);

            // the network on the reference inputs: X and P of the reference front end, the collated notes and taps
            const auto Xr = npy::load(wd + "/X.npy").f32(), Pr = npy::load(wd + "/P.npy").f32();
            const auto notes = npy::load(wd + "/notes.npy"), mask = npy::load(wd + "/mask.npy"), feat = npy::load(wd + "/feat.npy");
            const int N = static_cast<int>(notes.shape[0]);
            int nr = 0;
            for (double m : mask.data) {
                nr += m != 0;
            }
            NoteNet::Inputs in;
            net.resize(in, 1, N, Tf, fr);
            const auto nt = notes.f32(), ft = feat.f32();
            t0 = std::chrono::steady_clock::now();
            net.fillSlot(in, 0, fr, nt.data(), nr, ft.data(), Xr.data(), Pr.data());
            const double t_fill = secs(t0);
            const size_t nd = static_cast<size_t>(nr) * net.d();
            report("hd_raw", in.hd, npy::load(wd + "/hd_raw.npy"), g_tol, static_cast<size_t>(nr) * net.hdIn());
            report("prof_raw", in.prof, npy::load(wd + "/prof_raw.npy"), g_tol, static_cast<size_t>(nr) * net.profIn());
            {
                const int Tq = net.frames(Tf), PL = net.lanes();
                std::vector<float> rt(in.R.size());
                for (int c = 0; c < 4; c++)
                    for (int t = 0; t < Tq; t++)
                        for (int p = 0; p < PL; p++)
                            rt[(static_cast<size_t>(c) * Tq + t) * PL + p] = in.R[(static_cast<size_t>(t) * PL + p) * 4 + c];
                report("rasters", rt, npy::load(wd + "/rasters.npy"), g_tol);
            }
            report("s15 standardised taps", in.s15, npy::load(wd + "/s15_in.npy"), g_tol, static_cast<size_t>(nr) * net.nTaps() * net.tapDim());
            t0 = std::chrono::steady_clock::now();
            net.run(in, true);
            const double t_net = secs(t0);
            t0 = std::chrono::steady_clock::now();
            net.run(in, true); // again, warm (graph built, buffers allocated, shaders compiled)
            const double t_net2 = secs(t0);
            std::printf("  window graph: %d nodes, compute buffer %.1f MB; readers %.0f ms, network %.0f ms first / %.0f ms warm\n",
                        net.nodes(), net.computeBytes() / 1e6, t_fill * 1e3, t_net * 1e3, t_net2 * 1e3);
            std::vector<float> v;
            std::vector<int64_t> ne;
            const double tol_net = gpu ? -1 : g_tol;
            for (const char* n : {"lf_y0", "lf_blk0", "lf_attn2", "lf_attn5"}) {
                net.debug(n, v, ne);
                report(n, clToTorch(v, ne[0], ne[1], ne[2]), npy::load(wd + "/" + n + ".npy"), tol_net);
            }
            for (const char* n : {"lf_at_on", "lf_mean"}) {
                net.debug(n, v, ne);
                report(n, v, npy::load(wd + "/" + n + ".npy"), tol_net, static_cast<size_t>(nr) * ne[0]);
            }
            for (const char* n : {"lanes", "hd", "prof", "x_note", "s15", "tok", "ln_in", "blk0", "blk1", "blk2", "blk3", "blk4", "blk5", "blk6"}) {
                net.debug(n, v, ne);
                report(n, v, npy::load(wd + "/" + n + ".npy"), tol_net, nd);
            }
            net.hidden(v);
            report("h (ln_out)", v, npy::load(wd + "/h.npy"), tol_net, nd);
            net.heads(v);
            const int K = net.nInst(), E = net.embDim(), OD = net.outDim();
            std::vector<float> inst(static_cast<size_t>(nr) * K), emb(static_cast<size_t>(nr) * E), keep(nr);
            for (int j = 0; j < nr; j++) {
                const float* o = v.data() + static_cast<size_t>(j) * OD;
                double ss = 0;
                for (int c = 0; c < E; c++) {
                    ss += static_cast<double>(o[K + c]) * o[K + c];
                }
                for (int c = 0; c < K; c++) {
                    inst[static_cast<size_t>(j) * K + c] = o[c];
                }
                for (int c = 0; c < E; c++) {
                    emb[static_cast<size_t>(j) * E + c] = static_cast<float>(o[K + c] / std::max(std::sqrt(ss), 1e-12));
                }
                keep[j] = o[K + E];
            }
            report("inst logits", inst, npy::load(wd + "/inst.npy"), tol_net, inst.size());
            report("emb", emb, npy::load(wd + "/emb.npy"), tol_net, emb.size());
            report("keep", keep, npy::load(wd + "/keep.npy"), tol_net, keep.size());
            // the argmax of the instrument logits
            const auto ir = npy::load(wd + "/inst.npy");
            int agree = 0;
            for (int j = 0; j < nr; j++) {
                const float* a = inst.data() + static_cast<size_t>(j) * K;
                const double* b = ir.data.data() + static_cast<size_t>(j) * K;
                agree += (std::max_element(a, a + K) - a) == (std::max_element(b, b + K) - b);
            }
            std::printf("  instrument argmax agreement %d / %d\n", agree, nr);
        }

        // ---------------------------------------------------------------- pass 2 + gather
        Transcriber tr(mf, be, 4, LaneNet::Taps{true, true, false});
        std::vector<std::array<double, 3>> nin = fx.notes;
        for (auto& q : nin) {
            q[0] /= fx.r;
            q[1] /= fx.r;
        }
        if (stages.count("features")) {
            std::printf("[features] pass 2 on the reference (retuned) audio + gather\n");
            const std::string fd = dir + "/feats";
            for (int f32 = 0; f32 < 2; f32++) {
                Grouper::Debug dbg;
                dbg.taps_f32 = f32 != 0;
                gr.debug = &dbg;
                const auto t0 = std::chrono::steady_clock::now();
                const GroupFeatures f = gr.features(fx.y.data(), fx.y.size(), nin, tr);
                const double s = secs(t0);
                gr.debug = nullptr;
                std::printf(" taps accumulated in %s (%.2f s):\n", f32 ? "float32" : "float16 as the reference", s);
                const double tol = gpu ? -1 : g_tol;
                if (!f32) {
                    const auto meta = npy::load(fd + "/meta.npy");
                    report("meta", f.meta, meta, tol);
                    size_t same = 0;
                    for (size_t i = 0; i < meta.size(); i++) {
                        same += static_cast<double>(f.meta[i]) == meta.data[i];
                    }
                    std::printf("  meta: %zu / %zu values bit-identical\n", same, meta.size());
                }
                report(f32 ? "X (float32 taps)" : "X (float16 taps)", f.X, npy::load(fd + (f32 ? "/X_f32acc.npy" : "/X.npy")), tol);
                const auto q = npy::load(fd + (f32 ? "/feat_f32acc.npy" : "/feat.npy"));
                size_t eq = 0, d1 = 0;
                int dmax = 0;
                for (size_t i = 0; i < q.size(); i++) {
                    const int d = std::abs(static_cast<int>(f.feat[i]) - static_cast<int>(q.data[i]));
                    eq += d == 0;
                    d1 += d == 1;
                    dmax = std::max(dmax, d);
                }
                std::printf("  int8 taps: %.5f equal, %.5f off by 1, max diff %d\n", static_cast<double>(eq) / q.size(),
                            static_cast<double>(d1) / q.size(), dmax);
                report("fsc", f.fsc, npy::load(fd + (f32 ? "/fsc_f32acc.npy" : "/fsc.npy")), tol);
                if (!f32) {
                    // the float16 reference vs our float32 accumulation, for scale
                    report("(ref float16 X vs ref float32 X)", npy::load(fd + "/X.npy").f32(), npy::load(fd + "/X_f32acc.npy"), -1);
                }
            }
        }

        // ---------------------------------------------------------------- single pass: features from pass 1's store
        if (stages.count("store")) {
            std::printf("[store] featuresFromStore (one transcriber pass on y) vs features() (its own hop-0.5 pass)\n");
            const unsigned taps = (1u << TapSink::HIDDEN) | (1u << TapSink::STEM);
            auto t0 = std::chrono::steady_clock::now();
            const GroupFeatures f0 = gr.features(fx.y.data(), fx.y.size(), nin, tr);
            std::printf("  features(): %.2f s (pass 2 + gather)\n", secs(t0));
            for (double hop : {0.5, 0.25}) {
                auto store = gr.makeStoreSink();
                Maps maps2;
                Transcriber::RunOptions ro;
                ro.hop_frac = hop;
                ro.sink = store.get();
                ro.tap_mask = taps;
                ro.sub_maps = &maps2;
                ro.sub_step = 64000;
                t0 = std::chrono::steady_clock::now();
                tr.run(fx.y.data(), fx.y.size(), ro);
                const double t_pass1 = secs(t0);
                Grouper::Debug dbg;
                gr.debug = &dbg;
                t0 = std::chrono::steady_clock::now();
                const GroupFeatures f1 = gr.featuresFromStore(fx.y.data(), fx.y.size(), nin, *store, maps2, tr);
                const double t_fs = secs(t0);
                gr.debug = nullptr;
                auto bitsEq = [](const std::vector<float>& a, const std::vector<float>& b) {
                    size_t k = 0;
                    for (size_t i = 0; i < std::min(a.size(), b.size()); i++) {
                        k += std::memcmp(&a[i], &b[i], sizeof(float)) == 0;
                    }
                    return k;
                };
                size_t feq = 0;
                for (size_t i = 0; i < std::min(f0.feat.size(), f1.feat.size()); i++) {
                    feq += f0.feat[i] == f1.feat[i];
                }
                const size_t meq = bitsEq(f0.meta, f1.meta), seq = bitsEq(f0.fsc, f1.fsc), xeq = bitsEq(f0.X, f1.X);
                const bool ok = f0.n == f1.n && meq == f0.meta.size() && feq == f0.feat.size() && seq == f0.fsc.size() &&
                                xeq == f0.X.size() && f1.meta.size() == f0.meta.size() && f1.feat.size() == f0.feat.size();
                g_fail += ok ? 0 : 1;
                std::printf("  pass 1 at hop %.2f: %.2f s; store: %d hop-0.5 windows used, weight sum %s, H pairs %zu, S pairs %zu, "
                            "kept %.1f MB, peak %.1f MB\n", hop, t_pass1, dbg.store_windows_seen,
                            dbg.store_wsum_matches ? "identical" : "DIFFERS", dbg.store_pairs_h, dbg.store_pairs_s,
                            dbg.store_bytes / 1e6, dbg.store_peak_bytes / 1e6);
                std::printf("    the notes read %zu H and %zu S pairs (%.0f%% / %.0f%% of the kept ones)\n", dbg.store_need_h, dbg.store_need_s,
                            100.0 * dbg.store_need_h / std::max<size_t>(dbg.store_pairs_h, 1),
                            100.0 * dbg.store_need_s / std::max<size_t>(dbg.store_pairs_s, 1));
                std::printf("    featuresFromStore %.2f s (gather %.3f s, recompute %.2f s): %d / %d notes recomputed over %d / %d windows\n",
                            t_fs, dbg.t_store_gather, dbg.t_store_recompute, dbg.store_notes_recomputed, f1.n,
                            dbg.store_windows_recomputed, dbg.store_windows_total);
                std::printf("    bit-identical to features(): meta %zu/%zu, feat %zu/%zu, fsc %zu/%zu, taps %zu/%zu  %s\n", meq,
                            f0.meta.size(), feq, f0.feat.size(), seq, f0.fsc.size(), xeq, f0.X.size(), ok ? "ok" : "FAIL");
            }
        }

        // ---------------------------------------------------------------- predict + decode on the reference features
        const std::string vdir = dir + "/" + variant;
        if (stages.count("predict") && exists(vdir + "/pred/probs.npy")) {
            std::printf("[predict] on the reference features (float16 taps); target: the %s variant (exact PCEN)\n", variant.c_str());
            const GroupFeatures f = refFeatures(dir + "/feats", variant == "exact");
            Grouper::Debug dbg;
            gr.debug = &dbg;
            const auto t0 = std::chrono::steady_clock::now();
            const GroupPredictions P = gr.predict(fx.y.data(), fx.y.size(), f);
            const double s = secs(t0);
            const auto tracks = gr.tracks(f, P, fx.r, &dbg.trace);
            gr.debug = nullptr;
            std::printf("  %zu windows, %.2f s\n", P.wins.size(), s);
            const double tol = gpu ? -1 : 1e-4;
            report("probs vs " + variant, P.probs, npy::load(vdir + "/pred/probs.npy"), tol);
            report("emb vs " + variant, P.emb, npy::load(vdir + "/pred/emb.npy"), tol);
            report("h vs " + variant, P.h, npy::load(vdir + "/pred/h.npy"), tol);
            report("probs vs ref (FFT PCEN)", P.probs, npy::load(dir + "/pred/probs.npy"), -1);
            const auto lv = i64(npy::load(vdir + "/tracks/labels_track.npy")), lr = i64(npy::load(dir + "/tracks/labels_track.npy"));
            const auto iv = i64(npy::load(vdir + "/tracks/labels_inst.npy")), ir = i64(npy::load(dir + "/tracks/labels_inst.npy"));
            auto same = [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
                size_t k = 0;
                for (size_t i = 0; i < a.size(); i++) {
                    k += a[i] == b[i];
                }
                return static_cast<double>(k) / std::max<size_t>(a.size(), 1);
            };
            {
                // where the decode of our predictions departs from the variant's: ARI of every intermediate labelling
                auto onIdx = [&](const std::vector<int64_t>& idx, const std::vector<int64_t>& lab, size_t n) {
                    std::vector<int64_t> full(n, -1);
                    for (size_t q = 0; q < idx.size() && q < lab.size(); q++) {
                        full[static_cast<size_t>(idx[q])] = lab[q];
                    }
                    return full;
                };
                const std::string dd = vdir + "/decode/";
                const auto ridx = i64(npy::load(dd + "idx.npy"));
                const size_t n = static_cast<size_t>(f.n);
                std::printf("  decode steps vs %s (ARI): tracklets %.4f, agglo %.4f, refined %.4f, folded %.4f, absorbed %.4f\n",
                            variant.c_str(), ari(dbg.trace.tracklets, i64(npy::load(dd + "tracklets.npy"))),
                            ari(dbg.trace.agglo, i64(npy::load(dd + "agglo.npy"))),
                            ari(onIdx(dbg.trace.idx, dbg.trace.refined, n), onIdx(ridx, i64(npy::load(dd + "refined.npy")), n)),
                            ari(onIdx(dbg.trace.idx, dbg.trace.folded, n), onIdx(ridx, i64(npy::load(dd + "folded.npy")), n)),
                            ari(onIdx(dbg.trace.idx, dbg.trace.absorbed, n), onIdx(ridx, i64(npy::load(dd + "absorbed.npy")), n)));
            }
            std::printf("  tracks: ARI vs %s %.4f (same instrument %.4f); vs ref %.4f (%.4f)\n", variant.c_str(),
                        ari(dbg.labels_track, lv), same(dbg.labels_inst, iv), ari(dbg.labels_track, lr), same(dbg.labels_inst, ir));
            std::printf("  ours:     %s\n  %-9s %s\n  ref:      %s\n", trackList(tracks).c_str(), (variant + ":").c_str(),
                        refTrackList(vdir + "/tracks", names).c_str(), refTrackList(dir + "/tracks", names).c_str());
            if (!gpu) {
                compareTracks(("tracks vs " + variant).c_str(), tracks, vdir + "/tracks", names, true);
            }
        }
        if (stages.count("sensitivity") && exists(vdir + "/pred/probs.npy")) {
            // how much the decode alone moves when the predictions move by GPU-sized amounts: the variant's predictions
            // with relative noise eps (probabilities, embeddings, h, the windows' values), decoded again
            const double eps = std::stod(arg(argc, argv, "--eps", "2e-4"));
            std::printf("[sensitivity] decode of the %s predictions with %.0e relative noise\n", variant.c_str(), eps);
            const GroupFeatures f = refFeatures(dir + "/feats", false);
            std::vector<double> on, off, pitch;
            const GroupPredictions P0 = decode_check::loadPredictions(vdir + "/pred", on, off, pitch);
            const auto lv = i64(npy::load(vdir + "/tracks/labels_track.npy"));
            std::mt19937_64 rng(1);
            std::normal_distribution<double> nd(0.0, 1.0);
            auto jitter = [&](std::vector<float>& v, int row, bool unit, bool half) {
                for (size_t r = 0; r * row < v.size(); r++) {
                    double ss = 0;
                    for (int c = 0; c < row; c++) {
                        float& x = v[r * row + c];
                        x = static_cast<float>(x * (1.0 + eps * nd(rng)) + (unit ? eps * 0.1 * nd(rng) : 0.0));
                        ss += static_cast<double>(x) * x;
                    }
                    for (int c = 0; c < row; c++) {
                        float& x = v[r * row + c];
                        if (unit) {
                            x = static_cast<float>(x / std::sqrt(ss));
                        }
                        if (half) {
                            x = roundHalf(x);
                        }
                    }
                }
            };
            double amin = 1, asum = 0;
            const int trials = 10;
            std::string sizes;
            for (int t = 0; t < trials; t++) {
                GroupPredictions P = P0;
                jitter(P.probs, P.K, false, false);
                jitter(P.emb, P.E, true, false);
                jitter(P.h, P.D, false, true);
                for (GroupWindow& w : P.wins) {
                    jitter(w.emb, P.E, true, true);
                    jitter(w.probs, P.K, false, true);
                }
                Grouper::Debug dbg;
                gr.debug = &dbg;
                const auto tracks = gr.tracks(f, P, fx.r);
                gr.debug = nullptr;
                const double a = ari(dbg.labels_track, lv);
                amin = std::min(amin, a);
                asum += a;
                sizes += format("  trial %d: ARI %.4f  %s\n", t, a, trackList(tracks).c_str());
            }
            std::printf("%s  ARI vs %s over %d trials: mean %.4f, min %.4f\n", sizes.c_str(), variant.c_str(), trials, asum / trials, amin);
        }
        if (stages.count("decode")) {
            for (const std::string v : {std::string(""), std::string("/f16_exactpcen"), std::string("/f32_fftpcen"), std::string("/exact")}) {
                if (exists(dir + v + "/pred/probs.npy")) {
                    std::printf("[decode] on the reference predictions of %s\n", (dir + v).c_str());
                    g_fail += decode_check::checkDecodeRefs(dir + v);
                }
            }
        }

        // ---------------------------------------------------------------- end to end
        if (stages.count("e2e")) {
            const bool retunes = exists(dir + "/retuned.npy");
            for (int inject = retunes ? 1 : 0; inject >= 0; inject--) {
                std::printf("[e2e] Grouper::group from the audio and the input notes%s\n",
                            retunes ? (inject ? " (Python's retuned audio injected)" : " (our resampler)") : "");
                Grouper::Debug dbg;
                if (inject) {
                    dbg.inject_retuned = true;
                    dbg.retuned = fx.y;
                    dbg.r = fx.r;
                }
                gr.debug = &dbg;
                GroupInfo info;
                const auto t0 = std::chrono::steady_clock::now();
                const auto tracks = gr.group(fx.audio.data(), fx.audio.size(), fx.notes, tr, &info);
                const double s = secs(t0);
                gr.debug = nullptr;
                std::printf("  %.2f s total: tuning %.2f, resample %.2f, pass 2 %.2f, predict %.2f, decode %.3f\n", s, dbg.t_tuning,
                            dbg.t_resample, dbg.t_pass2, dbg.t_predict, dbg.t_decode);
                std::printf("  network compute buffer %.0f MB (batch of up to 4 windows x %d notes)\n", gr.net().computeBytes() / 1e6,
                            gr.maxNotes());
                std::printf("  info: tuning %.2f c, strength %.3f, retune %.2f c, r %.6f, notes %d -> %d\n", info.tuning_cents,
                            info.tuning_strength, info.retune_cents, info.retune_ratio, info.notes_in, info.notes_out);
                const auto lr = i64(npy::load(dir + "/tracks/labels_track.npy")), ir = i64(npy::load(dir + "/tracks/labels_inst.npy"));
                const auto lv = i64(npy::load(vdir + "/tracks/labels_track.npy")), iv = i64(npy::load(vdir + "/tracks/labels_inst.npy"));
                size_t si = 0, sv = 0;
                for (size_t i = 0; i < lr.size() && i < dbg.labels_inst.size(); i++) {
                    si += dbg.labels_inst[i] == ir[i];
                    sv += dbg.labels_inst[i] == iv[i];
                }
                std::printf("  ARI vs ref %.4f (same instrument %.4f), vs %s %.4f (%.4f)\n", ari(dbg.labels_track, lr),
                            static_cast<double>(si) / lr.size(), variant.c_str(), ari(dbg.labels_track, lv), static_cast<double>(sv) / lv.size());
                std::printf("  ours:     %s\n  %-9s %s\n  ref:      %s\n", trackList(tracks).c_str(), (variant + ":").c_str(),
                            refTrackList(vdir + "/tracks", names).c_str(), refTrackList(dir + "/tracks", names).c_str());
                if (!gpu && (inject || !retunes)) {
                    compareTracks(("tracks vs " + variant).c_str(), tracks, vdir + "/tracks", names, false);
                }
                compareTracks("tracks vs ref", tracks, dir + "/tracks", names, false);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
