// Parity of the C++ evidence repairer against the PyTorch reference dumps (py/dump_refs_evidence.py).
//
//   dettato_parity_evidence MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X] [--report]
//                           [--maps DIR] [--skip-full]
//
// MODEL holds the "tr." and "ev." tensors (py/convert.py --stages transcriber,evidence). Sections:
//   host     the host-side pieces on the reference's own inputs: _std_q / evidence assembly, input_notes, the windows,
//            candidates from the reference logits, edits / apply from the reference candidates (all exact)
//   model    the network on the reference batches: one window stage by stage, then every window's outputs and what
//            candidates / apply make of them
//   full     the C++ transcriber's pass 1 with the evidence sink, then repair(): notes, evidence, repaired notes, against
//            the f32-accumulation variant (exact on the CPU) and the package's float16 one (reported)
// Exit status 0 when every check passes. --report: print the comparisons without failing on them (the GPU runs).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "backend.hpp"
#include "common.hpp"
#include "evidence/evidence.hpp"
#include "evidence/features.hpp"
#include "evidence/hidden_store.hpp"
#include "evidence/net.hpp"
#include "evidence/repair.hpp"
#include "model_file.hpp"
#include "npy.hpp"
#include "transcriber/transcriber.hpp"

using namespace dtt;
using namespace dtt::ev;

namespace
{
int g_fail = 0;
double g_tol = 1e-4;
bool g_report = false;
bool g_verbose = false;

void check(bool ok, const char* what)
{
    if (!ok && !g_report) {
        g_fail++;
    }
    if (!ok) {
        std::printf("    ^ %s\n", what);
    }
}

double relL2(const float* a, const double* ref, size_t n, double* max_abs = nullptr)
{
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = a[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
        mx = std::max(mx, std::abs(d));
    }
    if (max_abs) {
        *max_abs = mx;
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

void tensor(const std::string& name, const std::vector<float>& ours, const npy::Array& ref, double tol = -1)
{
    if (tol < 0) {
        tol = g_tol;
    }
    if (ours.size() != ref.size()) {
        std::printf("  %-26s SIZE MISMATCH ours %zu ref %zu\n", name.c_str(), ours.size(), ref.size());
        check(false, "size");
        return;
    }
    double mx;
    const double r = relL2(ours.data(), ref.data.data(), ref.size(), &mx);
    const bool ok = r <= tol;
    std::printf("  %-26s rel_l2 %.3e  max_abs %.3e  %s\n", name.c_str(), r, mx, ok ? "ok" : "FAIL");
    check(ok, "tolerance");
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
bool flag(int argc, char** argv, const char* key)
{
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], key) == 0) {
            return true;
        }
    }
    return false;
}

struct Ref {
    std::string dir;
    npy::Array get(const std::string& n) const { return npy::load(dir + "/" + n + ".npy"); }
    bool has(const std::string& n) const { return std::ifstream(dir + "/" + n + ".npy").good(); }
};

std::vector<TranscribedNote> refNotes(const Ref& r)
{
    const auto on = r.get("notes_onset"), off = r.get("notes_offset"), p = r.get("notes_pitch"), c = r.get("notes_conf"),
               s = r.get("notes_slot");
    std::vector<TranscribedNote> v(on.size());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = {on.data[i], off.data[i], static_cast<int>(p.data[i]), static_cast<float>(c.data[i]), static_cast<int>(s.data[i])};
    }
    return v;
}

std::vector<InNote> refK(const Ref& r)
{
    const auto p = r.get("k_pitch"), v = r.get("k_vel"), on = r.get("k_on"), off = r.get("k_off");
    std::vector<InNote> k(p.size());
    for (size_t i = 0; i < k.size(); i++) {
        k[i] = {static_cast<int>(p.data[i]), static_cast<int>(v.data[i]), static_cast<float>(on.data[i]), static_cast<float>(off.data[i])};
    }
    return k;
}

std::vector<WindowSpan> refWins(const Ref& r, const std::string& v)
{
    const auto w = r.get(v + "_win");
    std::vector<WindowSpan> out(w.shape.empty() ? 0 : static_cast<size_t>(w.shape[0]));
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = {w.data[3 * i], static_cast<int>(w.data[3 * i + 1]), static_cast<int>(w.data[3 * i + 2])};
    }
    return out;
}

/** The reference candidates of window i (offsets into the concatenation). */
std::vector<Cand> refCands(const Ref& r, const std::string& v, const std::vector<WindowSpan>& wins, std::vector<std::vector<Cand>>& per)
{
    const auto p = r.get(v + "_cand_pitch"), on = r.get(v + "_cand_on"), off = r.get(v + "_cand_off"), sc = r.get(v + "_cand_score"),
               kd = r.get(v + "_cand_kind");
    std::vector<Cand> all(p.size());
    for (size_t i = 0; i < all.size(); i++) {
        all[i] = {static_cast<int>(p.data[i]), static_cast<float>(on.data[i]), static_cast<float>(off.data[i]),
                  static_cast<float>(sc.data[i]), static_cast<int>(kd.data[i])};
    }
    per.clear();
    size_t o = 0;
    for (const auto& w : wins) {
        const size_t m = static_cast<size_t>(w.hi - w.lo) * 3;
        per.emplace_back(all.begin() + o, all.begin() + o + m);
        o += m;
    }
    return all;
}

/** Window batch from the input notes + an evidence matrix in transcriber-note order. */
void batch(const std::vector<InNote>& k, const std::vector<int>& order, const float* ev, const WindowSpan& w, std::vector<int>& pitch,
           std::vector<int>& vel, std::vector<float>& on, std::vector<float>& dur, std::vector<float>& evw)
{
    const int n = w.hi - w.lo;
    pitch.resize(n);
    vel.resize(n);
    on.resize(n);
    dur.resize(n);
    evw.resize(static_cast<size_t>(n) * N_EV);
    for (int j = 0; j < n; j++) {
        const InNote& x = k[w.lo + j];
        pitch[j] = x.pitch;
        vel[j] = x.vel;
        on[j] = x.on - static_cast<float>(w.t0);
        dur[j] = std::max(x.off - x.on, 0.0f);
        std::memcpy(evw.data() + static_cast<size_t>(j) * N_EV, ev + static_cast<size_t>(order[w.lo + j]) * N_EV, N_EV * sizeof(float));
    }
}

/** Counts candidate differences: (kept notes, insertions) with a different onset / end / pitch, and the max score diff. */
struct CandDiff {
    size_t n = 0, kept_diff = 0, ins_diff = 0;
    double score_max = 0;
};
CandDiff diffCands(const std::vector<Cand>& a, const std::vector<Cand>& b, int n)
{
    CandDiff d;
    d.n = a.size();
    for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        const bool same = a[i].pitch == b[i].pitch && a[i].on == b[i].on && a[i].off == b[i].off && a[i].kind == b[i].kind;
        if (!same) {
            (static_cast<int>(i) < n ? d.kept_diff : d.ins_diff)++;
        }
        d.score_max = std::max(d.score_max, static_cast<double>(std::abs(a[i].score - b[i].score)));
    }
    return d;
}

/**
 * Compares repaired notes with a reference variant. Prints, for a one-to-one matching on (pitch, inserted): exact, within
 * 1 ms (onset and end; small numeric shifts), onset within 1 ms but a different end, onset within 30 ms, unmatched.
 * Returns the number of notes not matched exactly.
 */
size_t compareRepaired(const std::vector<RepairedNote>& ours, const RepairInfo& info, const Ref& r, const std::string& v, const char* label)
{
    const auto on = r.get(v + "_rep_onset"), off = r.get(v + "_rep_offset"), p = r.get(v + "_rep_pitch"), ins = r.get(v + "_rep_inserted"),
               sc = r.get(v + "_rep_score");
    size_t same_order = 0;
    double d_score = 0;
    if (ours.size() == on.size()) {
        for (size_t i = 0; i < ours.size(); i++) {
            const bool eq = ours[i].onset == on.data[i] && ours[i].offset == off.data[i] && ours[i].pitch == static_cast<int>(p.data[i]) &&
                            ours[i].inserted == (ins.data[i] != 0);
            same_order += eq;
            if (eq) {
                d_score = std::max(d_score, std::abs(static_cast<double>(ours[i].score) - sc.data[i]));
            }
        }
    }
    std::multimap<std::pair<int, int>, size_t> idx;
    for (size_t i = 0; i < on.size(); i++) {
        idx.emplace(std::make_pair(static_cast<int>(p.data[i]), ins.data[i] != 0 ? 1 : 0), i);
    }
    std::vector<char> used(on.size(), 0), done(ours.size(), 0);
    size_t cnt[4] = {0, 0, 0, 0};
    // passes of increasing tolerance: exact, 1 ms both ends, 1 ms onset, 30 ms onset
    for (int pass = 0; pass < 4; pass++) {
        for (size_t q = 0; q < ours.size(); q++) {
            if (done[q]) {
                continue;
            }
            const RepairedNote& n = ours[q];
            auto [a, b] = idx.equal_range({n.pitch, n.inserted ? 1 : 0});
            for (auto it = a; it != b; ++it) {
                const size_t i = it->second;
                if (used[i]) {
                    continue;
                }
                const double don = std::abs(n.onset - on.data[i]), doff = std::abs(n.offset - off.data[i]);
                const bool ok = pass == 0 ? (don == 0 && doff == 0) : pass == 1 ? (don <= 1e-3 + 1e-9 && doff <= 1e-3 + 1e-9)
                                : pass == 2 ? don <= 1e-3 + 1e-9 : don < 0.03;
                if (ok) {
                    used[i] = 1;
                    done[q] = 1;
                    cnt[pass]++;
                    break;
                }
            }
        }
    }
    const size_t m = cnt[0] + cnt[1] + cnt[2] + cnt[3];
    if (g_verbose) {
        // the notes not matched exactly, both sides
        std::vector<char> ex_used(on.size(), 0);
        for (size_t q = 0; q < ours.size(); q++) {
            bool hit = false;
            for (size_t i = 0; i < on.size() && !hit; i++) {
                if (!ex_used[i] && ours[q].pitch == p.data[i] && ours[q].inserted == (ins.data[i] != 0) && ours[q].onset == on.data[i] &&
                    ours[q].offset == off.data[i]) {
                    ex_used[i] = 1;
                    hit = true;
                }
            }
            if (!hit) {
                std::printf("      ours #%zu: pitch %d %s %.4f-%.4f score %.4f\n", q, ours[q].pitch, ours[q].inserted ? "ins " : "kept",
                            ours[q].onset, ours[q].offset, ours[q].score);
            }
        }
        for (size_t i = 0; i < on.size(); i++) {
            if (!ex_used[i]) {
                std::printf("      ref  #%zu: pitch %d %s %.4f-%.4f score %.4f\n", i, static_cast<int>(p.data[i]), ins.data[i] ? "ins " : "kept",
                            on.data[i], off.data[i], sc.data[i]);
            }
        }
    }
    std::printf("  %-26s ours %zu (kept %d, ins %d, dropped %d), ref %zu; in order %zu; exact %zu, <=1 ms %zu, onset <=1 ms "
                "(end differs) %zu, onset <30 ms %zu, unmatched ours %zu / ref %zu; max d_score (exact) %.2e\n",
                label, ours.size(), info.kept, info.inserted, info.dropped, on.size(), same_order, cnt[0], cnt[1], cnt[2], cnt[3],
                ours.size() - m, on.size() - m, d_score);
    return std::max(ours.size(), on.size()) - cnt[0];
}

double now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf REFS/<fixture> [--device cpu|gpu|N] [--threads N] [--tol X] [--report] [--maps DIR] "
                     "[--skip-full]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const Ref R{argv[2]};
    g_tol = std::stod(arg(argc, argv, "--tol", "1e-4"));
    g_report = flag(argc, argv, "--report");
    g_verbose = flag(argc, argv, "--verbose");
    try {
        Backend be(arg(argc, argv, "--device", "cpu"), std::stoi(arg(argc, argv, "--threads", "0")));
        std::printf("backend: %s\n", be.name().c_str());
        ModelFile mf(model, be.get());
        std::printf("weights: %.1f MB\n", mf.bytes() / 1e6);
        EvidenceRepairer rep(mf, be);
        RepairerNet& net = rep.net();
        const double keep_th = mf.f32("ev.keep_th"), insert_th = mf.f32("ev.insert_th");

        const auto tnotes = refNotes(R);
        const auto order_ref = R.get("k_order");
        std::vector<int> order_r(order_ref.size());
        for (size_t i = 0; i < order_r.size(); i++) {
            order_r[i] = static_cast<int>(order_ref.data[i]);
        }
        const auto kref = refK(R);
        const auto ev16 = R.get("ev16"), ev32 = R.get("ev32");
        const auto ev16f = ev16.f32(), ev32f = ev32.f32();
        const size_t N = tnotes.size();
        std::printf("fixture %s: %zu transcribed notes\n", R.dir.c_str(), N);

        // ================================================================== host pieces on reference inputs
        std::printf("[host]\n");
        {
            // _std_q + assembly from the reference map numbers and gathered vectors -> the reference evidence, exactly
            for (const char* v : {"32", "16"}) {
                Gathered g;
                g.at = R.get(std::string("at") + v).f32();
                g.mean = R.get(std::string("mean") + v).f32();
                const auto ev = assembleEvidence(R.get("mapf").f32(), g);
                const auto& ref = std::string(v) == "32" ? ev32 : ev16;
                size_t bad_map = 0, bad_q = 0;
                for (size_t i = 0; i < N; i++) {
                    for (int j = 0; j < N_EV; j++) {
                        if (ev[i * N_EV + j] != static_cast<float>(ref.data[i * N_EV + j])) {
                            if (bad_map + bad_q < 5) {
                                std::printf("    note %zu column %d: ours %.9g ref %.9g\n", i, j, ev[i * N_EV + j], ref.data[i * N_EV + j]);
                            }
                            (j < N_MAP ? bad_map : bad_q)++;
                        }
                    }
                }
                std::printf("  evidence assembly (f%s)    map / log-spread numbers differing %zu of %zu, int8 steps differing %zu of %zu  %s\n",
                            v, bad_map, N * N_MAP, bad_q, N * 2 * C_H, bad_map + bad_q == 0 ? "ok" : "FAIL");
                check(bad_map + bad_q == 0, "_std_q / assembly not exact");
            }
            // input_notes
            std::vector<InNote> k;
            std::vector<int> order;
            inputNotes(tnotes, k, order);
            size_t bad = k.size() != kref.size();
            for (size_t i = 0; i < std::min(k.size(), kref.size()); i++) {
                bad += k[i].pitch != kref[i].pitch || k[i].vel != kref[i].vel || k[i].on != kref[i].on || k[i].off != kref[i].off ||
                       order[i] != order_r[i];
            }
            std::printf("  input_notes                %zu notes, %zu differ  %s\n", k.size(), bad, bad == 0 ? "ok" : "FAIL");
            check(bad == 0, "input_notes");
            const auto wins = windowSpans(kref, net.maxWindow());
            const auto wref = refWins(R, "f16");
            size_t wbad = wins.size() != wref.size();
            for (size_t i = 0; i < std::min(wins.size(), wref.size()); i++) {
                wbad += wins[i].t0 != wref[i].t0 || wins[i].lo != wref[i].lo || wins[i].hi != wref[i].hi;
            }
            std::printf("  windows                    %zu, %zu differ  %s\n", wins.size(), wbad, wbad == 0 ? "ok" : "FAIL");
            check(wbad == 0, "windows");
            // candidates from the reference logits, and edits / apply from the reference candidates
            for (const char* v : {"f16", "f32"}) {
                const auto keep = R.get(std::string(v) + "_out_keep"), don = R.get(std::string(v) + "_out_don"),
                           dur = R.get(std::string(v) + "_out_dur"), ins = R.get(std::string(v) + "_out_ins");
                const auto wv = refWins(R, v);
                std::vector<std::vector<Cand>> per;
                refCands(R, v, wv, per);
                size_t row = 0, kd = 0, id = 0, tot = 0;
                double smax = 0;
                std::vector<float> lg;
                std::vector<Cand> c;
                Edits e_ref(kref.size()), e_ours(kref.size());
                for (size_t wi = 0; wi < wv.size(); wi++) {
                    const auto& w = wv[wi];
                    const int n = w.hi - w.lo;
                    lg.assign(static_cast<size_t>(n) * RepairerNet::N_OUT, 0.0f);
                    for (int j = 0; j < n; j++) {
                        float* o = lg.data() + static_cast<size_t>(j) * RepairerNet::N_OUT;
                        o[0] = static_cast<float>(keep.data[row + j]);
                        for (int q = 0; q < RepairerNet::N_DON; q++)
                            o[1 + q] = static_cast<float>(don.data[(row + j) * RepairerNet::N_DON + q]);
                        for (int q = 0; q < RepairerNet::N_DR; q++)
                            o[1 + RepairerNet::N_DON + q] = static_cast<float>(dur.data[(row + j) * RepairerNet::N_DR + q]);
                        for (int q = 0; q < 2 * RepairerNet::N_INS; q++)
                            o[1 + RepairerNet::N_DON + RepairerNet::N_DR + q] = static_cast<float>(ins.data[(row + j) * 2 * RepairerNet::N_INS + q]);
                    }
                    std::vector<int> pitch, vel;
                    std::vector<float> on, du, evw;
                    batch(kref, order_r, ev16f.data(), w, pitch, vel, on, du, evw);
                    candidates(lg.data(), n, pitch.data(), on.data(), du.data(), net, c);
                    const CandDiff d = diffCands(c, per[wi], n);
                    if (flag(argc, argv, "--verbose")) {
                        for (int j = 0; j < n; j++) {
                            const Cand &a = c[j], &b = per[wi][j];
                            if (a.on == b.on && a.off == b.off) {
                                continue;
                            }
                            const float* o = lg.data() + static_cast<size_t>(j) * RepairerNet::N_OUT;
                            std::vector<float> pr(RepairerNet::N_DON);
                            const bool don_differs = a.on != b.on;
                            const int nb = don_differs ? RepairerNet::N_DON : RepairerNet::N_DR, h = don_differs ? 10 : 3;
                            softmaxRow(o + (don_differs ? 1 : 1 + RepairerNet::N_DON), nb, pr.data());
                            const int ours_b = boxArgmax(pr.data(), nb, h);
                            std::printf("    %s window %zu note %d: ours on %.7f off %.7f, ref on %.7f off %.7f; box sums:", v, wi, j, a.on, a.off,
                                        b.on, b.off);
                            for (int i = std::max(0, ours_b - 3); i <= std::min(nb - 1, ours_b + 3); i++) {
                                float sum = 0.0f;
                                for (int q = std::max(0, i - h); q <= std::min(nb - 1, i + h); q++) {
                                    sum += pr[q];
                                }
                                std::printf(" [%d]%.9g", i, sum);
                            }
                            std::printf("\n");
                        }
                    }
                    kd += d.kept_diff;
                    id += d.ins_diff;
                    tot += d.n;
                    smax = std::max(smax, d.score_max);
                    addEdits(w, per[wi], e_ref);
                    addEdits(w, c, e_ours);
                    row += n;
                }
                std::printf("  candidates from %s logits  %zu: kept notes differing %zu, insertions differing %zu, max d_score %.2e  %s\n", v,
                            tot, kd, id, smax, kd + id == 0 ? "ok" : "FAIL");
                check(kd + id == 0, "candidates from reference logits");
                // edits arrays from the reference candidates
                const auto pk = R.get(std::string(v) + "_edit_pk"), eon = R.get(std::string(v) + "_edit_on"),
                           eoff = R.get(std::string(v) + "_edit_off"), rins = R.get(std::string(v) + "_ins");
                size_t ebad = 0;
                auto same = [](double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; };
                for (size_t i = 0; i < kref.size(); i++) {
                    ebad += !same(e_ref.pk[i], pk.data[i]) || !same(e_ref.on_r[i], eon.data[i]) || !same(e_ref.off_r[i], eoff.data[i]);
                }
                ebad += e_ref.ins.size() * 4 != rins.size();
                for (size_t i = 0; i < std::min(e_ref.ins.size(), rins.size() / 4); i++) {
                    for (int q = 0; q < 4; q++) {
                        ebad += e_ref.ins[i][q] != rins.data[4 * i + q];
                    }
                }
                std::printf("  edits from %s candidates   %zu values differ  %s\n", v, ebad, ebad == 0 ? "ok" : "FAIL");
                check(ebad == 0, "edits");
                RepairInfo info;
                const auto out = applyEdits(kref, e_ref, keep_th, insert_th, &info);
                const size_t bad_out = compareRepaired(out, info, R, v, (std::string("apply (") + v + " cands)").c_str());
                check(bad_out == 0, "apply from reference candidates");
                const auto out2 = applyEdits(kref, e_ours, keep_th, insert_th, &info);
                const size_t bad2 = compareRepaired(out2, info, R, v, (std::string("apply (") + v + " logits)").c_str());
                check(bad2 == 0, "apply from candidates of the reference logits");
            }
            // what the float16 accumulation changes in Python itself: the f32 variant's repaired notes against the package's
            {
                const auto on = R.get("f32_rep_onset"), off = R.get("f32_rep_offset"), p = R.get("f32_rep_pitch"), ins = R.get("f32_rep_inserted"),
                           sc = R.get("f32_rep_score");
                std::vector<RepairedNote> v32(on.size());
                RepairInfo inf;
                for (size_t i = 0; i < v32.size(); i++) {
                    v32[i] = {on.data[i], off.data[i], static_cast<int>(p.data[i]), ins.data[i] != 0, static_cast<float>(sc.data[i])};
                    (v32[i].inserted ? inf.inserted : inf.kept)++;
                }
                inf.transcribed = static_cast<int>(N);
                inf.dropped = inf.transcribed - inf.kept;
                compareRepaired(v32, inf, R, "f16", "python: f32 vs f16 itself");
            }
            // map evidence from the reference maps, when available (the 30 s fixture's transcriber dump)
            const std::string maps_dir = arg(argc, argv, "--maps", "");
            if (!maps_dir.empty()) {
                Maps m;
                const char* heads[] = {"onset", "frame", "offset", "onset2", "frame2", "offset2"};
                for (int h = 0; h < N_HEADS; h++) {
                    const auto a = npy::load(maps_dir + "/map_" + heads[h] + ".npy");
                    m.T = static_cast<int>(a.shape[0]);
                    m.L = static_cast<int>(a.shape[1]);
                    m.m[h] = a.f32();
                }
                const auto spn = noteSpans(tnotes, m.T);
                const auto L = R.get("span_L"), A = R.get("span_A"), B = R.get("span_B"), S = R.get("span_S");
                size_t sbad = 0;
                for (size_t i = 0; i < N; i++) {
                    sbad += spn[i].L != L.data[i] || spn[i].A != A.data[i] || spn[i].B != B.data[i] || spn[i].S != S.data[i];
                }
                const auto mf31 = mapEvidence(m, spn);
                const auto mref = R.get("mapf");
                size_t mbad = 0;
                for (size_t i = 0; i < mf31.size(); i++) {
                    mbad += mf31[i] != static_cast<float>(mref.data[i]);
                }
                std::printf("  spans / map_evidence       (reference maps) spans differing %zu, map numbers differing %zu of %zu  %s\n", sbad,
                            mbad, mf31.size(), sbad + mbad == 0 ? "ok" : "FAIL");
                check(sbad + mbad == 0, "spans / map_evidence on the reference maps");
            }
        }

        // ================================================================== the network on reference batches
        std::printf("[model]\n");
        {
            // one window, stage by stage
            const Ref W{R.dir + "/win"};
            if (W.has("ev")) {
                const auto pi = W.get("pitch"), ve = W.get("vel"), on = W.get("on"), du = W.get("dur");
                const int n = static_cast<int>(pi.size());
                std::vector<int> pitch(n), vel(n);
                for (int i = 0; i < n; i++) {
                    pitch[i] = static_cast<int>(pi.data[i]);
                    vel[i] = static_cast<int>(ve.data[i]);
                }
                const auto onf = on.f32(), duf = du.f32(), evf = W.get("ev").f32();
                std::vector<float> cf(static_cast<size_t>(n) * RepairerNet::N_CONT);
                net.contFeatures(n, onf.data(), duf.data(), pitch.data(), cf.data());
                const auto cref = W.get("cont_in");
                std::printf("  window %s: %d notes (bucket %d)\n", W.dir.c_str(), n, RepairerNet::bucket(n));
                tensor("cont features (110)", cf, cref);
                // per feature group: fourier on / ld / end, ld & dur, context (8), grid (28)
                const int groups[][2] = {{0, 24}, {24, 48}, {48, 72}, {72, 74}, {74, 82}, {82, 110}};
                const char* gname[] = {"fourier(on)", "fourier(log dur)", "fourier(end)", "log dur, dur", "context", "grid"};
                for (int g = 0; g < 6; g++) {
                    double mx = 0;
                    for (int i = 0; i < n; i++)
                        for (int j = groups[g][0]; j < groups[g][1]; j++)
                            mx = std::max(mx, std::abs(cf[static_cast<size_t>(i) * 110 + j] - cref.data[static_cast<size_t>(i) * 110 + j]));
                    std::printf("    %-20s max_abs %.3e\n", gname[g], mx);
                }
                net.setDebug(true);
                std::vector<float> out, v;
                net.run(n, pitch.data(), vel.data(), onf.data(), duf.data(), evf.data(), out);
                net.debug("emb", v);
                tensor("NoteEmbed", v, W.get("emb"));
                for (int i = 0; i < net.layers(); i++) {
                    net.debug(format("block_%d", i), v);
                    tensor(format("block %d", i), v, W.get(format("block_%d", i)));
                }
                net.debug("bb", v);
                const auto nw = mf.hostF32("ev.bb_norm_w");
                for (size_t i = 0; i < v.size(); i++) {
                    v[i] *= nw[i % nw.size()];
                }
                tensor("backbone (normed)", v, W.get("bb"));
                std::vector<float> keep(n), don(static_cast<size_t>(n) * RepairerNet::N_DON), dur(static_cast<size_t>(n) * RepairerNet::N_DR),
                    ins(static_cast<size_t>(n) * 2 * RepairerNet::N_INS);
                for (int i = 0; i < n; i++) {
                    const float* o = out.data() + static_cast<size_t>(i) * RepairerNet::N_OUT;
                    keep[i] = o[0];
                    std::copy_n(o + 1, RepairerNet::N_DON, don.data() + static_cast<size_t>(i) * RepairerNet::N_DON);
                    std::copy_n(o + 1 + RepairerNet::N_DON, RepairerNet::N_DR, dur.data() + static_cast<size_t>(i) * RepairerNet::N_DR);
                    std::copy_n(o + 1 + RepairerNet::N_DON + RepairerNet::N_DR, 2 * RepairerNet::N_INS, ins.data() + static_cast<size_t>(i) * 2 * RepairerNet::N_INS);
                }
                tensor("out keep", keep, W.get("out_keep"));
                tensor("out don", don, W.get("out_don"));
                tensor("out dur", dur, W.get("out_dur"));
                tensor("out ins", ins, W.get("out_ins"));
                net.setDebug(false);
                // the query chunking (windows above 512 notes) on this window: chunks of 64 must give the same outputs
                std::vector<float> out64;
                net.setQueryChunk(64);
                net.run(n, pitch.data(), vel.data(), onf.data(), duf.data(), evf.data(), out64);
                net.setQueryChunk(RepairerNet::Q_CHUNK);
                std::vector<double> o512(out.begin(), out.end());
                double mx;
                const double r = relL2(out64.data(), o512.data(), o512.size(), &mx);
                std::printf("  %-26s rel_l2 %.3e  max_abs %.3e  %s\n", "query chunks 64 vs 512", r, mx, r <= 1e-6 ? "ok" : "FAIL");
                check(r <= 1e-6, "query chunking changes the outputs");
            }
            // every window of each variant (its own evidence): outputs, candidates, apply
            for (const std::string v : {"f16", "f32"}) {
                const auto wv = refWins(R, v);
                std::vector<std::vector<Cand>> per;
                refCands(R, v, wv, per);
                const std::vector<float>& evv = v == "f16" ? ev16f : ev32f;
                std::vector<float> keep, don, dur, ins, lg;
                std::vector<int> pitch, vel;
                std::vector<float> on, du, evw;
                std::vector<Cand> c;
                Edits e(kref.size());
                size_t kd = 0, id = 0, tot = 0;
                double smax = 0;
                net.resetStats();
                const double t0 = now_s();
                for (size_t wi = 0; wi < wv.size(); wi++) {
                    batch(kref, order_r, evv.data(), wv[wi], pitch, vel, on, du, evw);
                    const int n = wv[wi].hi - wv[wi].lo;
                    net.run(n, pitch.data(), vel.data(), on.data(), du.data(), evw.data(), lg);
                    for (int i = 0; i < n; i++) {
                        const float* o = lg.data() + static_cast<size_t>(i) * RepairerNet::N_OUT;
                        keep.push_back(o[0]);
                        don.insert(don.end(), o + 1, o + 1 + RepairerNet::N_DON);
                        dur.insert(dur.end(), o + 1 + RepairerNet::N_DON, o + 1 + RepairerNet::N_DON + RepairerNet::N_DR);
                        ins.insert(ins.end(), o + 1 + RepairerNet::N_DON + RepairerNet::N_DR, o + RepairerNet::N_OUT);
                    }
                    candidates(lg.data(), n, pitch.data(), on.data(), du.data(), net, c);
                    const CandDiff d = diffCands(c, per[wi], n);
                    kd += d.kept_diff;
                    id += d.ins_diff;
                    tot += d.n;
                    smax = std::max(smax, d.score_max);
                    addEdits(wv[wi], c, e);
                }
                const double dt = now_s() - t0;
                const auto& st = net.stats();
                std::printf("  %s: all %zu windows: %.3f s (graph builds %d: %.3f s, host features %.3f s, compute %.3f s; max compute "
                            "buffer %.1f MB, %d nodes)\n",
                            v.c_str(), wv.size(), dt, st.builds, st.build_s, st.features_s, st.compute_s, st.compute_bytes_max / 1e6, st.nodes_max);
                tensor(v + ": all windows keep", keep, R.get(v + "_out_keep"));
                tensor(v + ": all windows don", don, R.get(v + "_out_don"));
                tensor(v + ": all windows dur", dur, R.get(v + "_out_dur"));
                tensor(v + ": all windows ins", ins, R.get(v + "_out_ins"));
                std::printf("  %s: candidates            %zu: kept notes differing %zu, insertions differing %zu, max d_score %.2e  %s\n",
                            v.c_str(), tot, kd, id, smax, kd + id == 0 ? "ok" : "DIFFER");
                check(kd + id == 0, "candidates from our logits");
                RepairInfo info;
                const auto out = applyEdits(kref, e, keep_th, insert_th, &info);
                const size_t bad = compareRepaired(out, info, R, v, (v + ": repaired (its evidence)").c_str());
                check(bad == 0, "repaired notes from the reference evidence");
            }
        }

        // ================================================================== the whole stage
        if (!flag(argc, argv, "--skip-full")) {
            std::printf("[full]\n");
            Transcriber tr(mf, be, 2, LaneNet::Taps{true, false, false});
            const auto audio = R.get("audio16k").f32();
            auto sink = rep.makeSink();
            double t0 = now_s();
            const Maps maps = tr.run(audio.data(), audio.size(), 0.25, sink.get(), 1u << TapSink::HIDDEN);
            const double t_run = now_s() - t0;
            const auto notes = tr.notes(maps);
            auto* hs = dynamic_cast<HiddenStore*>(sink.get());
            const auto hst = hs->stats();
            std::printf("  transcriber pass 1: %.1f s audio, %.2f s (sink: accumulate %.2f s, finalise %.2f s)\n", audio.size() / 16000.0,
                        t_run, hst.accum_s, hst.final_s);
            std::printf("  sink memory: ring %zu frames = %.1f MB (freed after the pass); kept %zu H + %zu cs vectors = %.1f MB; sums + "
                        "flags %.1f MB  (the whole song's tap: %.1f MB f32)\n",
                        hst.ring_frames, hst.ring_bytes / 1e6, hst.n_hidden, hst.n_prefix, hst.store_bytes / 1e6, hst.sums_bytes / 1e6,
                        static_cast<double>(maps.T) * maps.L * C_H * 4 / 1e6);
            // the transcriber's notes: same pitch / voice in the same order, times within 0.1 ms (the transcriber's own parity
            // criterion), and the same frames A / B and the same rounded repairer inputs
            size_t nbad = notes.size() != tnotes.size();
            double d_on = 0, d_off = 0, d_conf = 0;
            for (size_t i = 0; i < std::min(notes.size(), tnotes.size()); i++) {
                d_on = std::max(d_on, std::abs(notes[i].onset - tnotes[i].onset));
                d_off = std::max(d_off, std::abs(notes[i].offset - tnotes[i].offset));
                d_conf = std::max(d_conf, static_cast<double>(std::abs(notes[i].conf - tnotes[i].conf)));
                nbad += notes[i].pitch != tnotes[i].pitch || notes[i].slot != tnotes[i].slot || std::abs(notes[i].onset - tnotes[i].onset) > 1e-4 ||
                        std::abs(notes[i].offset - tnotes[i].offset) > 1e-4;
            }
            std::printf("  transcriber notes          ours %zu, ref %zu, differing %zu (max d_onset %.2e s, d_offset %.2e s, d_conf %.2e)  %s\n",
                        notes.size(), tnotes.size(), nbad, d_on, d_off, d_conf, nbad == 0 ? "ok" : "DIFFER");
            check(nbad == 0, "transcriber notes");
            if (nbad == 0) {
                const auto sp = noteSpans(notes, maps.T);
                {
                    const auto L = R.get("span_L"), A = R.get("span_A"), B = R.get("span_B"), S = R.get("span_S");
                    size_t sbad = 0;
                    for (size_t i = 0; i < N; i++) {
                        sbad += sp[i].L != L.data[i] || sp[i].A != A.data[i] || sp[i].B != B.data[i] || sp[i].S != S.data[i];
                    }
                    std::vector<InNote> k;
                    std::vector<int> order;
                    inputNotes(notes, k, order);
                    size_t kbad = 0;
                    for (size_t i = 0; i < N; i++) {
                        kbad += k[i].pitch != kref[i].pitch || k[i].vel != kref[i].vel || k[i].on != kref[i].on || k[i].off != kref[i].off ||
                                order[i] != order_r[i];
                    }
                    std::printf("  spans / input notes        spans differing %zu, input notes differing %zu  %s\n", sbad, kbad,
                                sbad + kbad == 0 ? "ok" : "DIFFER");
                    check(sbad + kbad == 0, "spans / input notes");
                }
                const auto mfe = mapEvidence(maps, sp);
                tensor("map numbers (31)", mfe, R.get("mapf"));
                const Gathered g = gatherHidden(*hs, sp);
                std::printf("  hidden-store misses        %d  %s\n", g.misses, g.misses == 0 ? "ok" : "FAIL");
                if (g.misses != 0) {
                    g_fail++; // never tolerated, also not with --report
                }
                tensor("at  vs f32 accumulation", g.at, R.get("at32"));
                {
                    // Python's mean is (cs[B] - cs[A]) / (B - A) of float32-rounded float64 prefix sums over the whole song: a
                    // difference in the sums below what float32 resolves at that magnitude can flip a rounding, so the two
                    // means may differ by up to (ulp(cs[A]) + ulp(cs[B])) / (B - A), plus the hidden states' own difference
                    // (twice the largest |at - at32|). Every element must be within that bound.
                    const auto mref = R.get("mean32"), aref = R.get("at32");
                    double at_max = 0;
                    for (size_t i = 0; i < aref.size(); i++) {
                        at_max = std::max(at_max, std::abs(g.at[i] - aref.data[i]));
                    }
                    auto ulp = [](float x) { x = std::abs(x); return static_cast<double>(std::nextafter(x, INFINITY) - x); };
                    size_t viol = 0, beyond_hidden = 0;
                    double worst = 0, gran_max = 0;
                    for (size_t i = 0; i < N; i++) {
                        const float* ca = hs->prefix(sp[i].A, sp[i].L);
                        const float* cb = hs->prefix(sp[i].B, sp[i].L);
                        if (ca == nullptr || cb == nullptr) {
                            continue;
                        }
                        const int n = std::max(sp[i].B - sp[i].A, 1);
                        for (int c = 0; c < C_H; c++) {
                            const double d = std::abs(g.mean[i * C_H + c] - mref.data[i * C_H + c]);
                            const double gran = (ulp(ca[c]) + ulp(cb[c])) / n;
                            gran_max = std::max(gran_max, gran);
                            const double bound = gran + 2 * at_max;
                            viol += d > bound;
                            beyond_hidden += d > 2 * at_max;
                            worst = std::max(worst, d / bound);
                        }
                    }
                    double mx;
                    const double r = relL2(g.mean.data(), mref.data.data(), mref.size(), &mx);
                    std::printf("  %-26s rel_l2 %.3e  max_abs %.3e; %zu elements beyond the hidden states' own difference (%.2e), all "
                                "within the float32 cumsum granularity (largest %.2e)? violations %zu, worst d / bound %.2f  %s\n",
                                "mean vs f32 accumulation", r, mx, beyond_hidden, 2 * at_max, gran_max, viol, worst, viol == 0 ? "ok" : "FAIL");
                    check(viol == 0, "mean beyond the cumsum granularity");
                }
                tensor("at  vs f16 (package)", g.at, R.get("at16"), 1.0);
                tensor("mean vs f16 (package)", g.mean, R.get("mean16"), 1.0);
                std::vector<int8_t> q;
                const auto ev = assembleEvidence(mfe, g, &q);
                auto evdiff = [&](const std::vector<float>& a, const std::vector<float>& b, size_t* nm, size_t* nq, size_t* notes_any) {
                    *nm = *nq = *notes_any = 0;
                    for (size_t i = 0; i < N; i++) {
                        bool any = false;
                        for (int j = 0; j < N_EV; j++) {
                            if (a[i * N_EV + j] != b[i * N_EV + j]) {
                                (j < N_MAP ? *nm : *nq)++;
                                any = true;
                            }
                        }
                        *notes_any += any;
                    }
                };
                size_t nm, nq, na;
                {
                    // the evidence is quantised: the int8 steps may differ by at most 1, the 31 float16 map numbers (inputs
                    // equal to ~3e-6) by at most one f16 step; the two float16 log-spreads move with their vectors:
                    // |d log sd| <= ||d x|| / (sqrt(368) sd) to first order, plus one f16 step
                    int max_step = 0;
                    double max_f16 = 0, spread_worst = 0, spread_max = 0;
                    const auto aref = R.get("at32"), mref = R.get("mean32");
                    auto f16_ulp = [](float lo) {
                        return std::ldexp(1.0, std::max(-24, static_cast<int>(std::floor(std::log2(std::max(lo, 6.1e-5f)))) - 10));
                    };
                    for (size_t i = 0; i < N; i++) {
                        for (int j = 0; j < N_EV; j++) {
                            const float a = ev[i * N_EV + j], b = ev32f[i * N_EV + j];
                            if (a == b) {
                                continue;
                            }
                            const float lo = std::min(std::abs(a), std::abs(b));
                            if (j < N_MAP - 2) {
                                const double st = std::abs(a - b) / f16_ulp(lo);
                                if (st > 1.0 + 1e-9 && g_verbose) {
                                    std::printf("      note %zu map column %d: ours %.6g ref %.6g (%.0f f16 steps)\n", i, j, a, b, st);
                                }
                                max_f16 = std::max(max_f16, st);
                            } else if (j < N_MAP) {
                                const float* x = (j == N_MAP - 2 ? g.at.data() : g.mean.data()) + i * C_H;
                                const double* xr = (j == N_MAP - 2 ? aref : mref).data.data() + i * C_H;
                                double d2 = 0, mu = 0, v2 = 0;
                                for (int c = 0; c < C_H; c++) {
                                    d2 += (x[c] - xr[c]) * (x[c] - xr[c]);
                                    mu += xr[c];
                                }
                                mu /= C_H;
                                for (int c = 0; c < C_H; c++) {
                                    v2 += (xr[c] - mu) * (xr[c] - mu);
                                }
                                const double sd = std::sqrt(v2 / C_H);
                                const double bound = std::sqrt(d2) / (std::sqrt(static_cast<double>(C_H)) * sd) * 1.01 + f16_ulp(std::max(std::abs(a), std::abs(b)));
                                spread_worst = std::max(spread_worst, std::abs(a - b) / bound);
                                spread_max = std::max(spread_max, static_cast<double>(std::abs(a - b)));
                            } else {
                                max_step = std::max(max_step, static_cast<int>(std::lround(std::abs(a - b) * Q)));
                            }
                        }
                    }
                    const bool ok = max_step <= 1 && max_f16 <= 1.0 + 1e-9 && spread_worst <= 1.0;
                    std::printf("  evidence steps             largest int8 difference %d step(s), largest map-number difference %.2f f16 "
                                "step(s), log-spreads: largest |d| %.2e, worst d / bound %.2f  %s\n",
                                max_step, max_f16, spread_max, spread_worst, ok ? "ok" : "FAIL");
                    check(ok, "evidence differs by more than its inputs explain");
                }
                evdiff(ev, ev32f, &nm, &nq, &na);
                std::printf("  evidence vs f32 variant    map numbers differing %zu of %zu, int8 steps differing %zu of %zu (notes with any: %zu)\n",
                            nm, N * N_MAP, nq, N * 2 * C_H, na);
                evdiff(ev, ev16f, &nm, &nq, &na);
                std::printf("  evidence vs f16 (package)  map numbers differing %zu, int8 steps differing %zu (notes with any: %zu)\n", nm, nq, na);
                evdiff(ev32f, ev16f, &nm, &nq, &na);
                std::printf("  python f32 vs f16 itself   map numbers differing %zu, int8 steps differing %zu (notes with any: %zu)\n", nm, nq, na);
                {
                    double mx;
                    const double r = relL2(ev.data(), ev32.data.data(), ev32.size(), &mx);
                    std::printf("  %-26s rel_l2 %.3e  max_abs %.3e (quantised: see the step counts above)\n", "evidence vs f32 variant", r, mx);
                }
            }
            RepairInfo info;
            rep.net().resetStats();
            t0 = now_s();
            const auto out = rep.repair(maps, notes, *sink, &info);
            const double t_rep = now_s() - t0;
            const auto& st = rep.net().stats();
            const auto& tm = rep.lastTiming();
            std::printf("  repair(): %.3f s = evidence %.3f s + network %.3f s (%d graph builds %.3f s, host features %.3f s, compute "
                        "%.3f s) + candidates / apply %.3f s; misses %d\n",
                        t_rep, tm.evidence_s, tm.net_s, st.builds, st.build_s, st.features_s, st.compute_s, tm.post_s, rep.lastMisses());
            if (rep.lastMisses() != 0) {
                g_fail++;
            }
            const size_t b32 = compareRepaired(out, info, R, "f32", "repaired vs f32 variant");
            check(b32 == 0, "repaired notes vs the f32-accumulation reference");
            compareRepaired(out, info, R, "f16", "repaired vs f16 (package)");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(g_fail ? "%d check(s) FAILED\n" : "all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
