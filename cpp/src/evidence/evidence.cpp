#include "evidence/evidence.hpp"

#include <algorithm>
#include <chrono>
#include <map>

#include "common.hpp"
#include "evidence/features.hpp"
#include "evidence/hidden_store.hpp"
#include "evidence/net.hpp"
#include "evidence/repair.hpp"
#include "model_file.hpp"

namespace dtt
{

EvidenceRepairer::EvidenceRepairer(const ModelFile& mf, const Backend& be) : mNet(std::make_unique<ev::RepairerNet>(mf, be))
{
    mOnTh = mf.f32("tr.th.on_th");
    mV2Th = mf.has("tr.th.v2_th") ? mf.f32("tr.th.v2_th") : -1.0f;
    mFrTh = mf.f32("tr.th.fr_th");
    mOffTh = mf.f32("tr.th.off_th");
    mKeepTh = mf.f32("ev.keep_th");
    mInsertTh = mf.f32("ev.insert_th");
    DTT_CHECK(mf.f32("ev.window_s") == 8.0f, "the repairer's window is %g s, this build assumes 8 s", mf.f32("ev.window_s"));
}

EvidenceRepairer::~EvidenceRepairer() = default;

std::unique_ptr<TapSink> EvidenceRepairer::makeSink() const
{
    return std::make_unique<ev::HiddenStore>(ev::DecodeThresholds{mOnTh, mV2Th, mFrTh, mOffTh});
}

std::vector<RepairedNote> EvidenceRepairer::repair(const Maps& maps, const std::vector<TranscribedNote>& notes, TapSink& sink,
                                                   RepairInfo* info) const
{
    using namespace ev;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
    const auto t_ev = clk::now();
    mTiming = Timing{};
    auto* hs = dynamic_cast<HiddenStore*>(&sink);
    DTT_CHECK(hs != nullptr, "repair() needs the sink from makeSink()");
    // the evidence of every note (transcribe_with_evidence), in Transcriber.notes order
    const auto sp = noteSpans(notes, maps.T);
    const auto mapf = mapEvidence(maps, sp);
    const Gathered g = gatherHidden(*hs, sp);
    mLastMisses = g.misses;
    const auto ev = assembleEvidence(mapf, g);
    mTiming.evidence_s = secs(t_ev);
    // the repairer's notes, sorted, and 8 s windows of them
    std::vector<InNote> k;
    std::vector<int> order;
    inputNotes(notes, k, order);
    const auto wins = windowSpans(k, mNet->maxWindow());
    // run the windows grouped by graph bucket (one graph build per bucket), keep each window's candidates
    std::vector<std::vector<Cand>> cands(wins.size());
    std::vector<size_t> by_bucket(wins.size());
    for (size_t i = 0; i < wins.size(); i++) {
        by_bucket[i] = i;
    }
    std::stable_sort(by_bucket.begin(), by_bucket.end(), [&](size_t a, size_t b) {
        return RepairerNet::bucket(wins[a].hi - wins[a].lo) < RepairerNet::bucket(wins[b].hi - wins[b].lo);
    });
    std::vector<int> pitch, vel;
    std::vector<float> on, dur, evw, logits;
    for (size_t wi : by_bucket) {
        const WindowSpan& w = wins[wi];
        const int n = w.hi - w.lo;
        pitch.resize(n);
        vel.resize(n);
        on.resize(n);
        dur.resize(n);
        evw.resize(static_cast<size_t>(n) * N_EV);
        const float t0 = static_cast<float>(w.t0);
        for (int j = 0; j < n; j++) {
            const InNote& x = k[w.lo + j];
            pitch[j] = x.pitch;
            vel[j] = x.vel;
            on[j] = x.on - t0;                     // (k["on"] - t0).astype(float32)
            dur[j] = std::max(x.off - x.on, 0.0f); // np.maximum(k["off"] - k["on"], 0)
            std::copy_n(ev.data() + static_cast<size_t>(order[w.lo + j]) * N_EV, N_EV, evw.data() + static_cast<size_t>(j) * N_EV);
        }
        auto t = clk::now();
        mNet->run(n, pitch.data(), vel.data(), on.data(), dur.data(), evw.data(), logits);
        mTiming.net_s += secs(t);
        t = clk::now();
        candidates(logits.data(), n, pitch.data(), on.data(), dur.data(), *mNet, cands[wi]);
        mTiming.post_s += secs(t);
    }
    const auto t_ap = clk::now();
    Edits e(k.size());
    for (size_t i = 0; i < wins.size(); i++) {
        addEdits(wins[i], cands[i], e);
    }
    RepairInfo tmp;
    auto out = applyEdits(k, e, static_cast<double>(mKeepTh), static_cast<double>(mInsertTh), info != nullptr ? info : &tmp);
    mTiming.post_s += secs(t_ap);
    return out;
}

} // namespace dtt
