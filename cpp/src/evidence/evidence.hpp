#pragma once

#include <memory>
#include <vector>

#include "transcriber/transcriber.hpp"

namespace dtt
{

class Backend;
class ModelFile;

/** A note after the evidence repair (EvidenceRepairer.apply in the Python package). */
struct RepairedNote {
    double onset, offset; // s, rounded to 1e-4 as Python's round(x, 4)
    int pitch;            // MIDI
    bool inserted;        // proposed by the repairer, not transcribed
    float score;          // P(keep) for a kept note (1 when its window passed through unchanged), P(missing) for an insertion
};

struct RepairInfo {
    int transcribed = 0, kept = 0, dropped = 0, inserted = 0;
};

namespace ev
{
    class RepairerNet;
}

/**
 * Stage 2: the evidence repairer (dettato/evidence). Usage, after the transcriber's first pass:
 *
 *     auto sink = rep.makeSink();
 *     Maps m = tr.run(audio, n, 0.25, sink.get(), 1u << TapSink::HIDDEN);
 *     auto tn = tr.notes(m);
 *     auto out = rep.repair(m, tn, *sink, &info);
 */
class EvidenceRepairer
{
public:
    /** mf holds the "ev." tensors and the transcriber's "tr." metadata (its decode thresholds). */
    EvidenceRepairer(const ModelFile& mf, const Backend& be);
    ~EvidenceRepairer();
    EvidenceRepairer(const EvidenceRepairer&) = delete;
    EvidenceRepairer& operator=(const EvidenceRepairer&) = delete;

    /**
     * The TapSink to pass to Transcriber::run for pass 1 (hop_frac 0.25, hidden tap). It keeps only the hidden states the
     * evidence vectors read (see evidence/hidden_store.hpp), not the whole song's.
     */
    std::unique_ptr<TapSink> makeSink() const;

    /**
     * After that pass: the maps, Transcriber::notes(maps) and the sink -> the repaired notes in the order
     * EvidenceRepairer.apply returns them: the kept transcribed notes in input order (sorted by onset, pitch), then the
     * insertions in proposal order; times rounded to 1e-4.
     */
    std::vector<RepairedNote> repair(const Maps& maps, const std::vector<TranscribedNote>& notes, TapSink& sink,
                                     RepairInfo* info) const;

    ev::RepairerNet& net() const { return *mNet; }
    /** Notes of the last repair() whose hidden vectors the sink had not kept (zeros used); 0 unless the store's
     *  candidate rules are wrong (the parity tool fails on any). */
    int lastMisses() const { return mLastMisses; }
    /** Seconds of the last repair(): evidence (maps + gather + quantisation), the network windows, candidates + apply. */
    struct Timing {
        double evidence_s = 0, net_s = 0, post_s = 0;
    };
    const Timing& lastTiming() const { return mTiming; }

private:
    std::unique_ptr<ev::RepairerNet> mNet;
    float mOnTh, mV2Th, mFrTh, mOffTh;
    // stored as float32; no float32 lies between 0.4 / 0.6 and their float32 roundings, so P >= th decides as in Python
    float mKeepTh, mInsertTh;
    mutable int mLastMisses = 0;
    mutable Timing mTiming;
};

} // namespace dtt
