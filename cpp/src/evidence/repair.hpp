#pragma once

#include <array>
#include <vector>

#include "evidence/evidence.hpp"
#include "evidence/features.hpp"

namespace dtt::ev
{

class RepairerNet;

/** model.py CAND_DT: an edited input note (kind 0) or an insertion proposal (kind 1), window-relative times. */
struct Cand {
    int pitch;
    float on, off, score;
    int kind;
};

/**
 * Repairer.candidates for one window: logits (n, RepairerNet::N_OUT) -> n edited notes, then 2 n insertion proposals
 * (note-major, slot-minor). on / dur: the window-relative onsets and durations the network saw.
 */
void candidates(const float* logits, int n, const int* pitch, const float* on, const float* dur, const RepairerNet& net,
                std::vector<Cand>& out);

/** A window of EvidenceRepairer.edits: notes k[lo, hi) with onsets in [t0, t0 + 8). */
struct WindowSpan {
    double t0;
    int lo, hi;
};
/** The windows edits() runs the model on (non-empty, at most max_win notes). */
std::vector<WindowSpan> windowSpans(const std::vector<InNote>& k, int max_win);

/** EvidenceRepairer.edits' result: per input note P(keep) / onset / end (NaN: passed through), and the insertions. */
struct Edits {
    std::vector<double> pk, on_r, off_r;
    std::vector<std::array<double, 4>> ins; // onset, end, pitch, score
    explicit Edits(size_t n);
};
/** Adds one window's candidates (as edits() does). */
void addEdits(const WindowSpan& w, const std::vector<Cand>& c, Edits& e);

/** EvidenceRepairer.apply (with _safe_dedup_idx): the repaired notes, and the counts. */
std::vector<RepairedNote> applyEdits(const std::vector<InNote>& k, const Edits& e, double keep_th, double insert_th,
                                     RepairInfo* info);

/** SLEEF's expf (u10, the AVX2 + FMA build torch uses for Vectorized<float>::exp), in scalar code. */
float sleefExpF(float d);
/** torch.softmax over a float32 row on the CPU: max, SLEEF exp(x - max), the 8-lane AVX2 sum, times 1 / sum. */
void softmaxRow(const float* x, int n, float* out);
/** model.py _box_argmax: argmax of the zero-padded box sum of half-width h (summed in order, float32); first maximum. */
int boxArgmax(const float* p, int n, int h);

} // namespace dtt::ev
