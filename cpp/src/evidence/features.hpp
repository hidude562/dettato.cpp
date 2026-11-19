#pragma once

#include <cstdint>
#include <vector>

#include "transcriber/transcriber.hpp"

namespace dtt::ev
{

class HiddenStore;

constexpr int C_H = 368;                // the transcriber's hidden channels
constexpr int N_MAP = 33;               // map numbers incl. the two log-spreads
constexpr int N_EV = N_MAP + 2 * C_H;   // 769
constexpr float Q = 32.0f;              // int8 steps of 1/32
constexpr double HOP = 0.01 * 2;        // the transcriber's frame hop, as model.hop_out computes it
constexpr int N_KEYS = 88;
constexpr int PITCH_LO = 21;

/** A note's place in the maps (transcribe_with_evidence): lane, first frame, end frame (exclusive), voice 1 / 2. */
struct NoteSpan {
    int L, A, B, S;
};
std::vector<NoteSpan> noteSpans(const std::vector<TranscribedNote>& notes, int T);

/** features.py map_evidence: (N, 31) row-major float32. */
std::vector<float> mapEvidence(const Maps& maps, const std::vector<NoteSpan>& sp);

/** numpy's float32 add.reduce over a contiguous run (pairwise summation, 8 accumulators, blocks of 128). */
float npSumF32(const float* a, size_t n);

/** np.log of a float32 as numpy computes it on x86 with AVX2 + FMA (its SIMD rational approximation, not libm). */
float npLogF32(float x);

/** features.py _std_q for one 368-vector, with numpy's float32 arithmetic: int8 steps of 1/32 and log(sd). */
void stdQ(const float* x, int8_t* q, float* log_sd);

/** float32 -> float16 (round to nearest even, as numpy's astype) -> float32. */
float roundF16(float x);

/** The gathered hidden vectors of every note (gather_hidden): at = H[A, L], mean = (cs[B] - cs[A]) / (B - A). */
struct Gathered {
    std::vector<float> at, mean; // (N, C_H)
    int misses = 0;              // notes whose A / B the store did not keep (must be 0)
};
Gathered gatherHidden(const HiddenStore& hs, const std::vector<NoteSpan>& sp);

/** transcribe_with_evidence's assembly: (N, 769) from the map numbers and the two hidden vectors. q (optional) gets the
 *  (N, 736) int8 steps. */
std::vector<float> assembleEvidence(const std::vector<float>& mapf, const Gathered& g, std::vector<int8_t>* q = nullptr);

/** A note as the repairer sees it (repair.py NOTE_DT, inst always 0). */
struct InNote {
    int pitch;
    int vel;   // 1..127
    float on;  // s, float32 of the onset rounded to 1e-4
    float off; // s, float32 of max(offset rounded to 1e-4, on + 1e-3)
};

/** repair.py input_notes: the notes sorted by (onset, pitch) and that order (k = notes[order]). */
void inputNotes(const std::vector<TranscribedNote>& notes, std::vector<InNote>& k, std::vector<int>& order);

/** numpy's round(x, 4) of a float64: rint(x * 1e4) / 1e4. */
double npRound4(double x);
/** Python's round(x, 4): the float64 nearest to the decimal rounding of x's exact value. */
double pyRound4(double x);

} // namespace dtt::ev
