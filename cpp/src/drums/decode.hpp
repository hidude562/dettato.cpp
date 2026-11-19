#pragma once

#include <string>
#include <vector>

#include "drums/drums.hpp"

namespace dtt
{

struct DrumProbs;
class ModelFile;

/** dettato/drums/infer.py: peak picking, the router decision, the timpani merge, transcribe()'s rounding. */
struct DrumDecoder {
    std::vector<int> keys, timpani_keys;
    float th_kit = 0.5f, th_drum = 0.5f, th_timpani = 0.5f; // compared with float32 maps, as numpy does
    double th_router = 0.5;                                  // compared in float64
    int gap = 3, fps = 50;
    double timpani_repeat = 0.04, hit_length = 0.1;

    DrumDecoder() = default;
    explicit DrumDecoder(const ModelFile& mf, const std::string& prefix = "dr");

    /** peaks(): (T, L) probabilities -> (frame, lane) of the local maxima >= th, at least gap frames apart per lane. */
    static void peaks(const float* prob, int T, int L, float th, int gap, std::vector<int>& fr, std::vector<int>& ln);

    /** Where a hit came from (for the parity tool): map 0 kit, 1 orchestral drum, 2 timpani; the velocity before rounding. */
    struct Source {
        int map, frame, lane;
        double raw;
    };

    /** decode() + transcribe()'s dicts: hits sorted by onset (stable), rounded as Python's round(x, 4) / round(v, 3). */
    std::vector<DrumHit> decode(const DrumProbs& P, std::string& model, std::vector<Source>* sources = nullptr) const;

    /** Python's round(x, nd) on a double: the correctly rounded decimal, half to even, read back. */
    static double pyRound(double x, int nd);
};

} // namespace dtt
