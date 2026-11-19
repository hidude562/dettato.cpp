#pragma once

#include <vector>

namespace dtt
{

struct DecodeParams {
    float on_th = 0.25f;
    float off_th = 0.3f;
    float fr_th = 0.4f;
    int fr_low_frames = 2;
    int on_nms = 3;
    double min_dur = 0.02;
    int off_min_frames = 2;
};

struct DecodedNote {
    double onset;  // s
    double offset; // s
    int pitch;     // MIDI
};

/**
 * dettato/transcriber/decode.py decode: one voice's (T, L) onset / frame / offset maps (row-major, lane fastest) ->
 * notes sorted by (onset, offset, pitch).
 */
std::vector<DecodedNote> decodeVoice(const float* onset, const float* frame, const float* offset, int T, int L, double hop,
                                     const DecodeParams& p, int pitch_lo = 21);

} // namespace dtt
