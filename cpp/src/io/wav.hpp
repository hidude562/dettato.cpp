#pragma once

#include <string>
#include <vector>

namespace dtt
{

struct Audio {
    std::vector<float> samples; // mono
    double rate = 0;
};

/** Reads a WAV file (PCM 8/16/24/32-bit integer or 32/64-bit float, any channel count) and downmixes to mono. */
Audio readWav(const std::string& path);

} // namespace dtt
