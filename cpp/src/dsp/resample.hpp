#pragma once

#include <cstddef>
#include <vector>

namespace dtt
{

/**
 * High-quality band-limited resampling (Kaiser-windowed sinc, linear phase, no delay), standing in for the Python
 * package's soxr.resample(..., quality="HQ"). Output length: round(n * sr_out / sr_in).
 * @param threads 0 = hardware concurrency.
 */
std::vector<float> resample(const float* x, size_t n, double sr_in, double sr_out, int threads = 0);

} // namespace dtt
