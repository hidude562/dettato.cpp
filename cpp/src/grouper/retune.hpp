#pragma once

#include <cstddef>

namespace dtt::retune
{

constexpr int SR = 16000;
constexpr double MIN_CENTS = 10.0;     // retune only when the estimate is at least this far from A440 ...
constexpr double MIN_STRENGTH = 0.12;  // ... and consistent enough (resultant length of the circular mean)

/**
 * grouper/retune.py tuning(): offset from A440 in cents [-50, 50) and strength [0, 1]. Harmonic salience of the square-root
 * magnitude STFT (8192 points, hop 1024) on a 5-cent grid, per-frame peaks above half the frame maximum, parabolically
 * refined, weighted circular mean modulo 100 c. float32 where numpy is float32, float64 where it is float64.
 */
void tuning(const float* y, size_t n, double& cents, double& strength);

/** retune.ratio(): (offset, strength, shift applied in cents, r); r = 1 when the recording is left as it is. */
void ratio(const float* y, size_t n, double& delta, double& strength, double& shift, double& r);

} // namespace dtt::retune
