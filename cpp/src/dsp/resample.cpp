#include "dsp/resample.hpp"

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <thread>

namespace dtt
{

namespace
{
    constexpr int TABLE_RES = 4096;  // kernel samples per zero crossing
    constexpr int ZEROS = 64;      // zero crossings each side of the kernel
    constexpr double BETA = 7.0;   // Kaiser window (64 / 7.0 / 0.96: the closest to soxr HQ we found, 0.36% rel L2)
    constexpr double PASS = 0.96;  // cutoff, fraction of the lower Nyquist

    double besselI0(double x)
    {
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 60; k++) {
            t *= (x / (2.0 * k)) * (x / (2.0 * k));
            s += t;
            if (t < 1e-17 * s) {
                break;
            }
        }
        return s;
    }

    /** h(u) for u in zero crossings, u in [0, ZEROS], tabulated. */
    const std::vector<double>& kernelTable()
    {
        static const std::vector<double> table = [] {
            std::vector<double> t(static_cast<size_t>(ZEROS) * TABLE_RES + 2);
            const double i0b = besselI0(BETA);
            for (size_t i = 0; i < t.size(); i++) {
                const double u = static_cast<double>(i) / TABLE_RES;
                const double sinc = u == 0.0 ? 1.0 : std::sin(std::numbers::pi * u) / (std::numbers::pi * u);
                const double r = u / ZEROS;
                const double win = r >= 1.0 ? 0.0 : besselI0(BETA * std::sqrt(1.0 - r * r)) / i0b;
                t[i] = sinc * win;
            }
            return t;
        }();
        return table;
    }
} // namespace

std::vector<float> resample(const float* x, size_t n, double sr_in, double sr_out, int threads)
{
    if (sr_in == sr_out || n == 0) {
        return std::vector<float>(x, x + n);
    }
    const size_t m = static_cast<size_t>(std::llround(static_cast<double>(n) * sr_out / sr_in));
    std::vector<float> y(m);
    const double step = sr_in / sr_out;               // input samples per output sample
    const double fc = PASS * std::min(1.0, sr_out / sr_in); // cutoff in units of the input Nyquist (zero-crossing rate)
    const double half = ZEROS / fc;                    // kernel half width, input samples
    const double gain = fc;
    const auto& tab = kernelTable();

    auto work = [&](size_t k0, size_t k1) {
        for (size_t k = k0; k < k1; k++) {
            const double t = static_cast<double>(k) * step;
            const long i0 = static_cast<long>(std::ceil(t - half));
            const long i1 = static_cast<long>(std::floor(t + half));
            double acc = 0.0;
            for (long i = std::max(0L, i0); i <= std::min(static_cast<long>(n) - 1, i1); i++) {
                const double u = std::abs(t - static_cast<double>(i)) * fc * TABLE_RES; // table index
                const size_t j = static_cast<size_t>(u);
                if (j + 1 >= tab.size()) {
                    continue;
                }
                const double f = u - static_cast<double>(j);
                acc += x[i] * (tab[j] + f * (tab[j + 1] - tab[j]));
            }
            y[k] = static_cast<float>(acc * gain);
        }
    };
    const int nt = threads > 0 ? threads : defaultThreadCount();
    std::vector<std::thread> pool;
    const size_t chunk = (m + nt - 1) / nt;
    for (int i = 0; i < nt; i++) {
        const size_t a = i * chunk, b = std::min(m, a + chunk);
        if (a < b) {
            pool.emplace_back(work, a, b);
        }
    }
    for (auto& t : pool) {
        t.join();
    }
    return y;
}

} // namespace dtt
