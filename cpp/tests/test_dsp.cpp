// The resampler and the STFT, against closed-form signals and a naive DFT. No weights needed.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

#include "dsp/resample.hpp"
#include "dsp/stft.hpp"

namespace
{
std::vector<float> sine(double freq, double rate, size_t n, double amp = 0.5)
{
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = static_cast<float>(amp * std::sin(2.0 * std::numbers::pi * freq * static_cast<double>(i) / rate));
    }
    return x;
}

double rms(const std::vector<float>& x, size_t from, size_t to)
{
    double s = 0;
    for (size_t i = from; i < to; i++) {
        s += static_cast<double>(x[i]) * x[i];
    }
    return std::sqrt(s / static_cast<double>(to - from));
}
} // namespace

TEST_CASE("resample keeps the length round(n * out / in)", "[pure][resample]")
{
    const std::vector<float> x(44101, 0.0f);
    CHECK(dtt::resample(x.data(), x.size(), 44100.0, 16000.0).size() == 16000);
    CHECK(dtt::resample(x.data(), 1000, 48000.0, 16000.0).size() == 333);
    CHECK(dtt::resample(x.data(), 1000, 22050.0, 44100.0).size() == 2000);
}

TEST_CASE("resample at the same rate is the identity", "[pure][resample]")
{
    const auto x = sine(440.0, 16000.0, 4000);
    const auto y = dtt::resample(x.data(), x.size(), 16000.0, 16000.0);
    CHECK(y == x);
}

TEST_CASE("resample passes a tone below the cutoff unchanged", "[pure][resample]")
{
    const auto x = sine(440.0, 44100.0, 44100);
    const auto y = dtt::resample(x.data(), x.size(), 44100.0, 16000.0);
    const auto want = sine(440.0, 16000.0, y.size());
    double err = 0;
    for (size_t i = 400; i + 400 < y.size(); i++) { // away from the edges, where the kernel sees the zero padding
        err = std::max(err, std::abs(static_cast<double>(y[i]) - want[i]));
    }
    CHECK(err < 2e-3);

    const auto up = dtt::resample(want.data(), want.size(), 16000.0, 44100.0);
    const auto want_up = sine(440.0, 44100.0, up.size());
    double err_up = 0;
    for (size_t i = 1000; i + 1000 < up.size(); i++) {
        err_up = std::max(err_up, std::abs(static_cast<double>(up[i]) - want_up[i]));
    }
    CHECK(err_up < 2e-3);
}

TEST_CASE("resample removes a tone above the new Nyquist", "[pure][resample]")
{
    const auto x = sine(10000.0, 44100.0, 44100); // above 8 kHz: must not alias into the 16 kHz signal
    const auto y = dtt::resample(x.data(), x.size(), 44100.0, 16000.0);
    CHECK(rms(y, 400, y.size() - 400) < 0.5 / std::sqrt(2.0) * 1e-3); // at least 60 dB down
}

TEST_CASE("PowerStft matches a direct DFT (torch.stft, center=True, zero padding)", "[pure][stft]")
{
    const int n_fft = 64, hop = 16;
    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / n_fft)); // periodic Hann
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(300);
    for (float& v : x) {
        v = nd(rng);
    }
    const int frames = 1 + static_cast<int>(x.size()) / hop;
    dtt::PowerStft stft(n_fft, hop, win);
    REQUIRE(stft.bins() == n_fft / 2 + 1);
    std::vector<float> out(static_cast<size_t>(frames) * stft.bins());
    stft.compute(x.data(), x.size(), frames, out.data());

    double num = 0, den = 0;
    for (int t = 0; t < frames; t++) {
        for (int k = 0; k < stft.bins(); k++) {
            std::complex<double> acc = 0;
            for (int j = 0; j < n_fft; j++) {
                const long s = static_cast<long>(t) * hop - n_fft / 2 + j;
                const double v = (s >= 0 && s < static_cast<long>(x.size())) ? x[static_cast<size_t>(s)] * win[j] : 0.0;
                acc += v * std::polar(1.0, -2.0 * std::numbers::pi * k * j / n_fft);
            }
            const double want = std::norm(acc);
            const double got = out[static_cast<size_t>(t) * stft.bins() + k];
            num += (got - want) * (got - want);
            den += want * want;
        }
    }
    CHECK(std::sqrt(num / den) < 1e-5);

    std::vector<float> mag(out.size());
    stft.computeMagnitude(x.data(), x.size(), frames, mag.data());
    for (size_t i = 0; i < out.size(); i++) {
        const double want = out[i];
        REQUIRE_THAT(static_cast<double>(mag[i]) * mag[i], Catch::Matchers::WithinRel(want, 1e-4) || Catch::Matchers::WithinAbs(want, 1e-6));
    }
}
