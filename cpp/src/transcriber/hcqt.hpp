#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dsp/stft.hpp"

namespace dtt
{

class ModelFile;

/** A dense filterbank stored row-sparse: each row's nonzero run. */
struct SparseRows {
    int rows = 0, cols = 0;
    std::vector<int> start, len;
    std::vector<size_t> off;
    std::vector<float> w;

    static SparseRows fromDense(const std::vector<float>& dense, int rows, int cols);
    /** out[r] = sum_c w[r, c] * in[c] */
    void apply(const float* in, float* out) const;
};

/**
 * The transcriber's harmonic CQT front end (dettato/transcriber/frontend.py HCQT), on the CPU:
 * audio -> (C, T, F) log power, T = n / 160, F = 288 (3 bins per semitone from A0), C = harmonic channels + extra windows.
 */
class Hcqt
{
public:
    explicit Hcqt(const ModelFile& mf, const std::string& prefix = "tr");

    int channels() const { return static_cast<int>(mShifts.size() + mExtra.size()); }
    int bins() const { return mBins; }
    int extendedBins() const { return mFb.rows; }
    static constexpr int HOP = 160;

    /** out: (channels, T, bins) with T = n / HOP. */
    void compute(const float* x, size_t n, float* out) const;

    /**
     * The same, plus P: the linear power on the extended log-frequency axis, (extendedBins, T) (grouper/front.py Front).
     * p_out may be null.
     */
    void compute(const float* x, size_t n, float* out, float* p_out) const;

private:
    struct Extra {
        std::unique_ptr<PowerStft> stft;
        SparseRows fb;
    };
    int mBins = 0;
    int mShiftLo = 0;
    float mLogEps = 1e-5f;
    std::vector<int> mShifts;
    std::unique_ptr<PowerStft> mStft;
    SparseRows mFb;
    std::vector<Extra> mExtra;
};

} // namespace dtt
