#include "transcriber/hcqt.hpp"

#include <cmath>

#include "common.hpp"
#include "model_file.hpp"

namespace dtt
{

SparseRows SparseRows::fromDense(const std::vector<float>& dense, int rows, int cols)
{
    DTT_CHECK(dense.size() == static_cast<size_t>(rows) * cols, "filterbank size mismatch");
    SparseRows s;
    s.rows = rows;
    s.cols = cols;
    for (int r = 0; r < rows; r++) {
        const float* row = dense.data() + static_cast<size_t>(r) * cols;
        int a = 0, b = cols;
        while (a < cols && row[a] == 0.0f) {
            a++;
        }
        while (b > a && row[b - 1] == 0.0f) {
            b--;
        }
        s.start.push_back(a);
        s.len.push_back(b - a);
        s.off.push_back(s.w.size());
        s.w.insert(s.w.end(), row + a, row + b);
    }
    return s;
}

void SparseRows::apply(const float* in, float* out) const
{
    for (int r = 0; r < rows; r++) {
        const float* w = this->w.data() + off[r];
        const float* x = in + start[r];
        float acc = 0.0f;
        for (int i = 0; i < len[r]; i++) {
            acc += w[i] * x[i];
        }
        out[r] = acc;
    }
}

Hcqt::Hcqt(const ModelFile& mf, const std::string& p)
{
    mBins = static_cast<int>(mf.u32(p + ".n_bins"));
    mShiftLo = static_cast<int>(mf.u32(p + ".shift_lo"));
    mLogEps = mf.f32(p + ".log_eps");
    mShifts = mf.i32s(p + ".shifts");
    const int n_fft = static_cast<int>(mf.u32(p + ".n_fft"));
    mStft = std::make_unique<PowerStft>(n_fft, HOP, mf.hostF32(p + ".hcqt.win"));
    const auto fb = mf.hostF32(p + ".hcqt.fb");
    const int nb = n_fft / 2 + 1;
    mFb = SparseRows::fromDense(fb, static_cast<int>(fb.size() / nb), nb);
    const auto extra = mf.i32s(p + ".extra_nffts");
    for (size_t i = 0; i < extra.size(); i++) {
        Extra e;
        e.stft = std::make_unique<PowerStft>(extra[i], HOP, mf.hostF32(p + ".hcqt.win_x" + std::to_string(i)));
        const auto fbx = mf.hostF32(p + ".hcqt.fb_x" + std::to_string(i));
        e.fb = SparseRows::fromDense(fbx, mBins, extra[i] / 2 + 1);
        mExtra.push_back(std::move(e));
    }
}

void Hcqt::compute(const float* x, size_t n, float* out) const
{
    compute(x, n, out, nullptr);
}

void Hcqt::compute(const float* x, size_t n, float* out, float* p_out) const
{
    const int T = static_cast<int>(n / HOP);
    const int F = mBins, NE = mFb.rows;
    const float pad = static_cast<float>(std::log(static_cast<double>(mLogEps)));

    std::vector<float> pw(static_cast<size_t>(T) * mStft->bins());
    mStft->compute(x, n, T, pw.data());
    // lf[t][b] = log(fb @ p + eps) on the extended axis
    std::vector<float> lf(static_cast<size_t>(T) * NE);
    for (int t = 0; t < T; t++) {
        float* row = lf.data() + static_cast<size_t>(t) * NE;
        mFb.apply(pw.data() + static_cast<size_t>(t) * mStft->bins(), row);
        if (p_out != nullptr) {
            for (int b = 0; b < NE; b++) {
                p_out[static_cast<size_t>(b) * T + t] = row[b];
            }
        }
        for (int b = 0; b < NE; b++) {
            row[b] = std::log(row[b] + mLogEps);
        }
    }
    size_t ch = 0;
    for (int s : mShifts) {
        const int a = s + mShiftLo;
        float* o = out + ch * static_cast<size_t>(T) * F;
        for (int t = 0; t < T; t++) {
            const float* row = lf.data() + static_cast<size_t>(t) * NE;
            for (int k = 0; k < F; k++) {
                o[static_cast<size_t>(t) * F + k] = (a + k < NE) ? row[a + k] : pad;
            }
        }
        ch++;
    }
    for (const Extra& e : mExtra) {
        std::vector<float> px(static_cast<size_t>(T) * e.stft->bins());
        e.stft->compute(x, n, T, px.data());
        float* o = out + ch * static_cast<size_t>(T) * F;
        for (int t = 0; t < T; t++) {
            float* row = o + static_cast<size_t>(t) * F;
            e.fb.apply(px.data() + static_cast<size_t>(t) * e.stft->bins(), row);
            for (int k = 0; k < F; k++) {
                row[k] = std::log(row[k] + mLogEps);
            }
        }
        ch++;
    }
}

} // namespace dtt
