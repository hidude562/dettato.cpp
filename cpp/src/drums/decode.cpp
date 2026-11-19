#include "drums/decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>

#include "common.hpp"
#include "drums/nets.hpp"
#include "model_file.hpp"

namespace dtt
{

DrumDecoder::DrumDecoder(const ModelFile& mf, const std::string& p)
{
    const auto k = mf.i32s(p + ".keys");
    const auto tk = mf.i32s(p + ".timpani_keys");
    keys.assign(k.begin(), k.end());
    timpani_keys.assign(tk.begin(), tk.end());
    th_kit = mf.f32(p + ".th.drum_kit");
    th_drum = mf.f32(p + ".th.orchestral_drum");
    th_timpani = mf.f32(p + ".th.timpani");
    th_router = std::strtod(mf.str(p + ".th.router_repr").c_str(), nullptr);
    gap = static_cast<int>(mf.u32(p + ".decode.peak_gap_frames"));
    fps = static_cast<int>(mf.u32(p + ".output_fps"));
    timpani_repeat = std::strtod(mf.str(p + ".decode.timpani_repeat_s").c_str(), nullptr);
    hit_length = std::strtod(mf.str(p + ".decode.hit_length_s").c_str(), nullptr);
}

void DrumDecoder::peaks(const float* prob, int T, int L, float th, int gap, std::vector<int>& fr, std::vector<int>& ln)
{
    fr.clear();
    ln.clear();
    std::vector<long> last(L, -10);
    std::vector<char> cand(L);
    for (int t = 0; t < T; t++) {
        const float* p = prob + static_cast<size_t>(t) * L;
        const float* left = prob + static_cast<size_t>(t > 0 ? t - 1 : 0) * L;      // the first row repeated
        const float* right = prob + static_cast<size_t>(t + 1 < T ? t + 1 : T - 1) * L; // the last row repeated
        for (int l = 0; l < L; l++) {
            const bool c = p[l] >= th && p[l] >= left[l] && p[l] >= right[l];
            if (c && t - last[l] > gap - 1) {
                fr.push_back(t);
                ln.push_back(l);
                last[l] = t;
            }
        }
    }
}

double DrumDecoder::pyRound(double x, int nd)
{
    // glibc's printf converts the exact binary value and rounds half to even, as Python's round() does
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", nd, x);
    return std::strtod(buf, nullptr);
}

std::vector<DrumHit> DrumDecoder::decode(const DrumProbs& P, std::string& model, std::vector<Source>* sources) const
{
    std::vector<Source> src;
    std::vector<double> on;
    std::vector<int> ks;
    std::vector<double> vs; // velocity before transcribe()'s clip (float32 values; timpani: already clipped to 0.3..1)
    std::vector<int> fr, ln;
    if (P.router < th_router) {
        DTT_CHECK(!P.kit.empty(), "the router picked the drum-kit model but it did not run");
        peaks(P.kit.data(), P.N, P.K, th_kit, gap, fr, ln);
        for (size_t i = 0; i < fr.size(); i++) {
            on.push_back(static_cast<double>(fr[i]) / fps);
            ks.push_back(keys[ln[i]]);
            vs.push_back(P.kit_vel[static_cast<size_t>(fr[i]) * P.K + ln[i]]);
            src.push_back({0, fr[i], ln[i], vs.back()});
        }
        model = "drum-kit";
    } else {
        peaks(P.drum.data(), P.N, P.K, th_drum, gap, fr, ln);
        std::map<int, std::vector<double>> seen;
        for (size_t i = 0; i < fr.size(); i++) {
            const double o = static_cast<double>(fr[i]) / fps;
            on.push_back(o);
            ks.push_back(keys[ln[i]]);
            vs.push_back(P.vel[static_cast<size_t>(fr[i]) * P.K + ln[i]]);
            src.push_back({1, fr[i], ln[i], vs.back()});
            seen[keys[ln[i]]].push_back(o);
        }
        peaks(P.timpani.data(), P.N, P.KT, th_timpani, gap, fr, ln);
        for (size_t i = 0; i < fr.size(); i++) {
            const double o = static_cast<double>(fr[i]) / fps;
            const int k = timpani_keys[ln[i]];
            bool repeat = false;
            for (double q : seen[k]) {
                repeat = repeat || std::abs(o - q) <= timpani_repeat;
            }
            if (repeat) {
                continue;
            }
            on.push_back(o);
            ks.push_back(k);
            // np.clip of a float32 value with Python floats stays float32
            const float v = P.timpani[static_cast<size_t>(fr[i]) * P.KT + ln[i]];
            vs.push_back(std::min(std::max(v, 0.3f), 1.0f));
            src.push_back({2, fr[i], ln[i], vs.back()});
        }
        model = "orchestral";
    }
    std::vector<size_t> order(on.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return on[a] < on[b]; });
    // transcribe(): np.clip(v, 0.05, 1.0). Without timpani hits the velocities are a float32 array (clipped in float32),
    // with them a float64 one; the rounding to 3 decimals gives the same value either way.
    std::vector<DrumHit> hits;
    hits.reserve(order.size());
    for (size_t i : order) {
        const double v = std::min(std::max(vs[i], 0.05), 1.0);
        hits.push_back({pyRound(on[i], 4), pyRound(on[i] + hit_length, 4), ks[i], static_cast<float>(pyRound(v, 3))});
        if (sources != nullptr) {
            sources->push_back(src[i]);
        }
    }
    return hits;
}

} // namespace dtt
