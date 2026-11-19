#include "drums/drums.hpp"

#include <chrono>
#include <cmath>
#include <limits>

#include "backend.hpp"
#include "common.hpp"
#include "drums/decode.hpp"
#include "drums/features.hpp"
#include "drums/nets.hpp"
#include "model_file.hpp"

namespace dtt
{

DrumTranscriber::DrumTranscriber(const ModelFile& mf, const Backend& be)
    : mFeatures(std::make_unique<DrumFeatures>(mf)), mNets(std::make_unique<DrumNets>(mf, be))
{
    mSampleRate = static_cast<int>(mf.u32("dr.sample_rate"));
    mMinSourceRate = static_cast<int>(mf.u32("dr.min_source_rate"));
    mThreads = be.threads();
    mDecoder = std::make_unique<DrumDecoder>(mf);
}

DrumTranscriber::~DrumTranscriber() = default;

std::vector<DrumHit> DrumTranscriber::transcribe(const float* audio44k, size_t n, DrumInfo* info) const
{
    return transcribe(audio44k, n, info, nullptr, nullptr, nullptr, false);
}

std::vector<DrumHit> DrumTranscriber::transcribe(const float* audio44k, size_t n, const float* audio16k, size_t n16, DrumInfo* info) const
{
    if (audio16k == nullptr) {
        return transcribe(audio44k, n, info);
    }
    // x16 = resample(x44 * gain) = gain * resample(x44): the features multiply by the gain as numpy does (float32)
    const float g32 = static_cast<float>(DrumFeatures::levelGain(audio44k, n));
    std::vector<float> x16(n16);
    for (size_t i = 0; i < n16; i++) {
        x16[i] = audio16k[i] * g32;
    }
    return transcribe(audio44k, n, info, &x16, nullptr, nullptr, false);
}

std::vector<DrumHit> DrumTranscriber::transcribe(const float* audio44k, size_t n, DrumInfo* info, const std::vector<float>* x16,
                                                 DrumTaps* taps, DrumTiming* timing, bool all_models) const
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    if (n < static_cast<size_t>(mSampleRate / 10)) {
        if (info != nullptr) {
            info->router = std::numeric_limits<float>::quiet_NaN();
            info->model.clear();
        }
        return {};
    }
    int T = 0;
    const auto feats = mFeatures->compute(audio44k, n, T, x16, nullptr, mThreads);
    const auto t1 = Clock::now();
    if (timing != nullptr) {
        timing->features += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const DrumProbs P = mNets->run(feats.data(), T, all_models, taps, timing);
    const auto t2 = Clock::now();
    std::string model;
    auto hits = mDecoder->decode(P, model);
    if (timing != nullptr) {
        timing->decode += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();
        timing->total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    if (info != nullptr) {
        info->router = static_cast<float>(P.router);
        info->model = model;
    }
    return hits;
}

} // namespace dtt
