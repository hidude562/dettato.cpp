#pragma once

// The grouper (dettato/grouper/core.py Grouper): every note into an instrument track, each track named by one of 73
// instruments.
//
//   notes + audio -> tuning normalisation (retune.hpp) -> transcriber pass 2 on that audio (hop 0.5, taps hidden + stem),
//   gathered per note while the windows stream (features.hpp) -> 24 s windows of notes through NoteNet (notenet.hpp, on the
//   ggml backend; front end on the CPU, front.hpp) -> per-note probabilities / embeddings -> decode.hpp -> tracks.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "grouper/decode.hpp"
#include "transcriber/transcriber.hpp"

namespace dtt
{

class Backend;
class GroupFront;
class ModelFile;
class Transcriber;
class NoteNet;

struct GroupTrack {
    std::string instrument;
    int program = 0;
    std::vector<std::array<double, 4>> notes; // onset s, offset s, pitch, velocity 1..127, sorted by onset
};

struct GroupInfo {
    double tuning_cents = 0, tuning_strength = 0, retune_cents = 0, retune_ratio = 1;
    int notes_in = 0, notes_out = 0;
};

/** The model's per-note inputs after pass 2 (core.py features / _Song). */
struct GroupFeatures {
    int n = 0;
    std::vector<float> meta;   // (n, 12) float32, notes sorted by (onset, pitch)
    std::vector<int8_t> feat;  // (n, 3, 368)
    std::vector<float> fsc;    // (n, 3), values rounded through float16
    std::vector<float> X;      // (n, 3, 368) the taps before quantisation (for the parity tests)
};

class Grouper
{
public:
    Grouper(const ModelFile& mf, const Backend& be);
    ~Grouper();
    Grouper(const Grouper&) = delete;
    Grouper& operator=(const Grouper&) = delete;

    /**
     * audio16k: the recording; notes: (onset, offset, pitch) in its time axis, any order; tr: a Transcriber built with both
     * the hidden and stem taps (pass 2 runs on it at hop_frac 0.5). Returns Grouper.group's tracks (largest first) and info.
     */
    std::vector<GroupTrack> group(const float* audio16k, size_t n, const std::vector<std::array<double, 3>>& notes, Transcriber& tr,
                             GroupInfo* info) const;

    // ------------------------------------------------------------------ the single-pass pipeline
    /** The tuning normalisation group() starts with: y is the audio on A440 (a copy of x16 when it is not retuned). */
    struct Retune {
        double delta = 0, strength = 0, shift = 0, r = 1;
        std::vector<float> y;
    };
    Retune retune(const float* x16, size_t n) const;

    /**
     * A sink for the one transcriber pass on y (both taps, any hop that contains the hop-0.5 windows: 0.5 or 0.25). It keeps
     * the float16-accumulated taps of the hop-0.5 windows at the (frame, lane) pairs a note could read (near onsets in the
     * final maps), so that featuresFromStore can gather them later; memory grows with the note frames.
     */
    std::unique_ptr<TapSink> makeStoreSink() const;

    /**
     * features(y, n, notes_y, tr) without a second pass: maps2 = that pass's hop-0.5 sub-maps (RunOptions::sub_maps,
     * sub_step 64000), store = the sink from makeStoreSink(). Notes whose frames the store lacks are gathered from a pass
     * over just the hop-0.5 windows covering them (tr.run with RunOptions::only). Bit-identical to features().
     */
    GroupFeatures featuresFromStore(const float* y, size_t n, const std::vector<std::array<double, 3>>& notes_y, TapSink& store,
                                    const Maps& maps2, Transcriber& tr) const;

    // ------------------------------------------------------------------ steps (exposed for the parity tests)
    /** Pass 2 + gather on audio y: notes (N, 3) on y's time axis. */
    GroupFeatures features(const float* y, size_t n, const std::vector<std::array<double, 3>>& notes, Transcriber& tr) const;
    /** Grouper.predict over a song: audio y (16 kHz) and its features. */
    GroupPredictions predict(const float* y, size_t n, const GroupFeatures& f) const;
    /** decode + Grouper._tracks: times are multiplied by r. */
    std::vector<GroupTrack> tracks(const GroupFeatures& f, const GroupPredictions& P, double r, GroupDecodeTrace* trace = nullptr) const;

    /** Optional hooks for the parity tool. */
    struct Debug {
        bool inject_retuned = false;     // use `retuned` / `r` instead of the C++ tuning estimate + resampler
        std::vector<float> retuned;
        double r = 1.0;
        bool exact_tuning_only = false;  // compute the tuning numbers but never resample (with inject_retuned = false)
        bool taps_f32 = false;           // accumulate pass 2's taps in float32 instead of the reference's float16
        // captured
        double delta = 0, strength = 0, shift = 0, ratio = 1;
        std::vector<float> y;            // the audio pass 2 ran on
        GroupFeatures features;
        GroupPredictions pred;
        GroupDecodeTrace trace;
        std::vector<int64_t> labels_inst, labels_track; // per note (meta order): decoded instrument / group index, -1 = none
        double t_tuning = 0, t_resample = 0, t_pass2 = 0, t_predict = 0, t_decode = 0;
        // featuresFromStore
        int store_notes_recomputed = 0, store_windows_recomputed = 0, store_windows_total = 0, store_windows_seen = 0;
        size_t store_pairs_h = 0, store_pairs_s = 0, store_bytes = 0, store_peak_bytes = 0, store_need_h = 0, store_need_s = 0;
        bool store_wsum_matches = false;
        double t_store_gather = 0, t_store_recompute = 0;
    };
    mutable Debug* debug = nullptr;

    const GroupFront& front() const { return *mFront; }
    const NoteNet& net() const { return *mNet; }
    NoteNet& net() { return *mNet; }
    int instruments() const { return static_cast<int>(mNames.size()); }
    const std::vector<std::string>& names() const { return mNames; }
    const std::vector<int>& programs() const { return mPrograms; }
    const GroupDecodeParams& decodeParams() const { return mDec; }
    double windowS() const { return mWindowS; }
    double windowHop() const { return mWindowHop; }
    int maxNotes() const { return mMaxNotes; }

private:
    const Backend& mBe;
    std::unique_ptr<NoteNet> mNet;
    std::unique_ptr<GroupFront> mFront;
    std::vector<std::string> mNames;
    std::vector<int> mPrograms;
    std::vector<double> mCredit;
    std::vector<TrackHeadLayer> mTh;
    GroupDecodeParams mDec;
    double mWindowS = 24.0, mWindowHop = 0.5;
    int mMaxNotes = 768;
    int mBatch = 4;
};

} // namespace dtt
