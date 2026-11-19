// The transcriber's peak-pick decode on hand-made activation maps. No weights needed.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "transcriber/decode.hpp"

namespace
{
constexpr int T = 200, L = 88;
constexpr double HOP = 0.02; // s per frame

struct Maps {
    std::vector<float> on = std::vector<float>(T * L, 0.0f);
    std::vector<float> fr = std::vector<float>(T * L, 0.0f);
    std::vector<float> off = std::vector<float>(T * L, 0.0f);

    /** A clean note on lane `lane` from frame t0 to frame t1 (the offset peak on t1). */
    void note(int lane, int t0, int t1)
    {
        on[t0 * L + lane] = 0.95f;
        if (t0 > 0) {
            on[(t0 - 1) * L + lane] = 0.4f;
        }
        for (int t = t0; t <= t1; t++) {
            fr[t * L + lane] = 0.9f;
        }
        off[t1 * L + lane] = 0.9f;
    }
    std::vector<dtt::DecodedNote> decode() const
    {
        return dtt::decodeVoice(on.data(), fr.data(), off.data(), T, L, HOP, dtt::DecodeParams{});
    }
};
} // namespace

TEST_CASE("silence decodes to no notes", "[pure][decode]")
{
    CHECK(Maps{}.decode().empty());
}

TEST_CASE("one clean note decodes to one note at its lane's pitch", "[pure][decode]")
{
    Maps m;
    m.note(39, 50, 100); // lane 39 = MIDI 60 (lanes start at A0 = 21)
    const auto notes = m.decode();
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].pitch == 60);
    CHECK(std::abs(notes[0].onset - 50 * HOP) <= HOP);
    CHECK(std::abs(notes[0].offset - 100 * HOP) <= 2 * HOP);
}

TEST_CASE("notes come out sorted by onset, one per peak", "[pure][decode]")
{
    Maps m;
    m.note(48, 120, 150); // MIDI 69
    m.note(39, 20, 60);   // MIDI 60
    m.note(43, 20, 80);   // MIDI 64, same onset
    const auto notes = m.decode();
    REQUIRE(notes.size() == 3);
    for (size_t i = 1; i < notes.size(); i++) {
        CHECK(notes[i - 1].onset <= notes[i].onset);
    }
    CHECK(notes.back().pitch == 69);
}

TEST_CASE("an onset peak without frame activity is a minimum-length note; one below threshold is nothing", "[pure][decode]")
{
    Maps m;
    m.on[30 * L + 10] = 0.95f; // above on_th (0.25), no frame activity: the note ends where the frame map is low
    auto notes = m.decode();
    REQUIRE(notes.size() == 1);
    CHECK(notes[0].pitch == 31);
    CHECK(std::abs(notes[0].offset - notes[0].onset - dtt::DecodeParams{}.min_dur) < 1e-12);

    Maps quiet;
    quiet.on[30 * L + 10] = 0.2f;
    quiet.fr[30 * L + 10] = 0.9f;
    CHECK(quiet.decode().empty());
}
