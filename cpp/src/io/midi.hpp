#pragma once

#include <string>
#include <vector>

namespace dtt
{

struct MidiNote {
    double onset;  // s
    double offset; // s
    int pitch;
    int velocity; // 1..127
};

struct MidiTrack {
    std::string name;
    int program = 0; // 0..127, ignored for drums
    bool is_drum = false;
    std::vector<MidiNote> notes;
};

/**
 * A type-1 Standard MIDI File at 120 BPM (times in seconds map to ticks at 480 per quarter note), one MTrk per track
 * after a tempo track. Melodic tracks get channels 0-8, 10-15 in order (reused past 15 tracks); drums channel 9.
 */
std::vector<unsigned char> writeMidi(const std::vector<MidiTrack>& tracks);
void saveMidi(const std::string& path, const std::vector<MidiTrack>& tracks);

} // namespace dtt
