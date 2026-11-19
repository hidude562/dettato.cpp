# API

Everything is in `cpp/include/dettato/dettato.hpp`, namespace `dtt`. The shape follows muscriptor.cpp's
`msl::Transcriber`: a static `load()` returning `std::expected`, a blocking `transcribe()`, options structs with
`should_cancel` and `on_progress` callbacks, and a device list.

```c++
#include "dettato/dettato.hpp"

auto p = dtt::Pipeline::load("dettato-f16.gguf");
if (!p) {
    log(dtt::describe(p.error()));
    return;
}
dtt::TranscribeOptions o;
o.quality = dtt::Quality::Fast;
o.on_progress = [](const dtt::Progress& pr) { setProgress(pr.overall); };
o.should_cancel = [&] { return cancelRequested.load(); };
auto r = p->transcribe(samples, 44100.0, o);
if (!r) {
    log(dtt::describe(r.error()), p->lastErrorMessage());
    return;
}
for (const dtt::Note& n : r->notes) {
    const dtt::Track& t = r->tracks[n.track];
    // t.name, t.program, t.is_drum; n.onset, n.offset (s), n.pitch, n.velocity
}
```

## Devices

`availableDevices()` lists the GPUs ggml finds (Metal, or Vulkan devices in the loader's order), then the CPU, always
last. Each `Device` has a `name`, a `backend` (`"Metal"`, `"Vulkan"` or `"CPU"`), `integrated` and `memory_total`
(bytes, 0 when unknown). The list is built once per process; the first call can take a second (the Vulkan loader).

`LoadOptions::device` is an index into that list. Left empty, `load()` picks the first discrete GPU, else the first
integrated one, else the CPU. `LoadOptions::n_threads` (0 = all logical CPUs) sets the CPU backend's threads and the host
code's.

## Loading

`Pipeline::load(path, options)` reads the GGUF (all four networks) and uploads the weights: 74 MB in f16, about a second.
Errors:

| `Error` | when |
|---|---|
| `FileNotFound` | no file at `path` |
| `InvalidCheckpoint` | not a GGUF, or not a dettato one, or a tensor is missing |
| `UnsupportedCheckpointVersion` | a dettato GGUF of another format version (this library reads version 1) |
| `DeviceUnavailable` | the device index is out of range, the backend cannot start, or it cannot run one of the networks' operations |
| `Cancelled` | `should_cancel` returned true |

On a GPU, the pipeline creates two backend instances on the device: the drum model runs on the second one while the
evidence repairer and the grouper use the first. Their weights are shared.

## Transcribing

`transcribe(mono, sample_rate, options)` takes the whole recording as mono float32 at its own rate (any rate from 16 kHz;
downmix stereo first). It returns a `Transcription`:

- `tracks`: the pitched instrument tracks, largest first, then the drum track if there are drums. Each has a `name`
  ("piano", "alto sax", ...), a General MIDI `program` and `is_drum`.
- `notes`: every note, sorted by onset, with `onset` / `offset` in seconds from the start of the signal, `pitch` (MIDI;
  the GM percussion key for drums), `velocity` (1-127) and `track`. dettato does not predict the loudness of pitched
  notes: their velocity is the note's confidence mapped to 20-127, as in the package. The drum model predicts its hits'
  velocities.
- `tuning_cents`: the recording's offset from A440 in cents, positive when it is sharp. When it is clearly off, the networks read the audio
  retuned to A440 and the times are mapped back, so the notes are in the recording's time but at the nearest
  equal-tempered pitch.
- `drum_model`: `"drum-kit"` or `"orchestral"` (the drum model routes the recording to one of the two), empty when the
  drums did not run.
- `notes_transcribed`, `notes_dropped`, `notes_inserted`: what the evidence repairer did.

`TranscribeOptions`:

- `quality`: `Best` reads the transcriber's 8 s windows every 2 s (the package's setting), `Fast` every 4 s: about 40 %
  less time on a GPU and the same benchmark score (`docs/BENCHMARK.md`).
- `drums`: run the drum model. It reads the 7.8-20 kHz band, so it only runs when `sample_rate` is at least 40 kHz.
  Below that, there is no drum track.
- `should_cancel`: polled between windows and between stages; returning true ends the call with `Error::Cancelled`.
- `on_progress`: `Progress{stage, stage_fraction, overall}`. The stages are `Transcribing` (the tuning check and the
  transcriber), `Repairing`, `Grouping` and `Drums`; `overall` weights them by their typical share of the time on a GPU
  and never goes backwards.

Errors: `InvalidArgument` for a sample rate below 16 kHz (or not finite), `Cancelled`, `Internal` for anything else (with
the reason in `lastErrorMessage()`, also set for `InvalidArgument`).

## Threads and memory

One `transcribe()` at a time per `Pipeline`; separate `Pipeline`s are independent. `transcribe()` is blocking (about 12 s
for a 4-min song on an RTX 5060 in Fast mode, a few minutes on a CPU), so a plugin calls it from a worker thread.

Peak resident memory of `dettato_cli` (measured, f16 weights): on the RTX 5060, 1.2 GB for a 3 s clip (the Vulkan
driver and fixed buffers) and 2.7 GB for a 4-min song; on the CPU, 2.7 GB already for a 3 s clip, because the compute
buffers of the networks live in host memory there. What grows with the length of the recording is host memory: copies
of the audio, the transcriber's output maps, the evidence repairer's store of hidden states around candidate notes
(130 MB for 4 minutes) and the grouper's store (up to 360 MB). Bringing these down is on the list.

## Environment variables

- `DETTATO_TIMING=1`: per-stage wall times on stderr.
- `DETTATO_FLASH_ATTN=0|1`: turn ggml's flash attention off or on for the transcriber's attention (default: on for GPUs;
  off for the CPU, where the explicit softmax is exact and as fast). A backend without flash attention for these shapes
  falls back to the explicit path by itself.
- `GGML_NO_BACKTRACE=1`: a failed ggml assertion prints its message instead of trying to attach a debugger.
