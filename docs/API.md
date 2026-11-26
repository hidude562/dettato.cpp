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
(bytes, 0 when unknown). The list is built once per process; the first call can take a second (the Vulkan loader). On
Windows, a build with the Vulkan backend on a machine without a Vulkan loader (`vulkan-1.dll`) lists the CPU alone.

`LoadOptions::device` is an index into that list. Left empty, `load()` picks the first discrete GPU, else the first
integrated one, else the CPU; when that GPU cannot run one of the networks (or cannot start), it loads on the CPU instead
and `error_detail` says why. A device chosen by index is never swapped: it loads there or fails with
`DeviceUnavailable`. `LoadOptions::n_threads` sets the CPU backend's threads and the host
code's. 0 = one per physical performance core the process may use (not efficiency cores or hyper-thread siblings, and
within a container's CPU quota): ggml's threads wait for each other at every graph node, so a slow or shared core holds
all of them back, and a plugin's host needs cores of its own.

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

`LoadOptions::error_detail`, when set, receives the reason for a failure (for instance the ggml operation a device cannot
run), for logs.

On a GPU, the pipeline creates two backend instances on the device: the drum model runs on the second one while the
evidence repairer and the grouper use the first. Their weights are shared.

## Transcribing

`transcribe(mono, sample_rate, options)` takes the whole recording as mono float32 at its own rate (any rate from 16 kHz;
downmix stereo first). Samples that are not finite (a damaged file) are read as silence, and an empty or very short
signal gives an empty transcription, not an error. It returns a `Transcription`:

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

- `drums`: run the drum model. It reads the 7.8-20 kHz band, so it only runs when `sample_rate` is at least 40 kHz.
  Below that, there is no drum track.
- `should_cancel`: returning true ends the call with `Error::Cancelled`. Polled between the stages, before every
  network call, after each batch of windows, in the drum model's host steps, and on the CPU also inside the networks at
  every graph node, so keep it cheap (reading a flag). The drums run on a second thread on a GPU and poll it too, but it
  is never called from two threads at once. A cancel returns once the network call in flight on each thread has
  finished: measured 0.0-0.1 s on the CPU and on an RTX 5060, in every stage. On a slower or busy GPU one call can take
  seconds.
- `on_progress`: `Progress{stage, stage_fraction, overall}`, reported per batch of windows in the transcriber and the
  grouper. The stages are `Transcribing` (the tuning check and the
  transcriber), `Repairing`, `Grouping` and `Drums`; `overall` weights them by their typical share of the time on a GPU
  and never goes backwards.

Errors: `InvalidArgument` for a sample rate below 16 kHz (or not finite), `Cancelled`, `Internal` for anything else (with
the reason in `lastErrorMessage()`, also set for `InvalidArgument`).

## Threads and memory

One `transcribe()` at a time per `Pipeline`; separate `Pipeline`s are independent. `transcribe()` is blocking (about 12 s
for a 4-min song on an RTX 5060, a few minutes on a CPU), so a plugin calls it from a worker thread.

Peak resident memory of `dettato_cli` (measured, f16 weights):

| | 30 s clip | 4-min song | 12-min recording |
|---|---|---|---|
| RTX 5060 (Vulkan) | 0.8 GB | 1.4 GB | 2.8 GB |
| CPU | 1.0 GB | 1.5 GB | not measured |

On the CPU the networks' compute buffers are host memory (about 0.5 GB each for the transcriber and the grouper, more
for the drums on a long song), so the pipeline frees each one after its stage. What grows with the length of the
recording: copies of the audio, the transcriber's output maps, the evidence repairer's store of hidden states around
candidate notes (130 MB for 4 minutes), the grouper's store (up to 360 MB) and the drum model, which reads the whole
song at once.

## Environment variables

- `DETTATO_TIMING=1`: per-stage wall times on stderr.
- `DETTATO_FLASH_ATTN=0|1`: turn ggml's flash attention off or on for the transcriber's attention (default: on for GPUs;
  off for the CPU, where the explicit softmax is exact and as fast). A backend without flash attention for these shapes
  falls back to the explicit path by itself.
- `GGML_NO_BACKTRACE=1`: a failed ggml assertion prints its message instead of trying to attach a debugger.
