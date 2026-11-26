# dettato.cpp

A C++ port of [dettato](https://github.com/kobimusic/dettato), KobiMusic's audio to MIDI transcriber, on
[ggml](https://github.com/ggml-org/ggml). It runs all four models of dettato with no Python, on the CPU, Metal or Vulkan. I
made it so dettato can be a backend for [NeuralNote](https://github.com/DamRsn/NeuralNote) next to
[muscriptor.cpp](https://github.com/DamRsn/muscriptor.cpp), so the API is close to that one.

You give it a recording and it gives back the notes in instrument tracks, plus a drum track. A 4 minute song takes 12.2
seconds on an RTX 5060. The Python package takes 20.7 seconds on the same card.

## Status

All four models are ported. I checked each one against the PyTorch package ([docs/PARITY.md](docs/PARITY.md)). On the CPU
in float32 each stage matches to about 1e-6. On four of the dettato benchmark sets it scores 67.74 composite F1. The
Python package scores 67.51. That difference is inside the error bars, so I would call them equal.

The output is close to the package, but it is not the same, because I changed a few parts on purpose:

- The Python package runs the transcriber twice. It runs it on the recording for the notes, and then on the recording
  tuned to A440 for the grouper. dettato.cpp checks the tuning first, so the transcriber only does one pass, on the tuned
  audio. For a song that is in tune this is the same. For a song that is off A440 the notes get better. For instance, the
  frame F1 on Dagstuhl ChoirSet goes from 79.2 to 81.2.
- The transcriber reads the song in windows of 8 seconds. The package moves the window by 2 seconds each time and
  dettato.cpp moves it by 4. This is faster, and the benchmark score stays the same.

On a 4 minute song 98% of the notes are the same as the package. The instrument tracks change more than the notes. If a
few notes change, a group of notes could all move to another track. The package has this too when its window step
changes. I also scored the tracks against the true instruments of the benchmark recordings, and they score close to the
package ([docs/BENCHMARK.md](docs/BENCHMARK.md)).

I ran it on Linux with Vulkan (an RTX 5060 and Intel integrated graphics) and on the CPU. CI builds and tests it on macOS
and on Windows, but only on the CPU, since the CI machines have no GPU that ggml could use. Oli Larkin ran it on an
Apple Silicon Mac, and found that the drum model could not run on Metal, so a song with drums failed there. His fix is
merged now (pull request #1). Vulkan has not run on a Windows GPU yet.

## Speed

A 4 minute song (nine instruments and drums) on an RTX 5060:

| | time | speed |
|---|---|---|
| **dettato.cpp** | **12.2 s** | **19.7x** |
| dettato Python package (CUDA) | 20.7 s | 11.6x |
| MuScriptor small, muscriptor.cpp | 28.2 s | 8.5x |
| MuScriptor medium, muscriptor.cpp | 67.7 s | 3.5x |

The Python package is timed the way it ships, with the 2 second step and both passes. The drums run on the GPU at the
same time as the grouper. On the CPU the same song takes about 2.5 minutes with 8 threads.
[docs/BENCHMARK.md](docs/BENCHMARK.md) has the details.

## Build

You need CMake 3.24 and a C++23 compiler (GCC 13, Clang 19 or Visual Studio 2022). For the GPU you need the Vulkan SDK, or
Xcode for Metal. Without them it builds for the CPU.

```bash
cd cpp
cmake --preset release
cmake --build --preset release --parallel 4
```

It fetches ggml v1.0.0 when it configures. [docs/BUILDING.md](docs/BUILDING.md) has the presets and options.

## Weights

The weights are one GGUF file with all four models (74 MB in float16). They are on Hugging Face at
[kobimusic/dettato-gguf](https://huggingface.co/kobimusic/dettato-gguf).

```bash
python3 py/fetch.py models/
```

This downloads `dettato-f16.gguf` and checks its sha256. You could also make the same file yourself from the safetensors
with `py/convert.py`.

## Use

```c++
#include "dettato/dettato.hpp"

auto p = dtt::Pipeline::load("models/dettato-f16.gguf");   // picks a GPU if there is one
if (!p) return fail(dtt::describe(p.error()));

auto r = p->transcribe(samples, 44100.0);                   // mono float32 at the recording's own rate
for (const dtt::Note& n : r->notes) {
    const dtt::Track& t = r->tracks[n.track];               // t.name, t.program, t.is_drum
}
```

`transcribe()` blocks until it is done, so a plugin would call it from a worker thread. It has a progress callback and a
cancel callback, like muscriptor.cpp. The drums only run when the recording is 40 kHz or more. [docs/API.md](docs/API.md)
has the rest.

## Command Line

```bash
build/tools/dettato_cli models/dettato-f16.gguf song.wav -o out/
```

It writes `out/song.mid` and `out/song.json`. It only reads WAV, so I would convert other formats with ffmpeg first.

### Filter CLI output by instrument

Use `--only drums` to write only drum tracks, or `--only bass,piano` to keep
tracks whose instrument names contain either fragment (case-insensitive).
Filtering applies to MIDI output, JSON track entries, and the printed output
track/note counts. It does not skip pitched-instrument inference. Drum inference
is skipped when no requested fragment can match "drums"; `--no-drums` still
takes precedence. An empty filter is a usage error. If no tracks match, that
input fails without writing output and reports the available track names.

## Using It From CMake

```cmake
add_subdirectory(third_party/dettato.cpp/cpp)
target_link_libraries(my_app PRIVATE dettato_ggml)
```

If the host already builds ggml (NeuralNote does for muscriptor.cpp), add ggml first. Then dettato.cpp links to that ggml
instead of fetching a second copy, since a binary can only have one ggml. It works with ggml v1.0.0 and with the older ggml
that muscriptor.cpp uses, so NeuralNote can build both on one ggml.

## Docs

| | |
|---|---|
| [API.md](docs/API.md) | the API, devices, errors, threads and memory |
| [BUILDING.md](docs/BUILDING.md) | presets, options, using it as a subproject |
| [MODEL.md](docs/MODEL.md) | the four models, what runs where, the GGUF format |
| [BENCHMARK.md](docs/BENCHMARK.md) | accuracy and speed |
| [PARITY.md](docs/PARITY.md) | how close each stage is to the PyTorch package |
| [TESTING.md](docs/TESTING.md) | the tests and the parity tools |
| [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | how the models map onto ggml |

## License

Apache-2.0, the same as dettato's code and weights. It includes pocketfft (BSD-3-Clause) and fetches ggml (MIT) and Catch2
(BSL-1.0) when it builds.
