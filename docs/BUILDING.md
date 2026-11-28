# Building

## Requirements

- CMake 3.24 or later, and Python 3 for `py/fetch.py` (standard library only).
- A C++23 compiler and standard library with `std::expected`: GCC 13+, Clang 19+ (with GCC's libstdc++, older Clang
  versions do not get `std::expected`; with libc++, Clang 17+), Apple Clang 15+ (Xcode 15), or MSVC 19.38+ (Visual
  Studio 2022 17.8).
- For the GPU: the Vulkan SDK (headers, loader, SPIR-V headers and `glslc`; on Ubuntu
  `apt install libvulkan-dev spirv-headers glslc`), or Xcode for Metal. Without either, the CPU backend is built.

ggml v1.0.0 is fetched at configure time (pinned by commit), and Catch2 v3.11.0 when the tests are built.

## Presets

Run from `cpp/`:

```bash
cmake --preset release && cmake --build --preset release
ctest --preset release
```

| Preset | What |
|---|---|
| `release` | Release, the default GPU backend: Metal on Apple, Vulkan when the SDK is found, else CPU only |
| `cpu` | Release, CPU only |
| `debug` | Debug |
| `macos-arm64`, `macos-x86_64` | One architecture per build tree (a universal build is two trees and `lipo`) |
| `windows-x64` | MSVC or clang-cl |

Build trees go next to `cpp/` (`build`, `build-cpu`, ...). Without presets: `cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release`.

## Options

| Option | Default | |
|---|---|---|
| `DETTATO_VULKAN` | ON when the Vulkan SDK is found (not on Apple) | the ggml Vulkan backend |
| `DETTATO_METAL` | ON on Apple | the ggml Metal backend (shaders embedded in the library) |
| `DETTATO_NATIVE` | OFF | build ggml for this machine's CPU only. Never for a binary you ship |
| `DETTATO_BUILD_TOOLS` | ON at top level | `dettato_cli`, `dettato_bench` and the parity tools |
| `DETTATO_BUILD_TESTS` | ON at top level | `dettato_tests` (Catch2) |
| `FETCHCONTENT_SOURCE_DIR_GGML` | | a local ggml checkout instead of fetching it |

The `DETTATO_VULKAN` / `DETTATO_METAL` / `DETTATO_NATIVE` options only apply when dettato.cpp fetches ggml itself. A host
that provides ggml configures it.

## Using it from another CMake project

The library is the static target `dettato_ggml`, with one public header, `dettato/dettato.hpp`.

```cmake
add_subdirectory(third_party/dettato.cpp/cpp)
target_link_libraries(my_app PRIVATE dettato_ggml)
```

A binary can only hold one copy of ggml. If the host already builds ggml (NeuralNote builds it for muscriptor.cpp), add
ggml first: `dettato.cpp` sees the `ggml` target and links against it instead of fetching its own. That ggml can be
v1.0.0, or the v0.23 fork muscriptor.cpp pins (DamRsn/ggml `b01bbb8`), so the two libraries can share one: add
muscriptor.cpp first and dettato.cpp builds on its ggml. CI builds and tests dettato.cpp on both. For the CPU backend, `GGML_LLAMAFILE=ON` makes the small f32 matmuls of the drum and grouper graphs 2-4x
faster.

`DETTATO_BUILD_TOOLS` and `DETTATO_BUILD_TESTS` default to OFF under `add_subdirectory`.

## Notes per platform

- **Linux**: tested with Vulkan on an NVIDIA RTX 5060 and on Intel Arc integrated graphics (Mesa), and with the CPU
  backend. The CPU build targets baseline x86-64;
  the hot host loops pick AVX2 / FMA / F16C at run time.
- **macOS**: CI builds the Metal backend on Apple Silicon, but GitHub's runners expose Apple's paravirtual GPU, which has
  no simdgroup operations, so ggml-metal runs no matmul there: the model tests run on the CPU (and check that Auto falls
  back to it). The first run on an Apple Silicon Mac (by Oli Larkin) found that the drum model's attention uses
  16-channel heads, below the smallest head size ggml-metal's flash attention supports (32), so with drums (audio at
  40 kHz or more) a transcription failed on Metal. His fix (pull request #1) pads those heads to 32 on every backend; the
  output on the CPU and on Vulkan is unchanged. The transcriber uses ggml's flash attention where the backend supports it
  and the explicit softmax otherwise.
- **Windows**: CI builds with MSVC, CPU only and with the Vulkan backend (Vulkan SDK 1.4.363, Ninja, from a Visual
  Studio developer prompt so that ggml's shader generator finds `cl`), and with clang-cl and lld-link, which is how
  NeuralNote builds it. It runs the tests on the CPU; Vulkan has not
  run on a Windows GPU yet. With its own ggml, dettato.cpp delay-loads `vulkan-1.dll` and checks for it before the first
  Vulkan call, so a Vulkan build still starts, on the CPU, on a machine without Vulkan; inside a host that brings ggml
  (NeuralNote, through muscriptor.cpp) the same check applies. With MSVC's `cl` the host loops use their portable scalar
  paths. With clang-cl they pick their FMA / AVX2 versions at run time from `cpuid`, because clang-cl links neither libgcc
  nor compiler-rt, which hold `__builtin_cpu_supports`'s CPU table.
