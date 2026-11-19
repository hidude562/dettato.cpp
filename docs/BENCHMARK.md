# Benchmarks

## Accuracy

Four of the dettato benchmark sets that download quickly: Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic and RWC
Royalty-Free (49 recordings, 89 min). Scored with `py/bench/score.py`, the MuScriptor paper's rules (onset, frame and
offset F1 on pitched notes, drum-hit F1, the composite is their mean). To reproduce: `py/bench/download_subset.py`,
`py/bench/build_sets.py DOWNLOADS SETS bach10 dcs phenicx rwc_r`, then `py/bench/run_cpp_sets.sh SETS OUT MODEL.gguf
[--fast]`; the Python package's run is `py/bench/run_dettato.py` (its `pipeline/` output).

| | composite | onset | frame | offset | drums |
|---|---|---|---|---|---|
| Python package (kobimusic/dettato c4c52e0, CUDA) | 67.51 | 69.31 | 82.16 | 53.16 | 65.39 |
| **dettato.cpp, Best** (RTX 5060, Vulkan, f16) | **67.75** | 69.63 | 82.80 | 53.35 | 65.23 |
| **dettato.cpp, Fast** | **67.74** | 69.55 | 82.74 | 53.42 | 65.23 |

All three are inside each other's 95 % intervals (about ±2.5 composite). The C++ pipeline differs from the package in one
place on purpose: the package runs the transcriber twice, once on the recording (for the notes) and once on the recording
retuned to A440 (for the grouper); dettato.cpp checks the tuning first and runs the transcriber once, on the retuned
audio, for both. On a recording that is in tune that is the same computation. On the detuned sets the notes improve
(Dagstuhl ChoirSet frame F1 79.0 -> 81.2, PHENICX 78.9 -> 79.7); RWC-R, in tune, is unchanged.

`score.py` does not score the instrument tracks. Against the package's tracks (`py/bench/compare_variants.py`, adjusted
Rand index over the notes both runs found, 49 recordings): 0.960, and the same set of instrument names on 41 of 49
recordings. On the 4-min test song (below) the package and dettato.cpp write the same nine instruments, with counts
within a few notes (`docs/PARITY.md`).

**Fast** reads the transcriber's 8 s windows every 4 s instead of every 2 s. On the package's own pipeline that changed
the composite by 0.00 (67.51 vs 67.51) and moved ~2 % of the notes, neutral on average; here, 67.74 vs 67.75.

## Speed

A 4-min song (240 s, 44.1 kHz stereo, nine instruments + drums), wall time of one `transcribe()` call after loading.
RTX 5060 (8 GB) with an Intel Core Ultra 7 265K, Ubuntu.

| | total | real-time factor |
|---|---|---|
| **dettato.cpp Fast, RTX 5060 (Vulkan, f16)** | **11.7 s** | **20.5x** |
| dettato.cpp Best, RTX 5060 (Vulkan, f16) | 19.4 s | 12.4x |
| Python package, RTX 5060 (CUDA, bf16) | 20.1 s | 11.9x |
| MuScriptor small via muscriptor.cpp, RTX 5060 (Vulkan) | 28.2 s | 8.5x |
| MuScriptor medium via muscriptor.cpp, RTX 5060 (Vulkan) | 67.7 s | 3.5x |

Over the 49 benchmark recordings: Fast 19.8x real time, Best 12.1x.

Where the time goes in Fast mode (`DETTATO_TIMING=1`): the transcriber 8.5 s, the grouper's features 1.5 s (most of it
recomputing the few windows that the store kept nothing for, near notes the repairer inserted), the grouper network and
decode 1.2 s, resampling and the tuning check 0.5 s, the evidence repairer 0.2 s. The drums (2.5 s) run alongside the
repairer and the grouper on a second backend instance of the same GPU.

The transcriber network runs at about 119 ms per 8 s window on the RTX 5060 (PyTorch with CUDA: ~92 ms). The gap is
elementwise passes over large f32 tensors (bias adds, residuals, GELU, layout copies) that ggml's Vulkan backend does not
fuse.

**CPU.** Measured before the one-pass change (the transcriber ran twice): Core Ultra 7 265K, 8 threads, f16 weights,
Best 323 s (0.74x real time), Fast 226 s (1.06x). The one-pass flow removes the grouper's own transcriber pass (about
100 s of that). The Python package on the CPU: about 30 minutes. ggml's CPU backend slows down sharply when its threads
share cores with other work (the drum graphs measured 8x slower at 20 threads on a busy machine): give it the physical
performance cores you can spare, not every logical CPU.

## Memory

Weights on the device: 74 MB (f16) or 130 MB (f32). On the host, for a 4-min song: the evidence repairer keeps 130 MB of
the transcriber's hidden states (only around candidate notes; the whole-song tensor the package keeps is 1.55 GB in
f32), the grouper's store 230 MB (360 MB peak).
