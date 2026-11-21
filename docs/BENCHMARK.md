# Benchmarks

## Accuracy

Four of the dettato benchmark sets that download quickly: Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic and RWC
Royalty-Free (49 recordings, 89 min). Scored with `py/bench/score.py`, the MuScriptor paper's rules (onset, frame and
offset F1 on pitched notes, drum-hit F1, the composite is their mean). To reproduce: `py/bench/download_subset.py`,
`py/bench/build_sets.py DOWNLOADS SETS bach10 dcs phenicx rwc_r`, then `py/bench/run_cpp_sets.sh SETS OUT MODEL.gguf`;
the Python package's run is `py/bench/run_dettato.py` (its `pipeline/` output).

| | composite (95 %) | onset | frame | offset | drums |
|---|---|---|---|---|---|
| Python package (kobimusic/dettato c4c52e0, CUDA) | 67.51 (65.00-70.05) | 69.31 | 82.16 | 53.16 | 65.39 |
| **dettato.cpp** (RTX 5060, Vulkan, f16) | **67.74** (65.23-70.29) | 69.55 | 82.74 | 53.42 | 65.23 |

Per set:

| set | recordings | onset | frame | offset | drums |
|---|---|---|---|---|---|
| Bach10 | 10 | 93.1 (package 92.7) | 93.8 (93.9) | 88.2 (88.1) | - |
| Dagstuhl ChoirSet | 20 | 57.0 (56.2) | 81.2 (79.2) | 44.7 (44.5) | - |
| PHENICX-Anechoic | 4 | 68.3 (68.5) | 79.5 (79.1) | 38.3 (37.5) | - |
| RWC Royalty-Free | 15 | 59.8 (59.8) | 76.4 (76.5) | 42.5 (42.5) | 65.2 (65.4) |

The package runs the transcriber twice, once on the recording (for the notes) and once on the recording retuned to A440
(for the grouper). dettato.cpp checks the tuning first and runs the transcriber once, on the retuned audio, for both.
On a recording that is in tune that is the same computation; on the sets that sit off A440 the notes improve (Dagstuhl
ChoirSet frame F1 79.2 -> 81.2), and RWC-R, in tune, is unchanged.

`score.py` does not score the instrument tracks. Against the package's tracks (`py/bench/compare_variants.py`, adjusted
Rand index over the notes both runs found, 49 recordings): 0.950, and the same set of instrument names on 40 of 49
recordings. On the 4-min test song (below) the package and dettato.cpp write the same nine instruments, with counts
within a few notes (`docs/PARITY.md`).

## Speed

A 4-min song (240 s, 44.1 kHz stereo, nine instruments + drums), wall time of one `transcribe()` call after loading.
RTX 5060 (8 GB) with an Intel Core Ultra 7 265K, Ubuntu.

| | total | real-time factor |
|---|---|---|
| **dettato.cpp, RTX 5060 (Vulkan, f16)** | **11.7 s** | **20.5x** |
| dettato.cpp, CPU (Core Ultra 7 265K, 8 threads, f16) | 147-170 s | 1.4-1.6x |
| Python package, RTX 5060 (CUDA, bf16) | 20.1 s | 11.9x |
| MuScriptor small via muscriptor.cpp, RTX 5060 (Vulkan) | 28.2 s | 8.5x |
| MuScriptor medium via muscriptor.cpp, RTX 5060 (Vulkan) | 67.7 s | 3.5x |

Over the 49 benchmark recordings: 19.8x real time.

Where the time goes (`DETTATO_TIMING=1`): the transcriber 8.5 s, the grouper's features 1.5 s (most of it
recomputing the few windows that the store kept nothing for, near notes the repairer inserted), the grouper network and
decode 1.2 s, resampling and the tuning check 0.5 s, the evidence repairer 0.2 s. The drums (2.5 s) run alongside the
repairer and the grouper on a second backend instance of the same GPU.

The transcriber network runs at about 119 ms per 8 s window on the RTX 5060 (PyTorch with CUDA: ~92 ms). The gap is
elementwise passes over large f32 tensors (bias adds, residuals, GELU, layout copies) that ggml's Vulkan backend does not
fuse.

**CPU.** Core Ultra 7 265K on its 8 performance cores (the default thread count), f16 weights: 147-170 s over three
runs on a machine that was also running other jobs, of which the transcriber 114-135 s, the grouper 25 s and the drums
11 s. The Python package on the CPU: about 30 minutes. ggml's CPU backend slows down sharply when its threads share
cores with other work (the drum graphs measured 8x slower at 20 threads on a busy machine), which is why the default
is the physical performance cores and not every logical CPU.

**Integrated GPU.** On the same machine's Intel Arc graphics (Arrow Lake, Mesa Vulkan driver) the pipeline runs and
gives the CPU's notes (30 s clip: the same pitched notes, drum hits and tracks), but takes 231 s for the song, slower
than the CPU's performance cores.

## Memory

Weights on the device: 74 MB (f16) or 130 MB (f32). Peak resident memory of the process for the 4-min song: 1.4 GB on
the RTX 5060, 1.5 GB on the CPU (`docs/API.md` has the breakdown). Of that, the evidence repairer keeps 130 MB of the
transcriber's hidden states (only around candidate notes; the whole-song tensor the package keeps is 1.55 GB in f32)
and the grouper's store 230 MB (360 MB peak).
