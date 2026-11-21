# Benchmarks

Measured on 2026-11-21 with the released build, on an RTX 5060 (8 GB) with an Intel Core Ultra 7 265K under Ubuntu,
over Vulkan with f16 weights, unless a row says otherwise.

## Accuracy

Four of the dettato benchmark sets that download quickly: Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic and RWC
Royalty-Free (49 recordings, 89 min). Scored with `py/bench/score.py`, the MuScriptor paper's rules (onset, frame and
offset F1 on pitched notes, drum-hit F1, the composite is their mean). To reproduce: `py/bench/download_subset.py`,
`py/bench/build_sets.py DOWNLOADS SETS bach10 dcs phenicx rwc_r`, then `py/bench/run_cpp_sets.sh SETS OUT MODEL.gguf`;
the Python package's run is `py/bench/run_dettato.py` (its `pipeline/` output).

| | composite (95 %) | onset | frame | offset | drums |
|---|---|---|---|---|---|
| Python package (kobimusic/dettato c4c52e0, CUDA) | 67.51 (65.00-70.05) | 69.31 | 82.16 | 53.16 | 65.39 |
| **dettato.cpp** | **67.74** (65.23-70.29) | 69.55 | 82.74 | 53.42 | 65.23 |

Each is inside the other's interval: these four sets do not separate the two.

Per set:

| set | recordings | onset | frame | offset | drums |
|---|---|---|---|---|---|
| Bach10 | 10 | 93.1 (package 92.7) | 93.8 (93.9) | 88.2 (88.1) | - |
| Dagstuhl ChoirSet | 20 | 57.0 (56.2) | 81.2 (79.2) | 44.7 (44.5) | - |
| PHENICX-Anechoic | 4 | 68.3 (68.5) | 79.5 (79.1) | 38.3 (37.5) | - |
| RWC Royalty-Free | 15 | 59.8 (59.8) | 76.4 (76.5) | 42.5 (42.5) | 65.2 (65.4) |

dettato.cpp differs from the package's defaults in two ways:

- **One transcriber pass.** The package runs the transcriber twice, once on the recording (for the notes) and once on
  the recording retuned to A440 (for the grouper). dettato.cpp checks the tuning first and runs the transcriber once, on
  the retuned audio, for both. On a recording that is in tune that is the same computation; on the sets that sit off
  A440 the notes improve (Dagstuhl ChoirSet frame F1 79.2 -> 81.2), and RWC-R, in tune, is unchanged.
- **Windows every 4 s.** The transcriber reads 8 s windows; the package moves them 2 s at a time, dettato.cpp 4 s. In the
  package itself that step leaves the composite where it was (67.51 with either) and moves about 2 % of the notes.

### Instrument tracks

`score.py` does not score the tracks, so they are measured two other ways.

Against the reference instruments (`py/bench/track_truth.py`: the adjusted Rand index between each note's track and its
reference instrument, over the pitched notes that match a reference note, averaged per set):

| | Bach10 | PHENICX-Anechoic | RWC Royalty-Free |
|---|---|---|---|
| Python package | 0.725 | 0.439 | 0.473 |
| Python package with windows every 4 s | 0.724 | 0.437 | 0.450 |
| **dettato.cpp** | 0.722 | 0.422 | 0.463 |

(Dagstuhl ChoirSet has one reference instrument, the voice, and every run scores 1.)

Against each other (`py/bench/compare_variants.py`, 49 recordings: the adjusted Rand index of the track assignment over
the notes both runs found, and the recordings whose sets of track names are the same):

| | track ARI | same track names |
|---|---|---|
| dettato.cpp vs the package | 0.950 | 40 / 49 |
| dettato.cpp vs the package with windows every 4 s | 0.949 | 42 / 49 |
| the package with windows every 4 s vs the package | 0.953 | 39 / 49 |

The tracks are the least stable part of the chain: the grouper's decode is a clustering, and a few notes that move can
change which clusters merge. dettato.cpp is as far from the package as the package's two window steps are from each
other, and all three score alike against the reference instruments. `docs/PARITY.md` has one song in detail.

## Speed

A 4-min song (240 s, 44.1 kHz stereo, nine instruments + drums), wall time of one `transcribe()` call after loading.

| | total | real-time factor |
|---|---|---|
| **dettato.cpp, RTX 5060 (Vulkan, f16)** | **12.2 s** | **19.7x** |
| dettato.cpp, CPU (Core Ultra 7 265K, 8 threads, f16) | 147-170 s | 1.4-1.6x |
| Python package, RTX 5060 (CUDA, bf16) | 20.7 s | 11.6x |
| MuScriptor small via muscriptor.cpp, RTX 5060 (Vulkan) | 28.2 s | 8.5x |
| MuScriptor medium via muscriptor.cpp, RTX 5060 (Vulkan) | 67.7 s | 3.5x |

dettato.cpp: 12.2-12.3 s over six runs. The package: 20.7 and 21.5 s after a warm-up run, with its defaults (windows
every 2 s, two passes); with windows every 4 s it ran the benchmark recordings at 16.7x real time against 9.1x. The
MuScriptor rows were measured on 2026-11-18 with muscriptor.cpp 8589219.

Over the 49 benchmark recordings: 18.5x real time, on a machine that was busy with other jobs during that run.

Where the time goes (`DETTATO_TIMING=1`): the transcriber 8.7 s, the grouper's features 1.4 s (most of it recomputing
the few windows that the store kept nothing for, near notes the repairer inserted), the grouper network and decode
1.4 s, resampling and the tuning check 0.55 s, the evidence repairer 0.15 s. The drums (2.7 s) run alongside the
repairer and the grouper on a second backend instance of the same GPU.

The transcriber network takes about 111 ms per 8 s window on the RTX 5060, two windows per call (128 ms with the front
end and reading the hidden states back; PyTorch with CUDA: about 92 ms for the network). The gap is elementwise passes
over large f32 tensors (bias adds, residuals, GELU, layout copies) that ggml's Vulkan backend does not fuse.

**CPU.** Core Ultra 7 265K on its 8 performance cores (the default thread count), f16 weights: 147-170 s over three
runs on a machine that was also running other jobs, of which the transcriber 114-135 s, the grouper 25 s and the drums
11 s. The Python package on the CPU: about 30 minutes (estimated from its first pass, not run to the end). ggml's CPU
backend slows down sharply when its threads share cores with other work (the drum graphs measured 8x slower at 20
threads on a busy machine), which is why the default is the physical performance cores and not every logical CPU.

**Integrated GPU.** On the same machine's Intel Arc graphics (Arrow Lake, Mesa Vulkan driver) the pipeline runs and
gives the CPU's notes (30 s clip: the same pitched notes, drum hits and tracks), but takes 231 s for the song, slower
than the CPU's performance cores. The automatic device choice still prefers an integrated GPU to the CPU.

## Memory

Weights on the device: 74 MB (f16) or 130 MB (f32). Peak resident memory of the process for the 4-min song: 1.4 GB on
the RTX 5060, 1.5 GB on the CPU (`docs/API.md` has the breakdown). Of that, the evidence repairer keeps 130 MB of the
transcriber's hidden states (only around candidate notes; the whole-song tensor the package keeps is 1.55 GB in f32)
and the grouper's store 230 MB (360 MB peak).
