# Parity with the PyTorch package

Reference: kobimusic/dettato at c4c52e0, run on the CPU in float32 (`py/dump_refs*.py`). C++: `dettato_parity*` tools
(`docs/TESTING.md`). "CPU f32" = the ggml CPU backend with f32 weights; "Vulkan f16" = the RTX 5060 with f16 weights.
Tolerances were never widened to pass; every remaining difference is traced below.

Each stage was checked against the package's own flow, in which the grouper runs its own transcriber pass. The pipeline
now runs the transcriber once and the grouper reads that pass through a store; `dettato_parity_grouper` checks that the
store's features are bit-identical to the grouper's own pass (`[store]`), so the per-stage results below carry over. The
one-pass flow differs from the package only on recordings that are off A440 (`docs/BENCHMARK.md`).

## Transcriber (`dettato_parity`)

| | CPU f32 | Vulkan f16 |
|---|---|---|
| HCQT | 1.8e-6 rel L2 | 1.6e-7 |
| every stage (stem ... logits) | 2e-7 – 8e-7 | 2e-4 – 1.3e-3 (Vulkan's matmul/conv round f32 through f16) |
| maps | 1e-7 – 2e-6 | ~1e-3 |
| notes, 30 s mix | 744 / 744 identical | 739 / 744 (the rest on a decision threshold) |

## Evidence repairer (`dettato_parity_evidence`)

- Network fed the reference evidence: 1.3e-7 – 5.6e-7 per stage; host parts bit-exact (numpy's float32 log, pairwise
  sums, torch's softmax/sigmoid arithmetic reproduced); repaired notes identical (728/728, 729/729, 4497/4497).
- Full chain (C++ transcriber -> sink -> repair), CPU f32: 30 s mix 729/729 identical to the reference's f32-accumulation
  variant; 4-min song 4493/4497. The 4: Python's mean is a difference of float32 prefix sums reaching ~2e5 (one step
  1.6e-2); the transcriber's ~1e-7 differences flip those roundings (every evidence difference is within that bound;
  one insertion scores 0.6000 vs Python's 0.6002 at the 0.6 threshold).
- Python's own float16 accumulation of the hidden tap vs its float32 variant: 723/728 and 4479/4497 identical.
- Memory: the sink keeps H only where a note's first frame can fall and prefix sums where its first or last frame can
  (proof from decode.py in `hidden_store.hpp`; 0 misses): 130 MB peak on the 4-min song vs 1.55 GB (f32) / 777 MB (f16)
  for the whole-song tensor Python keeps.
- Vulkan f16, full chain, 4-min song: 3624 notes exact, 765 within 1 ms, 106 within 30 ms, 2 unmatched.

## Grouper (`dettato_parity_grouper`)

- CPU f32: tuning within 1e-7 cents; window tensors 1e-8 – 2e-6; pass-2 gathered features with the reference's float16
  accumulation emulated exactly; decode (scipy average linkage + fcluster, Hungarian, argsort, erf) exact on all
  fixtures and on 3600 / 3000 / 927 / 120010 random cases.
- End to end, bit-identical tracks to the reference *with an exact PCEN*. Against the package itself: ARI 0.976 (30 s
  mix), 1.000 (a 30-cent-sharp copy that retunes), 0.988 (4-min song), same instruments. The whole gap is the
  package's PCEN, which convolves with a float32 FFT whose round-off swamps the 200-tap convolution next to silence
  (4.7e-2 rel L2 on a padded window) and depends on the FFT backend (running only the PCEN on CUDA's FFT moves Python's
  own mix to ARI 0.91). The C++ computes the convolution exactly.
- Vulkan f16 end to end: ARI 0.991 / 1.000 / 0.997 vs the exact-PCEN reference.
- Resampler (our soxr stand-in, 0.14 % rel L2 on a retune): identical track assignment on the retuned fixture.

## Drums (`dettato_parity_drums`)

- CPU f32, five fixtures (30 s, 87 s, 4 min, a 30 s pop clip that routes to drum-kit, 0.26 s): features 0.9 – 2.7e-7
  (Python's 16 kHz injected), ~70 tapped tensors <= 5.8e-6, router identical, decode exact, hits identical except one
  velocity per long fixture that rounds the other way at the 3rd decimal (float16 features; the reference value is
  within 1.4e-6 of the rounding boundary).
- Vulkan f16: worst tensor 9.3e-3 (an expert's LayerNorm after two 12,000-step GRU layers); hits 99.4 – 100 %.
- With our resampler instead of soxr: 98.3 – 100 % of hits identical.

## End to end (`py/bench/compare_e2e.py`, RTX 5060 with Vulkan and f16 weights vs the package on CUDA)

The pipeline differs from the package's defaults in two ways: one transcriber pass on the retuned audio, and 8 s windows
every 4 s instead of every 2 s. So it is compared with the package twice, as published and with its first pass at the
same 4 s step, and the package's two settings are compared with each other to show how much that step alone moves.
Both recordings are in tune, so the single pass changes nothing here.

4-min song (nine instruments and drums):

| | pitched notes | onset F1 | drum hits | drum F1 | track ARI |
|---|---|---|---|---|---|
| dettato.cpp vs the package at the same 4 s step | 3830 vs 3828 | 0.996 | 971 vs 976 | 0.996 | 0.937 |
| dettato.cpp vs the package as published | 3830 vs 3846 | 0.982 | 971 vs 976 | 0.996 | 0.911 |
| the package at 4 s vs the package as published | 3828 vs 3846 | 0.982 | 976 vs 976 | 1.000 | 0.915 |

30 s of the same song:

| | pitched notes | onset F1 | drum hits | drum F1 | track ARI |
|---|---|---|---|---|---|
| dettato.cpp vs the package at the same 4 s step | 566 vs 566 | 0.995 | 179 vs 177 | 0.994 | 0.969 |
| dettato.cpp vs the package as published | 566 vs 571 | 0.982 | 179 vs 177 | 0.994 | 0.867 |
| the package at 4 s vs the package as published | 566 vs 571 | 0.983 | 177 vs 177 | 1.000 | 0.893 |

Onset F1: same pitch, onsets within 50 ms. Track ARI: the adjusted Rand index of the track assignment over the notes
both runs found.

The notes and drum hits follow the package closely. The tracks are the unstable part, in the package as much as here:
the grouper's decode is a clustering, and a few notes moving changes which clusters merge. On the 4-min song all three
runs write piano, trumpet, trombone, cello, harp, violin, oboe and drums, and split the string and synth notes
differently: dettato.cpp into synth lead (196 notes) and string ensemble (201), the package at 4 s into string ensemble
(335) and synth strings (206), the package as published into synth strings (261) with more of them in the trombone track
(697 notes against 501 and 393). Against the reference instruments of the benchmark recordings the three score the same
(`docs/BENCHMARK.md`).
