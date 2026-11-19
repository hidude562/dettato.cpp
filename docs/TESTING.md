# Testing

## The test suite

```bash
cd cpp && cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu
```

`dettato_tests` (Catch2) has two kinds of tests:

- `[pure]`: no weights. The resampler against closed-form tones (passband error, stopband rejection, lengths), the STFT
  against a direct DFT, the peak-pick decode on hand-made maps, the MIDI writer (read back with a parser), the WAV reader
  (16-bit, 24-bit, float, stereo downmix) and the API's error paths.
- `[model]`: a dettato GGUF on the CPU. A synthetic tone sequence (C4, E4, G4, C5 plucked tones) must come out as those
  notes at 44.1 kHz and at 16 kHz; the progress must be monotonic; argument checks and
  cancellation must work. About a minute. They run when `DETTATO_TEST_MODEL` names the GGUF and are skipped otherwise:

```bash
python py/fetch.py models/                        # downloads dettato-f16.gguf and checks its sha256
DETTATO_TEST_MODEL=models/dettato-f16.gguf ctest --preset cpu
```

CI runs the `[pure]` tests on Linux (GCC, Clang), macOS (arm64, Metal build) and Windows (MSVC), and the `[model]` tests on
Linux.

## Parity with the PyTorch package

The networks were ported stage by stage against the Python package, and the tools that did it are here. Results:
`docs/PARITY.md`.

1. Check out the reference package at the commit the GGUF records (`dettato.reference_commit`):
   `git clone https://github.com/kobimusic/dettato ~/dettato-ref && git -C ~/dettato-ref checkout c4c52e0`.
   `py/dettato_ref.py` puts it on `sys.path` (set `DETTATO_REF` for another location). Python needs `torch numpy scipy
   soxr soundfile safetensors huggingface_hub gguf`.
2. Convert the weights in float32: `python py/convert.py models/dettato-f32.gguf --dtype f32`.
3. Dump the reference tensors on the CPU in float32. The fixtures are cut from recordings you provide (the ones behind
   `docs/PARITY.md` are not redistributable):
   ```bash
   python py/dump_refs.py testdata/refs --audio MIX.wav                         # 30 s mix + a synthetic tone
   python py/dump_refs_evidence.py testdata/refs_evidence --fixtures mix
   python py/dump_refs_grouper.py testdata/refs_grouper --fixtures mix,detuned
   python py/dump_refs_drums.py testdata/refs_drums --audio SONG44K.wav [--pop POP44K.wav]
   ```
   `DUMP_THREADS` sets torch's threads (default 16).
4. Compare stage by stage:
   ```bash
   build/tools/dettato_parity          models/dettato-f32.gguf testdata/refs/mix --device cpu
   build/tools/dettato_parity_evidence models/dettato-f32.gguf testdata/refs_evidence/mix --device cpu
   build/tools/dettato_parity_grouper  models/dettato-f32.gguf testdata/refs_grouper/mix --device cpu
   build/tools/dettato_parity_drums    models/dettato-f32.gguf testdata/refs_drums/mix30 --device cpu
   ```
   Each prints the relative L2 error of every intermediate tensor and the agreement of the discrete outputs, and ends with
   `all checks passed` or the number of failed checks. With `--device gpu` and the f16 GGUF they report the GPU's
   differences instead (looser tolerances: Vulkan rounds f32 matmul inputs through f16).

The standard the port was held to: on the CPU with f32 weights, every tensor within 1e-4 relative L2 of PyTorch (most
are ~1e-6), and the discrete outputs (notes, hits, tracks, instrument names) the same, except where the package itself
rounds through float16, which is traced and bounded in `docs/PARITY.md`. No tolerance was widened to pass.

## End to end and the benchmark

- `py/bench/compare_e2e.py CPP.mid PY.mid`: note and drum-hit F1, the track lists and the adjusted Rand index of the track
  assignment between the C++ output and the package's for one recording.
- `py/bench/run_cpp_sets.sh` and `py/bench/score.py`: the benchmark sets (`docs/BENCHMARK.md`).
