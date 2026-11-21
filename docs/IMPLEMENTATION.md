# Implementation notes

How the networks are mapped onto ggml v1.0.0, for anyone changing a graph. The code is in `cpp/src/<stage>/`; shared
pieces are `backend.hpp` (device and thread setup), `model_file.hpp` (the GGUF), `graph.hpp` (`StaticGraph`, `linear`,
`readTensor`), `dsp/` (STFT, resampler) and `io/` (WAV, MIDI).

## Layout

- ggml shapes are fastest-first: a numpy array of shape (a, b, c) written by the GGUF writer is ne [c, b, a].
- Activations run channels-last, ne [C, lane, frame, window], so every Linear / 1x1 conv is one `ggml_mul_mat(w, x)`
  with the torch weight (out, in) stored as it is. Keep that matmul 2D (`ggml_reshape_2d(x, C, n)`); reshaped to 3D or 4D
  it becomes a batched matmul, about 40 % slower on Vulkan. Biases that vary per lane go on a 3D view of the 2D result.
- `linear()` in graph.hpp is mul_mat + `ggml_prec_set_acc(y, GGML_PREC_F32)` + the bias.

## Fewer elementwise passes

Every elementwise pass over a large f32 tensor costs about 1 ms per 100 MB on an RTX 5060, and ggml's Vulkan backend
does not fuse them, so constants are folded at conversion (`docs/MODEL.md`): BatchNorm, LayerNorm affines, layer
scales, the depthwise biases. LayerNorm is then a bare `ggml_norm`. GELU is `ggml_gelu_erf` (PyTorch's default is the
exact form, `ggml_gelu` is the tanh approximation).

## Convolutions

- `ggml_conv_2d_direct` (no im2col buffer). Torch data (B, C, T, F) is ggml [F, T, C, B] and a torch kernel
  (OC, IC, KT, KF) stored as it is is [KF, KT, IC, OC], so `s0` / `p0` / `d0` are the F parameters.
- Depthwise: `ggml_conv_2d_dw_direct`. On a channels-last input the kernel must be channels-fastest: stored
  ne [C, 1, KW, KH] and permuted `(3, 2, 0, 1)` (`dwKernel` in transcriber/lanenet.cpp).

## The transcriber's own blocks

- **PitchMix** (a per-channel convolution across the 88 lanes) is a batched matmul with each channel's banded lane
  matrix, built in the graph from its taps and the stored shift tensor `tr.pmix_S`.
- **Bidirectional GRU** over a window's frames: both directions in one batched matmul per step (the reverse direction
  reads a time-reversed copy via `ggml_get_rows`), the outputs written with a `ggml_set_rows` chain so each step depends
  on the last and a GPU cannot reorder them. About 20 graph nodes per step: fine for a 200-frame window. Song-length
  recurrences (the drum model's, thousands of steps) run on the host instead, with the input projections in ggml
  (`drums/gru_host.cpp`).
- **Attention**: q/k/v are views of one qkv matmul, rotary embeddings with `ggml_rope_ext` NEOX mode on a 3D view (some
  backends only rope 3D). On a GPU, `ggml_flash_attn_ext` with the mask cast to f16 in the graph (Vulkan needs an f16
  mask and a head size that is a multiple of 8, hence the padding to 48); otherwise `ggml_soft_max_ext` with an f32
  mask broadcast over the heads. The graph build checks every op against the backend and rebuilds with the explicit path
  if flash attention is missing.
- Rotary embeddings with continuous positions (the evidence repairer's onset times) cannot use `ggml_rope`, which takes
  integer positions: cos / sin are computed on the host and the rotation is views + mul + add.

## Streaming over a song

`Transcriber::run` slides 8 s windows over the whole recording (every 4 s), in batches: 2 windows on a GPU, 1 on the
CPU (every batch size gives the same notes; these are the smallest that lose no speed, and the host buffers grow with
the batch). Three
things overlap: the host computes the next batch's harmonic CQT while the device runs the current batch, and the host
accumulates the previous batch's maps (Hann-weighted overlap-add). Consumers read the network's hidden states
through `TapSink`s as the windows stream past, instead of a whole-song tensor:

- the evidence repairer's sink keeps the hidden states only near frames where a candidate note can start, and prefix
  sums where one can start or end (`evidence/hidden_store.hpp` has the argument from the decode rules): 130 MB on a
  4-min song instead of the 1.55 GB the package holds;
- the grouper's store sink averages, from the same windows, exactly the windows a hop-0.5 pass would run, with their own
  weight sums, so its features are bit-identical to the grouper running its own pass. For the few notes the store kept
  nothing for (notes the repairer inserted far from any candidate), it recomputes those windows.

## Host arithmetic that reproduces numpy / scipy / torch

The decode stages are discrete, so the host code reproduces the Python package's arithmetic, not only its maths: numpy's
pairwise float32 sums, scipy's average-linkage clustering and `fcluster`, the Hungarian matching of
`linear_sum_assignment`, numpy's sort orders for ties, Cephes' `erf`, SLEEF's `expf` as torch uses it. Floating-point
contraction is off for the library (explicit `std::fma` where the reference fuses). The grouper's PCEN is computed
exactly; the package computes it with an FFT convolution whose round-off is large next to silence, the one deliberate
difference (`docs/PARITY.md`).

## ggml versions

The library builds on ggml v1.0.0 and on the v0.23 fork muscriptor.cpp pins (DamRsn/ggml `b01bbb8`), so a host can link
both libraries on one ggml. Two differences matter:

- v1.0.0 sets a result's accumulation precision with `ggml_prec_set_acc`; before it, one function per op did.
  `accF32()` (`ggml_compat.hpp`) calls whichever the build's ggml has, picked at compile time.
- ggml-vulkan before v1.0.0 cannot broadcast a matmul's first operand over dim 3. The two PitchMix blocks (the
  transcriber's and the grouper's lane network) multiply each channel's lane matrix with a batch of windows in dim 3,
  so where `Backend::matmulBroadcastsDim3()` says no, they repeat the matrices per window first (`ggml_repeat_4d`):
  the same result, a little more memory traffic. No other graph broadcasts over dim 3.

## Lessons

- A failed `GGML_ASSERT` tries to attach a debugger and hides the message: set `GGML_NO_BACKTRACE=1`.
- Graph inputs must be set before every compute: the allocator may reuse their memory once the graph has consumed them.
- Views count as graph nodes: give `StaticGraph` headroom.
- Vulkan's matmul and conv round f32 inputs through f16 internally: expect ~1e-3 relative differences on a GPU even with
  f32 weights.
- A failed `ggml_gallocr_reserve` leaves the allocator half-assigned: start over with a fresh allocator.
- ggml's CPU threads slow down sharply when they share cores with other work; the drums therefore run after the grouper
  on a CPU, and in parallel only on a GPU, and the default thread count is the physical performance cores
  (`cpu_threads.cpp`).
- A network's compute buffer is host memory on the CPU: the pipeline drops the transcriber and the grouper's graph after
  their last use in a call and builds them again for the next (milliseconds).
- A GPU backend stages each tensor read in a host buffer as large as the read and keeps it: the taps (100 MB per window
  batch) are read back in 16 MB pieces.
- glibc keeps memory that short-lived worker threads freed in per-thread arenas, so a process grew with every call
  (0.8 -> 2.0 GB over 16 calls); `transcribe()` ends with `malloc_trim`.
- On the CPU, `ggml_backend_cpu_set_abort_callback` lets `should_cancel` stop a graph at its next node instead of
  after it (a transcriber window takes seconds there).
- ggml's device registry creates every backend it was built with. On Windows, hosts delay-load `vulkan-1.dll` so that
  they start on machines without Vulkan, and the first Vulkan call then faults when the loader is missing:
  `Backend::deviceCount()` looks for the loader before it touches the registry (the CPU backend is never taken from
  it), and keeps implicit Vulkan layers out of the instance ggml creates, as muscriptor.cpp does.
