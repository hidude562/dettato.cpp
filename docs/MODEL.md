# The model and the GGUF file

## The four networks

dettato is four networks in a chain. dettato.cpp runs them in this order:

| Stage | Network (parameters) | Reads | Writes |
|---|---|---|---|
| tuning check | none (harmonic salience on a cents grid) | the 16 kHz audio | the offset from A440; audio clearly off it is resampled onto it |
| transcriber | LaneNet (15.3M) | a harmonic CQT of 8 s windows (88 piano-key lanes) | onset, frame and offset maps (two voices per key) and hidden states |
| evidence repairer | a transformer over candidate notes (4.9M) | the maps and the transcriber's hidden states around each candidate | keep / drop / insert decisions and corrected times |
| grouper | NoteNet, attention over the notes of 24 s windows (2.9M) | each note's maps and the transcriber's hidden and stem states, a PCEN spectrogram | per-note probabilities over 73 instruments and an embedding; a clustering decode turns them into named tracks |
| drums | a router, a drum-kit model and orchestral experts (5.0M) | the 44.1 kHz audio's 7.8-20 kHz band | drum hits with General MIDI keys and velocities |

The drums only need the audio, so on a GPU they run alongside the repairer and the grouper on their own backend
instance. The Python package runs the transcriber twice (once on the recording, once on the retuned recording for the
grouper); dettato.cpp checks the tuning first and runs it once on the retuned audio for both (`docs/BENCHMARK.md`).

| Part | Where it runs |
|---|---|
| STFTs and filterbanks (harmonic CQT, mel, PCEN), resampling | C++ on the CPU (pocketfft) |
| the four networks | ggml graphs on the chosen device |
| peak-pick decode, evidence features, clustering decode (average linkage, Hungarian matching) | C++ on the CPU |
| the drum model's song-length GRU recurrences | C++ on the CPU; their input projections in ggml |

## The GGUF

One file holds all four networks: `dettato-f16.gguf` (74 MB, the matrices in float16) or `dettato-f32.gguf` (130 MB).
Published at [huggingface.co/kobimusic/dettato-gguf](https://huggingface.co/kobimusic/dettato-gguf); made by
`py/convert.py` from the four safetensors repos (kobimusic/dettato, -evidence, -grouper, -drums) at the revisions pinned
in `REVISIONS`, which the file records.

Tensor names start with the stage: `tr.` (169 tensors), `ev.` (58), `gr.` (151), `dr.` (556). Vectors, norms, taps,
filterbanks and masks are always float32; `--dtype` only changes the large matrices and conv kernels.

Metadata:

| Key | |
|---|---|
| `dettato.format_version` | 1. `load()` refuses other versions with `Error::UnsupportedCheckpointVersion` |
| `dettato.weight_dtype` | `f16` or `f32` |
| `dettato.<stage>.source` | `kobimusic/<repo>@<revision>` the weights came from |
| `dettato.<stage>.config` | the stage's `config.json`, verbatim |
| `dettato.reference_commit` | the kobimusic/dettato commit whose code built the run-time buffers |
| `tr.*`, `ev.*`, `gr.*`, `dr.*` | each stage's hyperparameters and thresholds, read by the C++ (no constant is hard-coded twice) |

What the conversion changes, all exact algebra (the floats move by rounding only):

- BatchNorm in eval mode becomes a per-channel affine and is folded into the conv or 1x1 conv next to it where padding
  allows.
- LayerNorm affines are folded into the linear layer after them, so the graph uses a bare normalisation.
- Layer-scale gammas are folded into the linear layer before them.
- The transcriber's depthwise-conv biases become a per-lane bias of the next 1x1 conv.
- Attention heads of size 46 are zero-padded to 48 (flash attention kernels need a multiple of 8).
- Buffers the Python code builds at init time (filterbanks, windows, relative-position masks, the PitchMix shift matrix)
  are stored, so the C++ never recomputes them.
- Bidirectional GRU weights are laid out for one batched matmul per step over both directions, with `b_hr` / `b_hz`
  pre-added.

## Licence

The weights and the code are Apache-2.0 (KobiMusic). The training data notes are on each stage's model card.
