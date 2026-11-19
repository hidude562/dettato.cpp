---
license: apache-2.0
library_name: ggml
tags:
- music
- audio
- music-transcription
- automatic-music-transcription
- audio-to-midi
- multi-instrument
- gguf
base_model:
- kobimusic/dettato
- kobimusic/dettato-evidence
- kobimusic/dettato-grouper
- kobimusic/dettato-drums
---

# dettato-gguf

<p>
  <a href="https://huggingface.co/collections/kobimusic/dettato"><img alt="Collection: dettato" src="https://img.shields.io/badge/collection-dettato-FFD21E?logo=huggingface&logoColor=000"></a>
  <a href="https://github.com/hidude562/dettato.cpp"><img alt="Code: hidude562/dettato.cpp" src="https://img.shields.io/badge/code-dettato.cpp-181717?logo=github"></a>
  <a href="https://kobi.music"><img alt="kobi.music" src="https://img.shields.io/badge/web-kobi.music-B73E6A"></a>
</p>

**dettato-gguf** is all four models of dettato in one GGUF file, for [dettato.cpp](https://github.com/hidude562/dettato.cpp).
dettato.cpp runs them with ggml on the CPU, Metal or Vulkan, with no Python. These are the same weights as the four
safetensors repos. I converted them with `py/convert.py` from dettato.cpp.

| file | size | |
|---|---|---|
| `dettato-f16.gguf` | 73.7 MB | the large matrices in float16. This is the one I would use |
| `dettato-f32.gguf` | 130.0 MB | everything in float32, for the parity tests |

`SHA256SUMS` has the checksums. Each file also records the revisions it was converted from:

| model | repo | revision |
|---|---|---|
| transcriber (15.3M) | [kobimusic/dettato](https://huggingface.co/kobimusic/dettato) | `a0d1e354` |
| evidence repairer (4.9M) | [kobimusic/dettato-evidence](https://huggingface.co/kobimusic/dettato-evidence) | `5633f28e` |
| grouper (2.9M) | [kobimusic/dettato-grouper](https://huggingface.co/kobimusic/dettato-grouper) | `5a94cea2` |
| drums (5.0M) | [kobimusic/dettato-drums](https://huggingface.co/kobimusic/dettato-drums) | `97801f23` |

The conversion folds the BatchNorm, LayerNorm and layer scale weights into the layers next to them. It also pads the
transcriber's attention heads from 46 to 48 and stores the filterbanks, so the C++ does not have to build them. The
numbers only change by rounding. The [model docs](https://github.com/hidude562/dettato.cpp/blob/main/docs/MODEL.md) have
the details.

## Benchmark

On four of the dettato benchmark sets (Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic and RWC Royalty-Free), dettato.cpp
scores 67.75 composite F1 and the Python package scores 67.51. A 4 minute song takes 11.7 seconds on an RTX 5060 in Fast
mode, and the Python package takes 20.1 seconds on the same card. The
[benchmark docs](https://github.com/hidude562/dettato.cpp/blob/main/docs/BENCHMARK.md) have the full tables.

## Use

```bash
git clone https://github.com/hidude562/dettato.cpp && cd dettato.cpp
python py/fetch.py models/
cd cpp && cmake --preset release && cmake --build --preset release
../build/tools/dettato_cli ../models/dettato-f16.gguf song.wav -o out/ --fast
```

The C++ API is in the [README](https://github.com/hidude562/dettato.cpp). For Python, use the
[dettato package](https://github.com/kobimusic/dettato) and the safetensors repos instead.

## What to Expect

The models are the same as in the package, so the same notes apply. The weakest step is putting the notes into the
right instrument tracks, and the notes are performance MIDI that is not quantized to a beat. The drums only run on audio
recorded at 40 kHz or more. The training data and the details of each model are on its own card.

## License

Apache-2.0, the same as the dettato weights and code.
