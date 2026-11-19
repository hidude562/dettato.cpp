"""Reference tensors from the PyTorch implementation, for the C++ parity tests.

    python py/dump_refs.py OUT_DIR [--audio WAV] [--seconds 30]

Runs on the CPU in float32 (deterministic, and what the parity tests compare against). Writes .npy files:

    OUT_DIR/<fixture>/audio16k.npy                 the 16 kHz input
    OUT_DIR/<fixture>/win/*.npy                    one 8 s window through LaneNet, layer by layer (batch of 1)
    OUT_DIR/<fixture>/run/*.npy                    Transcriber.run over the fixture (hop_frac 0.25): maps and notes

Fixtures: "tone" (two sine notes, 4 s) and "mix" (the first --seconds of --audio, resampled with soxr HQ as the package does).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dettato_ref  # noqa: E402,F401  (puts the reference package on sys.path)
from dettato.transcriber import Transcriber  # noqa: E402
from dettato.transcriber.core import HEADS  # noqa: E402

torch.set_num_threads(int(os.environ.get("DUMP_THREADS", "16")))


def tone(seconds=4.0, rate=16000, freqs=(220.0, 330.0), start=0.5):
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for f in freqs:
        x += 0.2 * np.sin(2 * np.pi * f * t) * (t >= start)
    return x.astype(np.float32)


def save(d: Path, name: str, a):
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    np.save(d / f"{name}.npy", np.ascontiguousarray(a))


@torch.no_grad()
def dump_window(tr: Transcriber, audio: np.ndarray, d: Path, win_index: int):
    """One window as Transcriber.windows cuts it, through every stage of LaneNet."""
    from dettato.transcriber.core import SR, WIN_S

    W = int(round(WIN_S * SR))
    _, _, _, starts, pad, _ = tr._layout(len(audio), 0.25)
    x_all = np.concatenate([np.zeros(pad, np.float32), audio, np.zeros(pad + W, np.float32)])
    s0 = starts[min(win_index, len(starts) - 1)]
    x = torch.from_numpy(x_all[s0:s0 + W][None].copy())
    save(d, "x", x)
    net = tr.model
    outs = {}
    hooks = []
    names = {id(m): n for n, m in net.named_modules()}

    def hook(mod, inp, out):
        outs[names[id(mod)]] = out

    for n, m in net.named_modules():
        if n in ("front", "stem", "down", "up_norm", "head") or (n.startswith("layers.") and n.count(".") == 1) or \
                (n.startswith("refine.") and n.count(".") == 1):
            hooks.append(m.register_forward_hook(hook))
    taps = {}
    res = net(x, taps)
    for h in hooks:
        h.remove()
    save(d, "hcqt", outs["front"])                                 # (1, 9, 800, 288)
    save(d, "stem", outs["stem"])                                  # (1, C, 400, 96)
    save(d, "down", outs["down"])                                  # (1, C, 200, 88)
    keep = {"layers.0", "layers.5", "layers.6", "layers.7", f"layers.{len(net.layers) - 1}", f"refine.{len(net.refine) - 1}"}
    for n, v in outs.items():
        if n in keep:                                              # first block, first attention pair + GRU, trunk end
            save(d, n.replace(".", "_"), v)
    save(d, "up_norm", outs["up_norm"])
    save(d, "hidden", taps["hidden"])                              # (1, C, 400, 88)
    for k in HEADS:
        save(d, f"logit_{k}", res[k])                              # (1, 400, 88)


@torch.no_grad()
def dump_run(tr: Transcriber, audio: np.ndarray, d: Path):
    maps, buf = tr.run(audio, hop_frac=0.25, batch=2, taps=("hidden",))
    for k, v in maps.items():
        save(d, f"map_{k}", v)                                     # (T, 88) f32
    save(d, "hidden", buf["hidden"].float())                       # (T, 88, C), f16-accumulated as in the package
    notes = tr.notes(maps)
    for k, v in notes.items():
        save(d, f"notes_{k}", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--audio")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=2, help="which 8 s window of the mix to dump layer by layer")
    a = ap.parse_args()
    out = Path(a.out)
    tr = Transcriber.from_pretrained("cpu")
    fixtures = {"tone": tone()}
    if a.audio:
        import soundfile as sf
        import soxr

        x, sr = sf.read(a.audio, dtype="float32", always_2d=True)
        x = x.mean(1)[: int(a.seconds * sr)].astype(np.float32)
        fixtures["mix"] = soxr.resample(x, sr, 16000, quality="HQ").astype(np.float32)
    for name, audio in fixtures.items():
        d = out / name
        save(d, "audio16k", audio)
        dump_window(tr, audio, d / "win", a.window)
        dump_run(tr, audio, d / "run")
        print(f"{name}: {len(audio) / 16000:.1f} s -> {d}")
    (out / "REF_COMMIT").write_text(dettato_ref.ref_commit() + "\n")


if __name__ == "__main__":
    main()
