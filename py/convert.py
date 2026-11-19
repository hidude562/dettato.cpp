"""Convert the published dettato weights (safetensors + config.json, one Hub repo per stage) into one GGUF file.

    python py/convert.py out.gguf [--dtype f16|f32] [--weights DIR] [--stages transcriber,evidence,grouper,drums]

The weights are the four Hub repos at the revisions pinned in REVISIONS (downloaded to the Hub cache), or DIR holding them
(dettato/, dettato-evidence/, dettato-grouper/, dettato-drums/), as for DETTATO_WEIGHTS. The run-time buffers come from the
reference package (py/dettato_ref.py, $DETTATO_REF). The sources are recorded in the file (dettato.<stage>.source,
dettato.reference_commit). What the conversion changes, all exact:

- BatchNorm in eval mode becomes a per-channel affine, and is folded into the conv or 1x1 conv next to it where padding
  allows (a BN after a conv folds into it; a BN before a 1x1 conv folds into it).
- Buffers the Python code builds at init time (filterbanks, windows, relative-position masks) are stored, so the C++ side
  never recomputes them.
- Big matrices are stored at --dtype; vectors, norms, taps and masks stay f32.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
REPOS = {"transcriber": "dettato", "evidence": "dettato-evidence", "grouper": "dettato-grouper", "drums": "dettato-drums"}
# The Hub revisions the published GGUF (kobimusic/dettato-gguf) was converted from, and the parity checks ran against.
REVISIONS = {
    "transcriber": "a0d1e354d282c4ed458b83b287065587979b053e",
    "evidence": "5633f28ee330a48bc31bdd3f9f98bb1a71206141",
    "grouper": "5a94cea2329f53228d7bcf1fd6a9f8d9778cc75b",
    "drums": "97801f2328e32e4c87b12586aa81121f62cacacb",
}


def repo_dir(stage: str, weights: str | None) -> Path:
    if weights:
        return Path(weights).expanduser() / REPOS[stage]
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(f"kobimusic/{REPOS[stage]}", revision=REVISIONS[stage]))


def load_stage(stage: str, weights: str | None):
    from safetensors.numpy import load_file

    d = repo_dir(stage, weights)
    cfg = json.loads((d / "config.json").read_text())
    st = {k: v.astype(np.float32) if v.dtype != np.int64 else v for k, v in load_file(str(d / "model.safetensors")).items()}
    return cfg, st


class Writer:
    def __init__(self, path: str, dtype: str):
        import gguf

        self.gguf = gguf
        self.w = gguf.GGUFWriter(path, "dettato")
        self.big = np.float16 if dtype == "f16" else np.float32
        self.names: set[str] = set()

    def t(self, name: str, a: np.ndarray, big: bool = False):
        """Store a tensor. numpy shape (..., b, a) becomes ggml ne [a, b, ...]."""
        assert name not in self.names, name
        self.names.add(name)
        a = np.ascontiguousarray(a, dtype=self.big if big else np.float32)
        self.w.add_tensor(name, a)

    def u32(self, k, v):
        self.w.add_uint32(k, int(v))

    def f32(self, k, v):
        self.w.add_float32(k, float(v))

    def s(self, k, v):
        self.w.add_string(k, str(v))

    def arr(self, k, v):
        self.w.add_array(k, list(v))

    def close(self):
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()


def bn_affine(st, p, eps=1e-5):
    s = st[p + ".weight"] / np.sqrt(st[p + ".running_var"] + eps)
    return s, st[p + ".bias"] - st[p + ".running_mean"] * s


def fold_bn_after(w, b, st, p):
    """conv (w, b) followed by BatchNorm p -> one conv."""
    s, sh = bn_affine(st, p)
    return w * s.reshape((-1,) + (1,) * (w.ndim - 1)), b * s + sh


# ----------------------------------------------------------------------------- shared
def gru(W, st, gp: str, dst: str, layer: int = 0, ln: str | None = None):
    """One bidirectional nn.GRU layer, laid out for a batched step over both directions (direction = outermost dim).

    PyTorch: r = s(W_ir x + b_ir + W_hr h + b_hr), z = s(W_iz x + b_iz + W_hz h + b_hz),
             n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), h' = (1 - z) n + z h.
    Stored: ih.w (2, 3H, In), ih.b (2, 3H) with b_hr, b_hz already added (they only ever add to the input side),
            hh.w (2, 3H, H), hh.bn (2, H) = b_hn (it sits inside r * (...))."""
    sfx = [f"_l{layer}", f"_l{layer}_reverse"]
    wi = np.stack([st[gp + "weight_ih" + s] for s in sfx])
    wh = np.stack([st[gp + "weight_hh" + s] for s in sfx])
    bi = np.stack([st[gp + "bias_ih" + s] for s in sfx]).copy()
    bh = np.stack([st[gp + "bias_hh" + s] for s in sfx])
    h = wh.shape[2]
    if ln is not None:                                                                # a LayerNorm right before: fold it
        for d in range(2):
            wi[d], bi[d] = fold_ln(st, ln, wi[d], bi[d])
    bi[:, : 2 * h] += bh[:, : 2 * h]
    W.t(dst + ".ih.w", wi, big=True)                                                  # ne [In, 3H, 2]
    W.t(dst + ".ih.b", bi)                                                            # ne [3H, 2]
    W.t(dst + ".hh.w", wh, big=True)                                                  # ne [H, 3H, 2]
    W.t(dst + ".hh.bn", bh[:, 2 * h:])                                                # ne [H, 2]


# ----------------------------------------------------------------------------- transcriber
HEAD_PAD = 8  # attention head dims are padded to a multiple of this (46 -> 48): flash attention needs it on Vulkan


def pad_heads_qkv(w, b, heads, mult):
    """qkv (3 C, C_in) with heads of D = C / heads -> (3 heads Dp, C_in), Dp = D rounded up to mult, zero rows padded.
    Exact: zero query/key dims add nothing to q.k, zero value dims come out zero."""
    C = w.shape[0] // 3
    D = C // heads
    Dp = -(-D // mult) * mult
    wp = np.zeros((3, heads, Dp, w.shape[1]), np.float32)
    bp = np.zeros((3, heads, Dp), np.float32)
    wp[:, :, :D] = w.reshape(3, heads, D, -1)
    bp[:, :, :D] = b.reshape(3, heads, D)
    return wp.reshape(3 * heads * Dp, -1), bp.reshape(-1)


def pad_heads_out(w, heads, mult):
    """out (C_out, C) -> (C_out, heads Dp): zero columns for the padded value dims."""
    C = w.shape[1]
    D = C // heads
    Dp = -(-D // mult) * mult
    wp = np.zeros((w.shape[0], heads, Dp), np.float32)
    wp[:, :, :D] = w.reshape(w.shape[0], heads, D)
    return wp.reshape(w.shape[0], heads * Dp)


def pmix_kernel(taps: np.ndarray, offsets) -> np.ndarray:
    """PitchMix taps (C, K) at semitone offsets -> one depthwise conv kernel along the lanes, (C, 1, 1, 2R+1), with the
    block's residual (y + pmix(y)) folded in as a 1 at the centre: y_out[i] = sum_m k[m] y[i + m - R]."""
    r = max(abs(o) for o in offsets)
    k = np.zeros((taps.shape[0], 2 * r + 1), np.float32)
    for j, o in enumerate(offsets):
        k[:, o + r] = taps[:, j]
    assert 0 not in offsets
    k[:, r] = 1.0
    return k.reshape(taps.shape[0], 1, 1, 2 * r + 1)


def dw_kernel(w: np.ndarray) -> np.ndarray:
    """Depthwise conv weight, torch (C, 1, KH, KW) -> (KH, KW, 1, C), ggml ne [C, 1, KW, KH]: channels fastest, the layout
    ggml's channels-last (CWHN) depthwise conv reads. The graph permutes it to ne [KW, KH, 1, C]."""
    return np.ascontiguousarray(w[:, 0].transpose(1, 2, 0)[:, :, None, :])


def lane_block(W, st, src, dst, offsets, n_lanes):
    """LaneBlock: dw_t -> + PitchMix -> BN -> pw1 -> GELU -> pw2 -> * gamma, residual. Folded (exact algebra):
    - the residual of PitchMix as a centre tap of 1 (offset 0, the last tap);
    - BN into pw1; gamma into pw2;
    - dw_t's bias into pw1's bias: PitchMix of a per-channel constant is a per-(channel, lane) constant (the edge lanes
      see the zero padding), so pw1's bias becomes per lane, ne [736, n_lanes].
    PitchMix runs as a matmul with each channel's banded lane matrix, built in the graph from the taps and tr.pmix_S."""
    taps = np.concatenate([st[src + ".pmix.taps"], np.ones((st[src + ".pmix.taps"].shape[0], 1), np.float32)], 1)
    offs = list(offsets) + [0]
    W.t(dst + ".dw_w", st[src + ".dw_t.weight"])                                      # (C, 1, kt, 1) -> ne [1, kt, 1, C]
    W.t(dst + ".pmix", taps)                                                          # (C, K + 1) -> ne [K + 1, C]
    lanes = np.arange(n_lanes)
    reach = np.zeros((taps.shape[0], n_lanes), np.float64)                           # sum of the taps that land inside
    for k, o in enumerate(offs):
        inside = (lanes + o >= 0) & (lanes + o < n_lanes)
        reach[:, inside] += taps[:, k:k + 1]
    bz = st[src + ".dw_t.bias"][:, None].astype(np.float64) * reach                  # (C, lanes)
    s, sh = bn_affine(st, src + ".norm")
    w1 = st[src + ".pw1.weight"].reshape(st[src + ".pw1.weight"].shape[0], -1).astype(np.float64)
    w1s = w1 * s[None, :]
    b1 = st[src + ".pw1.bias"] + w1 @ sh                                              # (736,)
    W.t(dst + ".pw1.w", w1s, big=True)
    W.t(dst + ".pw1.b", (b1[:, None] + w1s @ bz).T)                                   # (lanes, 736) -> ne [736, lanes]
    g = st[src + ".gamma"]
    w2 = st[src + ".pw2.weight"].reshape(st[src + ".pw2.weight"].shape[0], -1)
    W.t(dst + ".pw2.w", w2 * g[:, None], big=True)
    W.t(dst + ".pw2.b", st[src + ".pw2.bias"] * g)


def pmix_shift(offsets, n_lanes) -> np.ndarray:
    """(P*P, K+1): row i * P + j is 1 in column k where lane j = lane i + offset k (offset 0 last). Times a channel's taps
    it gives that channel's PitchMix matrix W[i, j], laid out j fastest."""
    offs = list(offsets) + [0]
    S = np.zeros((n_lanes, n_lanes, len(offs)), np.float32)
    for k, o in enumerate(offs):
        i = np.arange(n_lanes)
        ok = (i + o >= 0) & (i + o < n_lanes)
        S[i[ok], i[ok] + o, k] = 1.0
    return S.reshape(n_lanes * n_lanes, len(offs))


def fold_ln(st, ln: str, w: np.ndarray, b: np.ndarray):
    """LayerNorm(x) then a linear map (w, b): the affine part of the norm goes into the map."""
    return w * st[ln + ".weight"][None, :], b + w @ st[ln + ".bias"]


def transcriber(W: Writer, cfg: dict, st: dict, prefix="tr"):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dettato_ref import transcriber_module  # the reference package, for the init-time buffers

    m = cfg["model"]
    net = transcriber_module().LaneNet(**m)
    fr = net.front
    c, heads = m["c"], m["attn_heads"]
    p = prefix
    W.u32(f"{p}.c", c)
    W.u32(f"{p}.depth", m["depth"])
    W.u32(f"{p}.kt", m["kt"])
    W.arr(f"{p}.dilations", m["dilations"])
    W.arr(f"{p}.attn_at", m["attn_at"])
    W.arr(f"{p}.gru_at", m["gru_at"])
    W.u32(f"{p}.attn_heads", heads)
    W.u32(f"{p}.gru_hidden", m["gru_hidden"])
    W.u32(f"{p}.refine", m["refine"])
    W.u32(f"{p}.n_lanes", m["n_lanes"])
    W.arr(f"{p}.pmix", m["pmix"])
    W.arr(f"{p}.heads", m["heads"])
    W.u32(f"{p}.n_fft", fr.n_fft)
    W.u32(f"{p}.n_bins", fr.n_bins)
    W.arr(f"{p}.shifts", fr.shifts)
    W.u32(f"{p}.shift_lo", -fr.lo)                                                  # lo <= 0
    W.arr(f"{p}.extra_nffts", fr.extra)
    W.f32(f"{p}.log_eps", fr.log_eps)
    for k, v in cfg["thresholds"].items():
        W.f32(f"{p}.th.{k}", v)
    # front end buffers
    W.t(f"{p}.hcqt.fb", fr.fb.numpy())                                               # (NE, 1025) -> ne [1025, NE]
    W.t(f"{p}.hcqt.win", fr.win.numpy())
    for i, _ in enumerate(fr.extra):
        W.t(f"{p}.hcqt.fb_x{i}", getattr(fr, f"fb_x{i}").numpy())
        W.t(f"{p}.hcqt.win_x{i}", getattr(fr, f"win_x{i}").numpy())
    W.t(f"{p}.pmix_S", pmix_shift(m["pmix"], m["n_lanes"]))                           # ne [K + 1, P * P]
    # stem
    s, sh = bn_affine(st, "stem.inorm")
    W.t(f"{p}.stem.in_scale", s)
    W.t(f"{p}.stem.in_shift", sh)
    for src, bn, dst in (("stem.c1", "stem.n1", "c1"), ("stem.c2", "stem.n2", "c2"), ("stem.down", "stem.n3", "down")):
        w, b = fold_bn_after(st[src + ".weight"], st[src + ".bias"], st, bn)
        W.t(f"{p}.stem.{dst}.w", w, big=True)                                        # (OC, IC, KT, KF) -> ne [KF, KT, IC, OC]
        W.t(f"{p}.stem.{dst}.b", b)
    # trunk: the torch ModuleList interleaves blocks, attention and GRU layers
    li = 0
    for i in range(m["depth"]):
        lane_block(W, st, f"layers.{li}", f"{p}.blk.{i}", m["pmix"], m["n_lanes"])
        li += 1
        g = m["attn_at"].index(i + 1) if (i + 1) in m["attn_at"] else None
        if g is not None:
            for ax in m["attn_axes"]:
                src, dst = f"layers.{li}", f"{p}.attn.{g}.{ax}"
                qw, qb = fold_ln(st, src + ".ln", st[src + ".qkv.weight"], st[src + ".qkv.bias"])
                qw, qb = pad_heads_qkv(qw, qb, heads, HEAD_PAD)
                W.t(dst + ".qkv.w", qw, big=True)                                    # (3 H Dp, C)
                W.t(dst + ".qkv.b", qb)
                gm = st[src + ".gamma"]
                W.t(dst + ".out.w", pad_heads_out(st[src + ".out.weight"] * gm[:, None], heads, HEAD_PAD), big=True)
                W.t(dst + ".out.b", st[src + ".out.bias"] * gm)
                if ax == "pitch":
                    rb = st[src + ".rel_bias"]                                       # (H, 2 n_pos - 1)
                    n_pos = (rb.shape[1] + 1) // 2
                    L = m["n_lanes"]
                    idx = np.arange(L)
                    rel = idx[None, :] - idx[:, None] + n_pos - 1                     # [query i, key j] -> j - i + n_pos - 1
                    W.t(dst + ".mask", rb[:, rel])                                   # (H, Lq, Lk) -> ne [Lk, Lq, H]
                li += 1
        if (i + 1) in m["gru_at"]:
            g = m["gru_at"].index(i + 1)
            src, dst = f"layers.{li}", f"{p}.gru.{g}"
            gru(W, st, src + ".gru.", dst, ln=src + ".ln")
            ow = st[src + ".out.weight"]
            hdim = ow.shape[1] // 2
            W.t(dst + ".out.wf", ow[:, :hdim], big=True)                                 # forward half of the projection
            W.t(dst + ".out.wr", ow[:, hdim:], big=True)                                 # reverse half
            W.t(dst + ".out.b", st[src + ".out.bias"])
            li += 1
    assert f"layers.{li}.gamma" not in st and f"layers.{li}.ln.weight" not in st, "unconsumed trunk layers"
    W.t(f"{p}.down.w", dw_kernel(st["down.weight"]))                                 # ne [C, 1, 1, 3]
    W.t(f"{p}.down.b", st["down.bias"])
    s, sh = bn_affine(st, "up_norm")
    W.t(f"{p}.up.scale", s)
    W.t(f"{p}.up.shift", sh)
    for i in range(m["refine"]):
        lane_block(W, st, f"refine.{i}", f"{p}.ref.{i}", m["pmix"], m["n_lanes"])
    W.t(f"{p}.head.w", st["head.weight"].reshape(len(m["heads"]), c), big=True)
    W.t(f"{p}.head.b", st["head.bias"])
    used = {k for k in st if not k.endswith("num_batches_tracked")}
    return used


def stage_converter(stage: str):
    """The transcriber lives here; every other stage in py/convert_<stage>.py, as convert(W, cfg, st)."""
    if stage == "transcriber":
        return transcriber
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module(f"convert_{stage}").convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--weights", default=os.environ.get("DETTATO_WEIGHTS"))
    ap.add_argument("--stages", default="transcriber,evidence,grouper,drums")
    a = ap.parse_args()
    W = Writer(a.out, a.dtype)
    W.u32("dettato.format_version", FORMAT_VERSION)
    W.s("dettato.weight_dtype", a.dtype)
    W.s("general.name", "dettato")
    W.s("general.license", "apache-2.0")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dettato_ref

    W.s("dettato.reference_commit", dettato_ref.ref_commit())
    for stage in a.stages.split(","):
        cfg, st = load_stage(stage, a.weights)
        W.s(f"dettato.{stage}.source", f"local:{REPOS[stage]}" if a.weights else f"kobimusic/{REPOS[stage]}@{REVISIONS[stage]}")
        W.s(f"dettato.{stage}.config", json.dumps(cfg))
        stage_converter(stage)(W, cfg, st)
    W.close()
    print(f"wrote {a.out}: {len(W.names)} tensors")


if __name__ == "__main__":
    main()
