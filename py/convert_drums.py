"""The drum stage (kobimusic/dettato-drums) for py/convert.py: ``convert(W, cfg, st)``, every tensor prefixed ``dr.``.

Models: ``dr.kit`` (the drum-kit DrumNet) and ``dr.ex{i}`` (orchestral expert i, its DrumNet under ``dr.ex{i}.base``).

What the conversion changes (exact algebra, float rounding only):
- BatchNorm (eval) folded into the conv before it: every stem ``cN`` conv, the depthwise and pointwise convs of each
  residual block, the conv block's depthwise conv.
- LayerNorm affines folded into the linear map that reads them: GRUBlock norm -> its GRU's input weights, SongAttention
  norm -> qkv.
- The velocity head's two linear maps (no activation between them) multiplied into one.
- BandGRU (a 20-step GRU across the bands of each frame; unrolled in the graph on a GPU, on the host on the CPU backend):
  both directions' input weights stacked into one matrix, b_hr and b_hz added to the input biases (as py/convert.py
  ``gru()``), b_hn kept apart.
- The stem's depthwise kernels are also stored channels-fastest (``dw.wcl``) for the CPU backend's channels-last blocks.

Song-length GRUs (GRUBlock, the family towers, the experts' 2-layer GRU) run their recurrence on the host: their input
weights (both directions stacked, ``ih.w`` (2 * 3H, In)) feed one matmul in the graph, the recurrent weights stay
float32 as ``hh.w`` (2, 3H, H) and ``hh.b`` (2, 3H) = the full b_hh, which the host adds as PyTorch does.

The feature front end's filterbanks and windows are stored as the Python code builds them (``dr.fb.*``, ``dr.win.*``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import bn_affine, dw_kernel, fold_bn_after, fold_ln  # noqa: E402

TB_DEFAULT_K, TB_DEFAULT_D = 15, 1
GRU_DEFAULT_H = 48
ATTN_DEFAULT_DA, ATTN_DEFAULT_HEADS = 32, 2


def _ref(name):
    import dettato_ref  # noqa: F401  (puts the reference package on sys.path)

    return importlib.import_module(name)


def _f64(st, ln: str) -> dict:
    """A LayerNorm's weight and bias in float64, for fold_ln."""
    return {ln + ".weight": st[ln + ".weight"].astype(np.float64), ln + ".bias": st[ln + ".bias"].astype(np.float64)}


class _Tracked(dict):
    """The state dict, remembering which entries were read."""

    def __init__(self, d):
        super().__init__(d)
        self.read: set[str] = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)


def _gru_host(W, st, gp: str, dst: str, layer: int = 0, ln: str | None = None):
    """One bidirectional nn.GRU layer for the host recurrence: ih.w (2*3H, In) [forward rows, then reverse], ih.b (2*3H)
    = b_ih (LayerNorm folded in when given), hh.w (2, 3H, H) f32, hh.b (2, 3H) = b_hh f32."""
    sfx = [f"_l{layer}", f"_l{layer}_reverse"]
    wi = [st[gp + "weight_ih" + s].astype(np.float64) for s in sfx]
    bi = [st[gp + "bias_ih" + s].astype(np.float64) for s in sfx]
    if ln is not None:
        for d in range(2):
            wi[d], bi[d] = fold_ln(_f64(st, ln), ln, wi[d], bi[d])
    W.t(dst + ".ih.w", np.concatenate(wi, 0), big=True)                       # ne [In, 6H]
    W.t(dst + ".ih.b", np.concatenate(bi, 0))                                 # ne [6H]
    W.t(dst + ".hh.w", np.stack([st[gp + "weight_hh" + s] for s in sfx]))     # ne [H, 3H, 2]
    W.t(dst + ".hh.b", np.stack([st[gp + "bias_hh" + s] for s in sfx]))       # ne [3H, 2]


def _gru_graph(W, st, gp: str, dst: str):
    """BandGRU's short GRU, unrolled in the graph: ih.w (2 * 3H, In) [forward rows, then reverse] and ih.b (2 * 3H) with
    b_hr, b_hz added (one matmul for both directions); per direction d in (f, r): hh{d}.w (3H, H), hh{d}.bn (H) = b_hn."""
    wi, bis = [], []
    for d, s in (("f", "_l0"), ("r", "_l0_reverse")):
        wh = st[gp + "weight_hh" + s]
        h = wh.shape[1]
        bi = st[gp + "bias_ih" + s].astype(np.float64).copy()
        bh = st[gp + "bias_hh" + s]
        bi[: 2 * h] += bh[: 2 * h]
        wi.append(st[gp + "weight_ih" + s])
        bis.append(bi)
        W.t(f"{dst}.hh{d}.w", wh, big=True)
        W.t(f"{dst}.hh{d}.bn", bh[2 * h:])
    W.t(f"{dst}.ih.w", np.concatenate(wi, 0), big=True)
    W.t(f"{dst}.ih.b", np.concatenate(bis, 0))


def _dw_cl(w: np.ndarray) -> np.ndarray:
    """Depthwise conv1d weight (C, 1, K) -> (1, K, 1, C), ggml ne [C, 1, K, 1]: channels fastest, for the channels-last
    (CWHN) depthwise conv; the graph permutes it to ne [K, 1, 1, C]."""
    return np.ascontiguousarray(w[:, 0, :].T.reshape(1, w.shape[2], 1, w.shape[0]))


def drumnet(W, st, src: str, dst: str, cfg: dict):
    """One DrumNet: state keys under src (e.g. "drum_kit."), tensors under dst (e.g. "dr.kit")."""
    # stem: spec tokens in order; torch indices: cN -> conv, BN, act; r -> Res2d; pN -> MaxPool
    ti = 0
    for j, tok in enumerate(t.strip() for t in cfg["stem"].split(",")):
        kind = tok[0]
        if kind == "c":
            w, b = fold_bn_after(st[f"{src}stem.net.{ti}.weight"], st[f"{src}stem.net.{ti}.bias"], st, f"{src}stem.net.{ti + 1}")
            W.t(f"{dst}.stem.{j}.w", w, big=True)                             # (OC, IC, 3, 3) -> ne [3, 3, IC, OC]
            W.t(f"{dst}.stem.{j}.b", b)
            ti += 3
        elif kind == "r":
            p = f"{src}stem.net.{ti}."
            w, b = fold_bn_after(st[p + "dw.weight"], st[p + "dw.bias"], st, p + "bn1")
            W.t(f"{dst}.stem.{j}.dw.w", w)                                    # (C, 1, 3, 3) -> ne [3, 3, 1, C]
            W.t(f"{dst}.stem.{j}.dw.wcl", dw_kernel(w))                       # ne [C, 1, 3, 3], for channels-last input
            W.t(f"{dst}.stem.{j}.dw.b", b)
            w, b = fold_bn_after(st[p + "pw1.weight"], st[p + "pw1.bias"], st, p + "bn2")
            W.t(f"{dst}.stem.{j}.pw.w", w.reshape(w.shape[0], w.shape[1]), big=True)   # (C, C) -> ne [C, C]
            W.t(f"{dst}.stem.{j}.pw.b", b)
            ti += 1
        elif kind == "p":
            ti += 1
        else:
            raise ValueError(tok)
    # collapse: BandGRU
    _gru_graph(W, st, f"{src}collapse.rnn.", f"{dst}.bgru")
    W.t(f"{dst}.bgru.proj.w", st[f"{src}collapse.proj.weight"], big=True)
    W.t(f"{dst}.bgru.proj.b", st[f"{src}collapse.proj.bias"])
    # temporal blocks, in order
    bi = 0
    for spec in cfg["temporal"]:
        for _ in range(spec.get("n", 1)):
            p, q = f"{src}temporal.{bi}.", f"{dst}.t{bi}"
            if spec["t"] == "tb":
                w, b = fold_bn_after(st[p + "dw.weight"], st[p + "dw.bias"], st, p + "bn")
                W.t(q + ".dw.w", _dw_cl(w))
                W.t(q + ".dw.b", b)
                W.t(q + ".pw.w", st[p + "pw.weight"][:, :, 0], big=True)
                W.t(q + ".pw.b", st[p + "pw.bias"])
            elif spec["t"] == "gru":
                _gru_host(W, st, p + "rnn.", q + ".rnn", ln=p + "norm")
                W.t(q + ".out.w", st[p + "out.weight"], big=True)
                W.t(q + ".out.b", st[p + "out.bias"])
            elif spec["t"] == "attn":
                qw, qb = fold_ln(_f64(st, p + "n"), p + "n", st[p + "qkv.weight"].astype(np.float64),
                                 st[p + "qkv.bias"].astype(np.float64))
                W.t(q + ".qkv.w", qw, big=True)
                W.t(q + ".qkv.b", qb)
                W.t(q + ".o.w", st[p + "o.weight"], big=True)
                W.t(q + ".o.b", st[p + "o.bias"])
            else:
                raise ValueError(spec)
            bi += 1
    # onset heads: the shared linear head, then the family towers (input weights of every tower stacked: one matmul)
    W.t(f"{dst}.rest.w", st[f"{src}on.rest.weight"], big=True)
    W.t(f"{dst}.rest.b", st[f"{src}on.rest.bias"])
    n_t = len(cfg["towers"])
    ihw, ihb = [], []
    for g in range(n_t):
        gp = f"{src}on.towers.{g}."
        for s in ("_l0", "_l0_reverse"):
            ihw.append(st[gp + "weight_ih" + s])
            ihb.append(st[gp + "bias_ih" + s])
        W.t(f"{dst}.tw{g}.hh.w", np.stack([st[gp + "weight_hh" + s] for s in ("_l0", "_l0_reverse")]))
        W.t(f"{dst}.tw{g}.hh.b", np.stack([st[gp + "bias_hh" + s] for s in ("_l0", "_l0_reverse")]))
        W.t(f"{dst}.tw{g}.head.w", st[f"{src}on.heads.{g}.weight"])           # (|g|, 2h + w), host side, f32
        W.t(f"{dst}.tw{g}.head.b", st[f"{src}on.heads.{g}.bias"])
    W.t(f"{dst}.tw.ih.w", np.concatenate(ihw, 0), big=True)                    # (towers * 2 * 3h, w)
    W.t(f"{dst}.tw.ih.b", np.concatenate(ihb, 0))
    # velocity: Linear(w, 16) then Linear(16, K), no activation -> one map
    w1, b1 = st[f"{src}vel.0.weight"].astype(np.float64), st[f"{src}vel.0.bias"].astype(np.float64)
    w2, b2 = st[f"{src}vel.1.weight"].astype(np.float64), st[f"{src}vel.1.bias"].astype(np.float64)
    W.t(f"{dst}.vel.w", w2 @ w1, big=True)
    W.t(f"{dst}.vel.b", w2 @ b1 + b2)


def expert(W, st, src: str, dst: str, cfg: dict):
    drumnet(W, st, src + "base.", dst + ".base", cfg["drum_kit"])
    for c in ("c1", "c2", "c3"):
        W.t(f"{dst}.cq.{c}.w", st[f"{src}cq.{c}.weight"], big=True)           # (OC, IC, KT, KF) -> ne [KF, KT, IC, OC]
        W.t(f"{dst}.cq.{c}.b", st[f"{src}cq.{c}.bias"])
    for n in ("n1", "n2", "n3"):
        W.t(f"{dst}.cq.{n}.w", st[f"{src}cq.{n}.weight"])
        W.t(f"{dst}.cq.{n}.b", st[f"{src}cq.{n}.bias"])
    W.t(f"{dst}.cq.proj.w", st[f"{src}cq.proj.weight"], big=True)             # (d, C * F), column c * F + f
    W.t(f"{dst}.cq.proj.b", st[f"{src}cq.proj.bias"])
    W.t(f"{dst}.inp.w", st[f"{src}inp.weight"], big=True)
    W.t(f"{dst}.inp.b", st[f"{src}inp.bias"])
    for layer in range(cfg["orchestral"]["adapter"]["layers"]):
        _gru_host(W, st, f"{src}rnn.", f"{dst}.rnn{layer}", layer=layer)
    for n in ("qkv", "ao", "delta", "timp"):
        W.t(f"{dst}.{n}.w", st[f"{src}{n}.weight"], big=True)
        W.t(f"{dst}.{n}.b", st[f"{src}{n}.bias"])
    W.t(f"{dst}.ln.w", st[f"{src}ln.weight"])
    W.t(f"{dst}.ln.b", st[f"{src}ln.bias"])
    W.t(f"{dst}.gate0.w", st[f"{src}gate.0.weight"])                          # host side (one vector per recording)
    W.t(f"{dst}.gate0.b", st[f"{src}gate.0.bias"])
    W.t(f"{dst}.gate2.w", st[f"{src}gate.2.weight"])
    W.t(f"{dst}.gate2.b", st[f"{src}gate.2.bias"])


def convert(W, cfg: dict, st: dict, prefix: str = "dr"):
    import torch

    model = _ref("dettato.drums.model")
    fe = _ref("dettato.drums.features")
    assert cfg["model_type"] == "dettato-drums" and cfg["keys"] == model.KEYS and cfg["timpani_keys"] == model.TIMPANI_KEYS
    kc = cfg["drum_kit"]
    assert kc.get("collapse", "fgru") == "fgru" and kc.get("act", "relu") == "silu", "the C++ port assumes fgru + SiLU"
    kinds = [s["t"] for s in kc["temporal"] for _ in range(s.get("n", 1))]
    assert kinds == ["tb", "gru", "attn"], f"the C++ port assumes temporal blocks tb, gru, attn, got {kinds}"
    tb, gr, at = kc["temporal"]
    idx = [model.KEYS.index(k) for g in kc["towers"] for k in g]
    for k in [k for k in st if k.endswith(".on.idx")]:
        assert list(st[k]) == idx, f"{k}: the towers' key order differs from the config"
    st = _Tracked(st)
    p = prefix
    # constants
    W.u32(f"{p}.sample_rate", cfg["sample_rate"])
    W.u32(f"{p}.min_source_rate", cfg["min_source_rate"])
    W.u32(f"{p}.output_fps", cfg["output_fps"])
    W.arr(f"{p}.keys", cfg["keys"])
    W.arr(f"{p}.timpani_keys", cfg["timpani_keys"])
    # thresholds: f32 for the probability comparisons (numpy compares a float32 map with the threshold cast to float32);
    # the router is compared in float64, so it is also stored exactly, as its repr
    for k, v in cfg["thresholds"].items():
        W.f32(f"{p}.th.{k}", v)
    W.s(f"{p}.th.router_repr", repr(float(cfg["thresholds"]["router"])))
    # infer.py hard-codes its decode constants (peaks' default gap, the 40 ms timpani rule, 100 ms hits, 50 fps); the
    # config's "decode" block documents them. Store the code's values, exactly (repr), after checking they agree.
    infer = _ref("dettato.drums.infer")
    assert infer.FPS == cfg["output_fps"] == 50
    assert cfg["decode"] == {"peak_gap_frames": 3, "timpani_repeat_s": 0.04, "hit_length_s": 0.1}, cfg["decode"]
    W.u32(f"{p}.decode.peak_gap_frames", 3)
    W.s(f"{p}.decode.timpani_repeat_s", repr(0.04))
    W.s(f"{p}.decode.hit_length_s", repr(0.1))
    W.s(f"{p}.stem", kc["stem"])
    W.u32(f"{p}.w", kc["w"])
    W.u32(f"{p}.tb.k", tb.get("k", TB_DEFAULT_K))
    W.u32(f"{p}.tb.d", tb.get("d", TB_DEFAULT_D))
    W.u32(f"{p}.gru.h", gr.get("h", GRU_DEFAULT_H))
    W.u32(f"{p}.attn.da", at.get("da", ATTN_DEFAULT_DA))
    W.u32(f"{p}.attn.heads", at.get("heads", ATTN_DEFAULT_HEADS))
    W.u32(f"{p}.collapse_m", kc["collapse_m"])
    W.u32(f"{p}.tower_h", kc.get("tower_h", 24))
    W.arr(f"{p}.tower_sizes", [len(g) for g in kc["towers"]])
    W.arr(f"{p}.tower_keys", [k for g in kc["towers"] for k in g])
    ad = cfg["orchestral"]["adapter"]
    W.u32(f"{p}.n_experts", cfg["orchestral"]["n_models"])
    for k in ("d", "h", "layers", "heads", "cq_d"):
        W.u32(f"{p}.ex.{k}", ad[k])
    # code constants of model.py / features.py / infer.py
    W.u32(f"{p}.embed_chunk", 8000)
    W.u32(f"{p}.embed_margin", 200)
    W.u32(f"{p}.kit_long_above", 9000)
    W.u32(f"{p}.n_mel", model.N_MEL)
    W.u32(f"{p}.n_cq", model.N_CQ)
    W.f32(f"{p}.log_floor", fe.FLOOR)
    W.f32(f"{p}.level_dbfs", -18.0)
    # front end: the filterbanks and windows as the Python code builds them
    for name in ("lo", "hi", "cq"):
        W.t(f"{p}.fb.{name}", fe._filters(name).numpy())                       # (bands, bins) -> ne [bins, bands]
    for n in (512, 1411, fe.CQ_NFFT):
        W.t(f"{p}.win.{n}", torch.hann_window(n).numpy())
    W.u32(f"{p}.cq_nfft", fe.CQ_NFFT)
    # the networks
    drumnet(W, st, "drum_kit.", f"{p}.kit", kc)
    for i in range(cfg["orchestral"]["n_models"]):
        expert(W, st, f"orchestral.{i}.", f"{p}.ex{i}", cfg)
    # every state entry must have been read (the BatchNorm step counters and the towers' index buffer aside)
    left = sorted(k for k in st if k not in st.read and not k.endswith("num_batches_tracked") and not k.endswith(".on.idx"))
    assert not left, f"unconverted state entries: {left[:8]}"
    return st.read
