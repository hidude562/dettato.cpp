"""The grouper (kobimusic/dettato-grouper) -> GGUF tensors prefixed "gr.", for py/convert.py --stages ...,grouper.

What the conversion changes (exact algebra; only float rounding moves):
- LayerNorm affines fold into the linear map that follows: each transformer Block's ln1 -> qkv, ln2 -> ff.0, every
  LanePitchAttn's ln -> qkv, s15_ln -> s15_net.0 (the norm runs per tap, the linear map reads the three taps concatenated),
  LBlock's LayerNorm2d -> pw1. ln_in and ln_out stay explicit (their outputs are the residual stream / the "h" output).
- Layer-scale gammas fold into the preceding linear map (LBlock pw2, LanePitchAttn out).
- LBlock: the depthwise time conv's bias passes through (I + PitchMix) as a per-(channel, lane) constant: stored as
  "pm_bias" (P, C), added after the PitchMix matmul (before the LayerNorm2d, so it cannot go further).
- rope_time on onset seconds: x1 cos - x2 sin | x1 sin + x2 cos (halves layout). The qkv map gets two extra blocks of rows
  that produce rot(q) = [-q2, q1] and rot(k) = [-k2, k1] per head, so the graph rotates with q * cos + rot(q) * sin.
- The three note heads (inst, emb, keep) are one matrix (K + E + 1 rows).
- HDemix's prior is stored as softplus(prior) (computed by torch, as the model computes it).
- The front end's buffers (HCQT filterbanks / windows) are stored under gr.hcqt.*, the PitchMix shift indicator for the
  grouper's 96 lanes as gr.pmix_S, the rotary frequencies as gr.rope_freqs.
- The track head (applied in float64 by the decode) and the CREDIT matrix are stored in float32 at any --dtype.
The instrument table (names, first programs, families) and the decode parameters go into the metadata.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import bn_affine, fold_ln, pmix_shift  # noqa: E402,F401

PMIX_OFFSETS = (-2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31)   # grouper/model.py PitchMix default


def _rot_rows(w: np.ndarray, heads: int) -> np.ndarray:
    """Rows of a (d, in) projection -> the rows giving rot(x) = [-x2, x1] per head (halves layout)."""
    d = w.shape[0]
    D = d // heads
    half = D // 2
    assert 2 * half == D, "rope_time with an odd head dim keeps a tail; not handled"
    out = np.empty_like(w)
    for h in range(heads):
        a = h * D
        out[a:a + half] = -w[a + half:a + D]
        out[a + half:a + D] = w[a:a + half]
    return out


def convert(W, cfg: dict, st: dict, prefix: str = "gr"):
    import torch
    import torch.nn.functional as F

    import dettato_ref  # noqa: F401
    from dettato.grouper import front as FR
    from dettato.grouper.instruments import FAMILIES, INSTRUMENTS, PROGRAM, CREDIT

    p = prefix
    m = cfg["model"]
    d, heads, c = m["d"], m["heads"], m["lanes"]
    P = m["n_lanes"]
    K = len(cfg["instruments"])
    assert cfg["instruments"] == INSTRUMENTS, "instrument list of the weights does not match the reference package"
    assert list(cfg["programs"]) == [int(x) for x in PROGRAM]
    assert st["heads.inst.weight"].shape[0] == K

    # ---- metadata
    for k in ("d", "depth", "heads", "ff", "emb", "hd", "hd_sus", "lanes", "lane_depth", "lane_ts", "lane_attn", "lane_exp",
              "n_lanes", "s15_hid", "s15_dim"):
        W.u32(f"{p}.{k}", m[k])
    W.arr(f"{p}.s15_taps", [int(t) for t in m["s15_taps"]])
    W.f32(f"{p}.pos_scale", m["pos_scale"])
    W.f32(f"{p}.window_s", cfg["window_s"])
    W.f32(f"{p}.window_hop", cfg["window_hop"])
    W.u32(f"{p}.max_notes", cfg["max_notes"])
    W.u32(f"{p}.batch", cfg.get("batch", 8))
    # the decode compares float64 distances with these: stored as Python reprs, parsed exactly by the C++ side
    import inspect
    from dettato.grouper import decode as D
    dec = {k: v.default for k, v in inspect.signature(D.decode).parameters.items() if v.default is not inspect.Parameter.empty}
    dec.update(cfg["decode"])
    for k in ("tau", "tau2", "lam", "alpha", "absorb", "absorb_frac"):
        W.s(f"{p}.dec.{k}", repr(float(dec[k])))
    for k in ("min_notes", "refine"):
        W.u32(f"{p}.dec.{k}", dec[k])
    for k in ("ec", "merge"):
        W.u32(f"{p}.dec.{k}", 1 if dec[k] else 0)
    # PCEN's constants (front.pcen defaults; the model uses them as float32)
    pc = {k: v.default for k, v in inspect.signature(FR.pcen).parameters.items() if v.default is not inspect.Parameter.empty}
    for k in ("s", "alpha", "delta", "r", "eps"):
        W.f32(f"{p}.pcen.{k}", pc[k])
    W.u32(f"{p}.pcen.K", pc["K"])
    W.u32(f"{p}.n_inst", K)
    W.s(f"{p}.instruments", ",".join(INSTRUMENTS))
    W.arr(f"{p}.programs", [int(x) for x in PROGRAM])
    W.s(f"{p}.families", ";".join(",".join(FAMILIES[n]) for n in INSTRUMENTS))
    W.t(f"{p}.credit", CREDIT.astype(np.float32))                                     # (K, K) predicted x true, exact in f32

    # LaneFront structure, as LaneFront.__init__ derives it
    depth = m["lane_depth"]
    dils = [(1, 2, 4, 8)[i % 4] for i in range(depth)]
    n_attn = m["lane_attn"]
    attn_at = sorted(set(range(depth - 1, -1, -max(1, depth // max(n_attn, 1)))[:n_attn]) if n_attn else set())
    W.arr(f"{p}.lane_dil", dils)
    W.arr(f"{p}.lane_attn_at", attn_at)

    # ---- front end buffers (the grouper's own HCQT(), default arguments)
    fr = FR.Front().h
    W.u32(f"{p}.n_fft", fr.n_fft)
    W.u32(f"{p}.n_bins", fr.n_bins)
    W.arr(f"{p}.shifts", fr.shifts)
    W.u32(f"{p}.shift_lo", -fr.lo)
    W.arr(f"{p}.extra_nffts", fr.extra)
    W.f32(f"{p}.log_eps", fr.log_eps)
    W.t(f"{p}.hcqt.fb", fr.fb.numpy())
    W.t(f"{p}.hcqt.win", fr.win.numpy())
    for i, _ in enumerate(fr.extra):
        W.t(f"{p}.hcqt.fb_x{i}", getattr(fr, f"fb_x{i}").numpy())
        W.t(f"{p}.hcqt.win_x{i}", getattr(fr, f"win_x{i}").numpy())

    # ---- note embedding + scalar features
    W.t(f"{p}.pitch_emb", st["pitch_emb.weight"])                                     # (128, d) -> ne [d, 128]
    for i in (0, 2):
        W.t(f"{p}.feat.{i}.w", st[f"feat.{i}.weight"], big=True)
        W.t(f"{p}.feat.{i}.b", st[f"feat.{i}.bias"])

    # ---- HDemix encoder, Prof net
    pi = F.softplus(torch.from_numpy(st["hd.prior"])).numpy()
    W.t(f"{p}.hd.pi", pi)
    for i in (0, 2):
        W.t(f"{p}.hd.enc.{i}.w", st[f"hd.enc.{i}.weight"], big=True)
        W.t(f"{p}.hd.enc.{i}.b", st[f"hd.enc.{i}.bias"])
        W.t(f"{p}.prof.{i}.w", st[f"prof.net.{i}.weight"], big=True)
        W.t(f"{p}.prof.{i}.b", st[f"prof.net.{i}.bias"])

    # ---- LaneFront
    W.t(f"{p}.lf.stem.w", st["lanes.stem.weight"], big=True)                         # (OC, IC, KT, KF) -> ne [KF, KT, IC, OC]
    W.t(f"{p}.lf.stem.b", st["lanes.stem.bias"])
    W.t(f"{p}.lf.occ.w", st["lanes.occ.weight"].reshape(c, 4))                        # ne [4, C]
    W.t(f"{p}.lf.occ.b", st["lanes.occ.bias"])
    W.t(f"{p}.pmix_S", pmix_shift(PMIX_OFFSETS, P))                                   # ne [K + 1, P * P]
    lanes = np.arange(P)
    for i in range(depth):
        s, q = f"lanes.blocks.{i}", f"{p}.lf.blk.{i}"
        taps = np.concatenate([st[s + ".pm.taps"], np.ones((c, 1), np.float32)], 1)   # centre tap 1 = the residual y + pm(y)
        offs = list(PMIX_OFFSETS) + [0]
        W.t(q + ".dw_w", st[s + ".dw_t.weight"])                                      # (C, 1, 7, 1) -> ne [1, 7, 1, C]
        W.t(q + ".pmix", taps)                                                        # ne [K + 1, C]
        reach = np.zeros((c, P), np.float64)
        for k, o in enumerate(offs):
            inside = (lanes + o >= 0) & (lanes + o < P)
            reach[:, inside] += taps[:, k:k + 1]
        W.t(q + ".pm_bias", (st[s + ".dw_t.bias"][:, None].astype(np.float64) * reach).T)   # (P, C) -> ne [C, P]
        w1 = st[s + ".pw1.weight"].reshape(-1, c)
        w1f, b1f = w1 * st[s + ".norm.w"][None, :], st[s + ".pw1.bias"] + w1 @ st[s + ".norm.b"]
        W.t(q + ".pw1.w", w1f, big=True)
        W.t(q + ".pw1.b", b1f)
        g = st[s + ".gamma"]
        w2 = st[s + ".pw2.weight"].reshape(c, -1)
        W.t(q + ".pw2.w", w2 * g[:, None], big=True)
        W.t(q + ".pw2.b", st[s + ".pw2.bias"] * g)
    for i in attn_at:
        s, q = f"lanes.attn.{i}", f"{p}.lf.attn.{i}"
        qw, qb = fold_ln(st, s + ".ln", st[s + ".qkv.weight"], st[s + ".qkv.bias"])
        W.t(q + ".qkv.w", qw, big=True)
        W.t(q + ".qkv.b", qb)
        gm = st[s + ".gamma"]
        W.t(q + ".out.w", st[s + ".out.weight"] * gm[:, None], big=True)
        W.t(q + ".out.b", st[s + ".out.bias"] * gm)
        rb = st[s + ".rel_bias"]                                                      # (H, 2P - 1)
        rel = lanes[None, :] - lanes[:, None] + P - 1                                 # [query i, key j]
        W.t(q + ".mask", rb[:, rel])                                                  # (H, Lq, Lk) -> ne [Lk, Lq, H]
    W.t(f"{p}.lf.read.w", st["lanes.read.weight"], big=True)
    W.t(f"{p}.lf.read.b", st["lanes.read.bias"])

    # ---- s15: LayerNorm per tap folded into the first linear map of the concatenated taps
    ntap = len(m["s15_taps"])
    D15 = m["s15_dim"]
    w0 = st["s15_net.0.weight"]                                                       # (hid, ntap * D15)
    lw, lb = st["s15_ln.weight"], st["s15_ln.bias"]
    w0f = w0 * np.tile(lw, ntap)[None, :]
    b0f = st["s15_net.0.bias"] + sum(w0[:, j * D15:(j + 1) * D15] @ lb for j in range(ntap))
    W.t(f"{p}.s15.0.w", w0f, big=True)
    W.t(f"{p}.s15.0.b", b0f)
    W.t(f"{p}.s15.2.w", st["s15_net.2.weight"], big=True)
    W.t(f"{p}.s15.2.b", st["s15_net.2.bias"])

    # ---- transformer
    W.t(f"{p}.ln_in.w", st["ln_in.weight"])
    W.t(f"{p}.ln_in.b", st["ln_in.bias"])
    for i in range(m["depth"]):
        s, q = f"blocks.{i}", f"{p}.blk.{i}"
        qw, qb = fold_ln(st, s + ".ln1", st[s + ".qkv.weight"], st[s + ".qkv.bias"])     # rows: q, k, v
        wq, wk = qw[:d], qw[d:2 * d]
        bq, bk = qb[:d], qb[d:2 * d]
        rw = np.concatenate([qw, _rot_rows(wq, heads), _rot_rows(wk, heads)], 0)        # q, k, v, rot q, rot k
        rbias = np.concatenate([qb, _rot_rows(bq[:, None], heads)[:, 0], _rot_rows(bk[:, None], heads)[:, 0]])
        W.t(q + ".qkv.w", rw, big=True)
        W.t(q + ".qkv.b", rbias)
        W.t(q + ".o.w", st[s + ".o.weight"], big=True)
        W.t(q + ".o.b", st[s + ".o.bias"])
        fw, fb = fold_ln(st, s + ".ln2", st[s + ".ff.0.weight"], st[s + ".ff.0.bias"])
        W.t(q + ".ff0.w", fw, big=True)
        W.t(q + ".ff0.b", fb)
        W.t(q + ".ff2.w", st[s + ".ff.2.weight"], big=True)
        W.t(q + ".ff2.b", st[s + ".ff.2.bias"])
    D = d // heads
    half = D // 2
    W.t(f"{p}.rope_freqs", (1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))).numpy())
    W.t(f"{p}.ln_out.w", st["ln_out.weight"])
    W.t(f"{p}.ln_out.b", st["ln_out.bias"])
    hw = np.concatenate([st["heads.inst.weight"], st["heads.emb.weight"], st["heads.keep.weight"]], 0)
    hb = np.concatenate([st["heads.inst.bias"], st["heads.emb.bias"], st["heads.keep.bias"]], 0)
    W.t(f"{p}.heads.w", hw, big=True)                                                 # (K + E + 1, d)
    W.t(f"{p}.heads.b", hb)
    # the track head: always float32 (the decode runs it in float64 from these values)
    for j, i in enumerate((0, 2, 4)):
        W.t(f"{p}.th.{j}.w", st[f"heads.th.{i}.weight"])
        W.t(f"{p}.th.{j}.b", st[f"heads.th.{i}.bias"])
    W.u32(f"{p}.th_layers", 3)
    used = {k for k in st}
    return used
