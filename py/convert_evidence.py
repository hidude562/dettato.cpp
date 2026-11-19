"""The evidence repairer (kobimusic/dettato-evidence, dettato/evidence/model.py Repairer) -> GGUF tensors prefixed "ev.".

    python py/convert.py out.gguf --stages transcriber,evidence [--dtype f16|f32]

What the conversion changes (all exact algebra, tiny float changes only):

- RMSNorm weights folded into the linear maps that read the normed output, where nothing else reads it: each block's n1 into
  q / kv, n2 into ff.w1, the backbone's final norm into the five output heads. NoteEmbed's norm is NOT folded: its output is
  the residual stream itself.
- q / kv become one matrix with rows [q, k, q', k', v] (960 x 192). q' (k') is q (k) with the rows of each rotary pair swapped
  (zero rows for the 4 unrotated dims of a head), so the interleaved continuous RoPE is q * C + q' * S with host-built C / S
  (C = cos on both rows of a pair, S = -sin on the first, +sin on the second; C = 1, S = 0 on the unrotated dims): the same
  float operations as torch's x1 * cos - x2 * sin, x1 * sin + x2 * cos.
- The five heads (keep 1, don 201, dur 161, ins.0 579, ins.1 579) become one 1521 x 192 matrix + bias.
- Buffers the Python code builds at run time are stored as torch computes them on the CPU: the rotary frequencies, the Fourier
  wavelengths, the beat-grid periods, the duration-bin centres, the duration-tolerance matrix and the 161 duration-ratio
  factors exp(rb * 0.05 - 4).
- The embedding tables stay float32 and are looked up on the host.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def convert(W, cfg: dict, st: dict, prefix: str = "ev"):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dettato_ref  # noqa: F401  (the reference package, for the run-time buffers)
    from dettato.evidence import model as M
    from dettato.evidence import repair as R

    p = prefix
    m = cfg["model"]
    d, layers, heads, n_ev = m["d"], m["layers"], m["heads"], m["n_ev"]
    hd = d // heads
    net = M.Repairer(**m)
    rope = net.bb.rope
    used = rope.used
    ffh = net.bb.blocks[0].ff.w2.in_features
    W.u32(f"{p}.d", d)
    W.u32(f"{p}.layers", layers)
    W.u32(f"{p}.heads", heads)
    W.u32(f"{p}.n_ev", n_ev)
    W.u32(f"{p}.ff_hidden", ffh)
    W.u32(f"{p}.rope_used", used)
    W.u32(f"{p}.max_win", R.MAX_WIN)
    W.u32(f"{p}.n_don", M.N_DON)
    W.u32(f"{p}.n_dr", M.N_DR)
    W.u32(f"{p}.n_ins_p", M.N_INS_P)
    W.u32(f"{p}.n_ins_t", M.N_INS_T)
    W.u32(f"{p}.n_dur", M.N_DUR)
    W.u32(f"{p}.n_slot", M.N_SLOT)
    W.f32(f"{p}.keep_th", cfg["keep_th"])
    W.f32(f"{p}.insert_th", cfg["insert_th"])
    W.f32(f"{p}.window_s", M.L_WIN)
    W.f32(f"{p}.onset_th", R.ONSET_TH)
    W.f32(f"{p}.dedup_s", R.DEDUP_S)
    assert cfg.get("window_s", M.L_WIN) == M.L_WIN

    # ---- run-time buffers, as torch builds them on the CPU (float32)
    W.t(f"{p}.rope_w", torch.cat([getattr(rope, f"w{i}") for i in range(len(rope.axes))]).numpy())   # (14,) time then pitch
    lam = lambda lo, hi, n: torch.exp(torch.linspace(math.log(lo), math.log(hi), n)).numpy()          # noqa: E731  fourier()
    W.t(f"{p}.four_on", lam(0.01, 32.0, 12))                                       # onset and onset + dur
    W.t(f"{p}.four_ld", lam(0.04, 40.0, 12))                                       # log duration
    W.t(f"{p}.grid_P", torch.exp(torch.linspace(math.log(0.18), math.log(1.5), 72)).numpy())         # grid_features
    W.t(f"{p}.dur_cent", M.DUR_CENT)                                               # (128,)
    W.t(f"{p}.dur_W", net._dur_matrix(torch.device("cpu")).numpy())                # (128, 128) [i, j]: |c_j - c_i| <= tol_i
    # candidates: torch.exp(rb * DR_STEP - DR_MAX) for every duration-ratio bin (torch.exp is MKL's vsExp on the CPU, not
    # reproducible from C; its result does not depend on the position in the tensor, so a table is exact)
    W.t(f"{p}.dr_exp", torch.exp(torch.arange(M.N_DR).float() * M.DR_STEP - M.DR_MAX).numpy())

    # ---- note embedding
    W.t(f"{p}.emb.inst", st["emb.inst.weight"])                                    # (130, d) host lookup
    W.t(f"{p}.emb.pitch", st["emb.pitch.weight"])                                  # (256, d)
    W.t(f"{p}.emb.vel", st["emb.vel.weight"])                                      # (16, d)
    for src, dst in (("emb.cont", "cont"), ("emb.evm", "evm")):
        W.t(f"{p}.{dst}.0.w", st[src + ".0.weight"], big=True)                     # (d, in) -> ne [in, d]
        W.t(f"{p}.{dst}.0.b", st[src + ".0.bias"])
        W.t(f"{p}.{dst}.2.w", st[src + ".2.weight"], big=True)
        W.t(f"{p}.{dst}.2.b", st[src + ".2.bias"])
    W.t(f"{p}.emb.norm", st["emb.norm.w"])

    # ---- backbone
    partner = np.zeros(d, np.int64)                       # row r of q' = row partner[r] of q (or a zero row when unrotated)
    rotated = np.zeros(d, bool)
    for h in range(heads):
        for j in range(hd):
            r = h * hd + j
            if j < used:
                partner[r], rotated[r] = h * hd + (j ^ 1), True
    for i in range(layers):
        b = f"bb.blocks.{i}"
        n1 = st[b + ".n1.w"][None, :]
        q = st[b + ".att.q.weight"] * n1
        kv = st[b + ".att.kv.weight"] * n1
        k, v = kv[:d], kv[d:]
        qp = np.where(rotated[:, None], q[partner], 0.0)
        kp = np.where(rotated[:, None], k[partner], 0.0)
        W.t(f"{p}.blk.{i}.qkv", np.concatenate([q, k, qp, kp, v], 0), big=True)    # (5d, d) -> ne [d, 5d]
        W.t(f"{p}.blk.{i}.o", st[b + ".att.o.weight"], big=True)
        W.t(f"{p}.blk.{i}.w1", st[b + ".ff.w1.weight"] * st[b + ".n2.w"][None, :], big=True)   # (2 ffh, d)
        W.t(f"{p}.blk.{i}.w2", st[b + ".ff.w2.weight"], big=True)                  # (d, ffh)
    nw = st["bb.norm.w"][None, :]
    hw = np.concatenate([st["keep.weight"], st["don.weight"], st["dur.weight"], st["ins.0.weight"], st["ins.1.weight"]], 0)
    hb = np.concatenate([st["keep.bias"], st["don.bias"], st["dur.bias"], st["ins.0.bias"], st["ins.1.bias"]], 0)
    assert hw.shape[0] == 1 + M.N_DON + M.N_DR + M.N_SLOT * (1 + M.N_INS_P + M.N_INS_T + M.N_DUR)
    W.t(f"{p}.head.w", hw * nw, big=True)                                          # (1521, d)
    W.t(f"{p}.head.b", hb)
    W.t(f"{p}.bb_norm_w", st["bb.norm.w"])                                         # unused by the graph: the parity tests
    return set(st)
