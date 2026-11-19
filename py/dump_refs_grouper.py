"""Reference tensors of the grouper (dettato/grouper) for the C++ parity tests.

    python py/dump_refs_grouper.py OUT_DIR [--fixtures mix,detuned,song] [--notes-device cuda|cpu] [--cases]

The grouper runs on the CPU in float32 (torch.set_num_threads(DUMP_THREADS, default 16)). Fixtures: "mix" (the 30 s mix of
testdata/refs/mix), "detuned" (that mix played 30 cents sharp, so the retune path runs) and "song" (the 4-min song). Their
input notes are the repaired notes of the Python pipeline (Transcriber + EvidenceRepairer, --notes-device; cached in
notes_in.npy). Per fixture OUT_DIR/<fixture>/:

    audio16k.npy, notes_in.npy (N, 3), tuning.npy [delta, strength, shift, r], retuned.npy (when it retunes)
    feats/      pass 2 (hop 0.5) maps, the transcriber's own notes, assign_slots output (nt), meta / feat / fsc as gather
                makes them, the taps before quantisation (X: float16-accumulated as the package does; X_f32acc and the
                matching feat_f32acc / fsc_f32acc: float32-accumulated, what the C++ port computes)
    win/        one window as predict collates it (batch of 1), through every stage: the front end (X = the reference,
                X_log, X_pcen_exact / X_exact = PCEN as an exact convolution), the readers, LaneFront, tokens, blocks, heads
    pred/ decode/ tracks/   the reference (= the package): predict() outputs, the decode's step-by-step labels, the tracks
    <variant>/{pred,decode,tracks}   the same for the variants: f16_exactpcen (the package's float16 tap accumulation and
                PCEN as an exact convolution: the C++ port's arithmetic), f32_fftpcen, exact (float32 taps + exact PCEN)
    variants.json               how far each variant (tap accumulation x PCEN) lands from the reference
cases/ (with --cases): random scipy / numpy cases for the decode's ports (linkage + fcluster, linear_sum_assignment,
np.argsort, erf).
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dettato_ref  # noqa: E402  (puts the reference package on sys.path)
from dettato.grouper.instruments import INSTRUMENTS  # noqa: E402

torch.set_num_threads(int(os.environ.get("DUMP_THREADS", "16")))


# ===================================================================================================== decode trace
def _save(d: Path, name: str, a):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", np.ascontiguousarray(a))


def _cluster_dist(e, p, lam):
    """The float32 condensed distance matrix decode._cluster hands to scipy (same numpy expressions)."""
    from scipy.spatial.distance import squareform
    e = e.astype(np.float32)
    Dm = 1 - e @ e.T
    if lam > 0:
        pn = p.astype(np.float32)
        pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)
        Dm = Dm + lam * (1 - pn @ pn.T)
    Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    np.fill_diagonal(Dm, 0)
    return squareform(Dm, checks=False)


def dump_decode(d: Path, P: dict, notes: tuple, th: list, params: dict, with_dist: bool = True):
    from dettato.grouper import decode as D
    from dettato.grouper.instruments import CREDIT

    d = Path(d)
    pd_, dd = d / "pred", d / "decode"
    on, off, pitch = (np.asarray(a, np.float64) for a in notes)
    n = len(on)
    _save(pd_, "probs", np.asarray(P["probs"], np.float32))
    _save(pd_, "emb", np.asarray(P["emb"], np.float32))
    _save(pd_, "h", np.asarray(P["h"], np.float16))
    _save(pd_, "W", np.array([P["W"]], np.float64))
    wins = P["wins"]
    E, K = P["emb"].shape[1], P["probs"].shape[1]
    _save(pd_, "win_t0", np.array([w[0] for w in wins], np.float64))
    _save(pd_, "win_off", np.cumsum([0] + [len(w[1]) for w in wins]).astype(np.int64))
    _save(pd_, "win_sel", np.concatenate([np.asarray(w[1], np.int64) for w in wins]) if wins else np.zeros(0, np.int64))
    _save(pd_, "win_emb", np.concatenate([np.asarray(w[2], np.float16) for w in wins]) if wins else np.zeros((0, E), np.float16))
    _save(pd_, "win_probs", np.concatenate([np.asarray(w[3], np.float16) for w in wins]) if wins else np.zeros((0, K), np.float16))
    _save(pd_, "notes_on", on)
    _save(pd_, "notes_off", off)
    _save(pd_, "notes_pitch", pitch)

    for i, (Wm, b) in enumerate(th):
        _save(dd, f"th_w{i}", np.asarray(Wm, np.float64))
        _save(dd, f"th_b{i}", np.asarray(b, np.float64))
    _save(dd, "credit", np.asarray(CREDIT, np.float64))
    defaults = {k: v.default for k, v in inspect.signature(D.decode).parameters.items() if v.default is not inspect.Parameter.empty}
    p = {**defaults, **params}
    keys = ("tau", "tau2", "lam", "min_notes", "refine", "alpha", "ec", "merge", "absorb", "absorb_frac")
    _save(dd, "params", np.array([float(p[k]) for k in keys], np.float64))

    if with_dist:                                                     # the float32 distances of every window (bit tests)
        ws = sorted(wins, key=lambda w: w[0])
        dist = [_cluster_dist(w[2], w[3], p["lam"]) if len(w[1]) > 1 else np.zeros(0, np.float32) for w in ws]
        _save(dd, "dist_off", np.cumsum([0] + [len(x) for x in dist]).astype(np.int64))
        _save(dd, "dist", np.concatenate(dist).astype(np.float32) if dist else np.zeros(0, np.float32))

    # ---- decode(), step by step (the body of dettato/grouper/decode.py decode, with the intermediate labels kept)
    tau, tau2, lam, mn = p["tau"], p["tau2"], p["lam"], p["min_notes"]
    tracklets = D.group_chain(P, on, tau, lam)
    _save(dd, "tracklets", tracklets.astype(np.int64))
    track = D.group_tracklet(P, on, tau, tau2, lam)
    _save(dd, "agglo", track.astype(np.int64))
    idx = np.nonzero(track >= 0)[0]
    _save(dd, "idx", idx.astype(np.int64))
    groups_ref = D.decode(P, (on, off, pitch), th, **params) if n else []
    if len(idx):
        tids, inv = np.unique(track[idx], return_inverse=True)
        T = len(tids)
        probs = P["probs"]
        size = np.bincount(inv, minlength=T)
        emb = P["emb"][idx].astype(np.float64)
        for _ in range(p["refine"]):
            keepc = np.nonzero(size >= mn)[0]
            if len(keepc) < 2:
                break
            cen = np.zeros((T, emb.shape[1]))
            np.add.at(cen, inv, emb)
            cen = cen[keepc] / np.maximum(np.linalg.norm(cen[keepc], axis=1, keepdims=True), 1e-9)
            sc = emb @ cen.T
            if lam > 0:
                pm = np.zeros((T, probs.shape[1]))
                np.add.at(pm, inv, probs[idx].astype(np.float64))
                pm = pm[keepc] / np.maximum(np.linalg.norm(pm[keepc], axis=1, keepdims=True), 1e-9)
                pn = probs[idx].astype(np.float64)
                sc = sc + lam * (pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)) @ pm.T
            inv = keepc[np.argmax(sc, 1)]
            size = np.bincount(inv, minlength=T)
        _save(dd, "refined", inv.astype(np.int64))
        mass = np.zeros((T, probs.shape[1]))
        np.add.at(mass, inv, probs[idx])
        big = np.nonzero(size >= mn)[0]
        if len(big) == 0:
            big = np.array([int(np.argmax(size))])
        isbig = np.zeros(T, bool)
        isbig[big] = True
        bign = mass[big] / np.maximum(np.linalg.norm(mass[big], axis=1, keepdims=True), 1e-9)
        src_v = mass / np.maximum(np.linalg.norm(mass, axis=1, keepdims=True), 1e-9)
        for t in range(T):
            if not isbig[t] and size[t] > 0:
                inv[inv == t] = big[int(np.argmax(bign @ src_v[t]))]
        _save(dd, "folded", inv.astype(np.int64))
        if p["absorb"] > 0 and p["absorb_frac"] > 0:
            inv = D.absorb_small(inv, emb, on[idx], off[idx], p["absorb_frac"], p["absorb"])
        _save(dd, "absorbed", inv.astype(np.int64))
        out = {}                                                      # naming + merging, to check the replication
        for t in np.unique(inv):
            ids = idx[inv == t]
            z = D.track_logits(ids, P, (on, off, pitch), th, p["alpha"])
            if p["ec"]:
                q = np.exp(z - z.max())
                k = int(np.argmax(CREDIT @ (q / q.sum())))
            else:
                k = int(np.argmax(z))
            if p["merge"] and k in out:
                out[k] = np.concatenate([out[k], ids])
            else:
                out[k if p["merge"] else (k, int(t))] = ids
        mine = [((k if p["merge"] else k[0]), np.sort(ids)) for k, ids in out.items()]
        assert len(mine) == len(groups_ref) and all(a == c and np.array_equal(b, e) for (a, b), (c, e) in zip(mine, groups_ref)), \
            "replicated decode differs from decode()"
    else:
        for k in ("refined", "folded", "absorbed"):
            _save(dd, k, np.zeros(0, np.int64))
    _save(dd, "groups_k", np.array([k for k, _ in groups_ref], np.int64))
    _save(dd, "groups_off", np.cumsum([0] + [len(ids) for _, ids in groups_ref]).astype(np.int64))
    _save(dd, "groups_ids", np.concatenate([np.asarray(ids, np.int64) for _, ids in groups_ref]) if groups_ref else np.zeros(0, np.int64))
    return groups_ref


# ===================================================================================================== random cases
def gen_cases(out: Path, seed: int = 0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    save = lambda n, a: np.save(out / f"{n}.npy", np.ascontiguousarray(a))

    # ---- average linkage + fcluster
    ms, ys, ts, labs = [], [], [], []
    for c in range(1200):
        m = int(rng.integers(1, 121)) if c % 8 == 0 else (int(rng.integers(1, 61)) if c % 4 else int(rng.integers(1, 12)))
        kind = c % 6
        if m == 1:
            y = np.zeros(0)
        elif kind == 0:                                               # continuous
            y = pdist(rng.normal(0, 1, (m, int(rng.integers(1, 6)))))
        elif kind == 1:                                               # quantised: many exact ties
            y = np.round(rng.random(m * (m - 1) // 2) * rng.integers(2, 12)) / 10.0
        elif kind == 2:                                               # duplicated points
            base = rng.normal(0, 1, (max(1, m // 3), 2))
            X = base[rng.integers(0, len(base), m)]
            y = np.round(pdist(X), 3)
        elif kind == 3:                                               # all equal
            y = np.full(m * (m - 1) // 2, float(rng.choice([0.0, 0.05, 0.1, 0.3])))
        else:                                                         # decode._cluster distances on float16 vectors
            e = rng.normal(0, 1, (m, 32)); e /= np.linalg.norm(e, axis=1, keepdims=True)
            e[rng.random(m) < 0.3] = e[0]
            e = e.astype(np.float16).astype(np.float32)
            p = np.exp(rng.normal(0, 3, (m, 73))); p /= p.sum(1, keepdims=True)
            p = p.astype(np.float16).astype(np.float32)
            pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
            Dm = 1 - e @ e.T + 0.5 * (1 - pn @ pn.T)
            Dm = np.clip((Dm + Dm.T) / 2, 0, None); np.fill_diagonal(Dm, 0)
            y = squareform(Dm, checks=False).astype(np.float64)
        y = np.asarray(y, np.float64)
        if m > 1:
            Z = linkage(y, "average")
            hs = Z[:, 2]
            tt = [0.1, float(rng.choice(hs)), float(rng.random() * (hs.max() + 0.1))]   # one threshold exactly at a merge height
            lab = [fcluster(Z, t, "distance") - 1 for t in tt]
        else:
            tt = [0.1, 0.1, 0.1]
            lab = [np.zeros(1, np.int64)] * 3
        ms.append(m); ys.append(y); ts.append(tt); labs.append(np.concatenate(lab))
    save("cluster_m", np.array(ms, np.int64))
    save("cluster_yoff", np.cumsum([0] + [len(y) for y in ys]).astype(np.int64))
    save("cluster_y", np.concatenate(ys))
    save("cluster_t", np.array(ts, np.float64))
    save("cluster_laboff", np.cumsum([0] + [len(l) for l in labs]).astype(np.int64))
    save("cluster_lab", np.concatenate(labs).astype(np.int64))

    # ---- linear_sum_assignment (minimise), like -overlap counts: integer costs with many ties
    shp, costs, rows, cols = [], [], [], []
    for c in range(3000):
        nr, nc = int(rng.integers(1, 14)), int(rng.integers(1, 14))
        if c % 3 == 0:
            C = -rng.integers(0, 3, (nr, nc)).astype(np.float64)
        elif c % 3 == 1:
            C = -rng.poisson(rng.random() * 5, (nr, nc)).astype(np.float64)
        else:
            C = rng.normal(0, 1, (nr, nc))
        if c % 17 == 0:
            C[:] = 0.0
        r, k = linear_sum_assignment(C)
        shp.append((nr, nc)); costs.append(C.ravel()); rows.append(r); cols.append(k)
    save("lsap_shape", np.array(shp, np.int64))
    save("lsap_coff", np.cumsum([0] + [len(c) for c in costs]).astype(np.int64))
    save("lsap_cost", np.concatenate(costs))
    save("lsap_roff", np.cumsum([0] + [len(r) for r in rows]).astype(np.int64))
    save("lsap_rows", np.concatenate(rows).astype(np.int64))
    save("lsap_cols", np.concatenate(cols).astype(np.int64))

    # ---- np.argsort (the default kind; on AVX2 numpy uses x86-simd-sort, which is not stable)
    xs, refs = [], []
    sizes = list(range(1, 300)) * 3 + [257, 300, 400, 600, 601, 700, 1000, 1500, 2000, 4097] * 3
    for c, n in enumerate(sizes):
        k = c % 5
        if k == 0:
            a = rng.integers(0, max(2, n // 4), n).astype(np.float64)
        elif k == 1:
            a = rng.integers(0, 3, n).astype(np.float64)
        elif k == 2:
            a = -rng.integers(1, 40, n).astype(np.float64)
        elif k == 3:
            a = np.sort(rng.integers(0, 5, n)).astype(np.float64)[:: (1 if c % 2 else -1)].copy()
        else:
            a = rng.normal(0, 1, n)
        xs.append(a); refs.append(np.argsort(a))
    save("argsort_off", np.cumsum([0] + [len(a) for a in xs]).astype(np.int64))
    save("argsort_x", np.concatenate(xs))
    save("argsort_ref", np.concatenate(refs).astype(np.int64))

    # ---- scipy.special.erf
    x = np.concatenate([rng.normal(0, 3, 60000), rng.uniform(-9, 9, 30000), rng.normal(0, 0.3, 30000),
                        [0.0, -0.0, 1.0, -1.0, np.nextafter(1.0, 2), np.nextafter(1.0, 0), 8.0, -8.0, 30.0, -30.0]])
    save("erf_x", x)
    save("erf_ref", erf(x))


# ===================================================================================================== grouper fixtures
SR = 16000
MIN_END_S = 0.03            # pipeline.py: every pitched note lasts at least 30 ms before grouping
DETUNE_CENTS = 30.0         # the "detuned" fixture: the mix played 30 cents sharp (retune.resample, soxr HQ)


def save(d: Path, name: str, a):
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    np.save(d / f"{name}.npy", np.ascontiguousarray(a))


def input_notes(x, d: Path, device: str):
    """The repaired notes the Python pipeline groups (pipeline.Dettato.transcribe): Transcriber + EvidenceRepairer on x.
    Cached in d/notes_in.npy (they are only an input: both sides group the same notes)."""
    f = d / "notes_in.npy"
    if f.exists():
        return np.load(f)
    from dettato.evidence import EvidenceRepairer
    from dettato.transcriber import Transcriber

    tr = Transcriber.from_pretrained(device)
    rp = EvidenceRepairer.from_pretrained(device)
    rep, _, info = rp(tr, x)
    notes = np.stack([rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1)
    save(d, "notes_in", notes.astype(np.float64))
    print(f"  input notes on {device}: {info}", flush=True)
    del tr, rp
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return notes


@torch.no_grad()
def run_taps(tr, audio, hop_frac=0.5, batch=4, taps=("hidden", "stem")):
    """Transcriber.run(audio, hop_frac, batch, taps, pad_batch=True) (Grouper.features' call), keeping the taps twice: in
    float16 exactly as the package accumulates them, and in float32 (what the C++ port accumulates)."""
    from dettato.transcriber.core import HEADS
    from dettato.transcriber.frontend import N_KEYS

    n = len(audio)
    W, Fw, T, starts, pad, g_pad = tr._layout(n, hop_frac)
    w = torch.from_numpy(np.hanning(Fw + 2)[1:-1].astype(np.float32))
    P = {k: torch.zeros(T, N_KEYS) for k in HEADS}
    b16, b32 = {}, {}
    wsum = torch.zeros(T)
    for ss, pr, tp in tr.windows(audio, hop_frac, batch, taps, True):
        for k, s0 in enumerate(ss):
            g0 = tr.frame_offset(s0, g_pad)
            a, b = max(0, g0), min(T, g0 + Fw)
            if b <= a:
                continue
            wt = w[a - g0:b - g0]
            for hd in HEADS:
                P[hd][a:b] += pr[hd][k, a - g0:b - g0] * wt[:, None]
            for name, v in tp.items():
                if name not in b16:
                    b16[name] = torch.zeros(T, N_KEYS, v.shape[1], dtype=torch.float16)
                    b32[name] = torch.zeros(T, N_KEYS, v.shape[1])
                xw = v[k, :, a - g0:b - g0, :N_KEYS].permute(1, 2, 0).float() * wt[:, None, None]
                b16[name][a:b] += xw.half()
                b32[name][a:b] += xw
            wsum[a:b] += wt
    ws = wsum.clamp(min=1e-6)
    maps = {hd: (P[hd] / ws[:, None]).numpy() for hd in HEADS}
    for name in b16:
        b16[name] /= ws[:, None, None].half()
        b32[name] /= ws[:, None, None]
    return maps, b16, b32


@torch.no_grad()
def gather_X(P, H, S, notes, hop):
    """core.gather's taps before quantisation: (N, 3, C) float32 (H at the onset, H over the note, S over the attack)."""
    from dettato.grouper.core import PITCH_LO, SPAN_CAP, STEM_SPAN

    T, L = P["onset"].shape
    N = len(notes)
    nt = torch.from_numpy(notes)
    lane = (nt[:, 2].round().long() - PITCH_LO).clamp(0, L - 1)
    t0 = (nt[:, 0] / hop).round().long().clamp(0, T - 1)
    te = torch.maximum((nt[:, 1] / hop).ceil().long(), t0 + 1).clamp(max=T)
    Cc = H.shape[2]

    def span(M, cap):
        acc = torch.zeros(N, Cc)
        cnt = torch.zeros(N)
        lim = torch.minimum(te, t0 + cap)
        for k in range(cap):
            t = t0 + k
            m = (t < lim).float()
            acc += M[t.clamp(max=T - 1), lane].float() * m[:, None]
            cnt += m
        return acc / cnt.clamp(min=1)[:, None]

    return torch.stack([H[t0, lane].float(), span(H, SPAN_CAP), span(S, STEM_SPAN)], 1)


def quantise(X):
    sc = X.abs().amax(-1).clamp(min=1e-6) / 127.0
    feat = (X / sc[..., None]).round().clamp(-127, 127).to(torch.int8).numpy()
    return feat, sc.numpy().astype(np.float16)


def pcen_exact(X, s=0.025, alpha=0.98, delta=2.0, r=0.5, eps=1e-6, K=200):
    """front.pcen with the causal convolution done exactly (float64, direct) instead of by a float32 FFT."""
    E = torch.exp(X.float()).permute(0, 1, 3, 2)
    k = s * (1 - s) ** np.arange(K, dtype=np.float64)
    Ed = E.double()
    T = Ed.shape[-1]
    M = torch.zeros_like(Ed)
    for j in range(min(K, T)):
        M[..., j:] += float(k[j]) * Ed[..., :T - j]
    M = M.float().clamp(min=0)
    out = (E / (eps + M) ** alpha + delta) ** r - delta ** r
    return torch.log(out + 1e-4).permute(0, 1, 3, 2)


@contextmanager
def exact_pcen(on=True):
    from dettato.grouper import front as FR

    orig = FR.pcen
    if on:
        FR.pcen = pcen_exact
    try:
        yield
    finally:
        FR.pcen = orig


def log_hcqt(front, audio):
    """Front.forward up to the stacked log HCQT (B, ch, T, 288)."""
    fr = front.h
    audio = audio.float()
    T = audio.shape[-1] // 160
    P = torch.matmul(fr.fb, fr._pow(audio, fr.n_fft, fr.win)[:, :, :T])
    lf = torch.log(P + fr.log_eps)
    chans = []
    for s in fr.shifts:
        a = s - fr.lo
        x = lf[:, a:a + fr.n_bins]
        if x.shape[1] < fr.n_bins:
            x = F.pad(x, (0, 0, 0, fr.n_bins - x.shape[1]), value=math.log(fr.log_eps))
        chans.append(x)
    for i, nf in enumerate(fr.extra):
        px = fr._pow(audio, nf, getattr(fr, f"win_x{i}"))[:, :, :T]
        chans.append(torch.log(torch.matmul(getattr(fr, f"fb_x{i}"), px) + fr.log_eps))
    return torch.stack(chans, 1).transpose(2, 3)


@torch.no_grad()
def dump_window(g, song, d: Path, wi: int):
    """One window as Grouper.predict cuts and collates it (batch of 1), through every stage of NoteNet."""
    from dettato.grouper import front as FR
    from dettato.grouper.core import collate, song_windows
    from dettato.grouper.model import note_scalars

    m = g.model
    W = g.window_s
    jobs = song_windows(song.meta[:, 0], W, W * g.window_hop, g.max_notes)
    wi = min(wi, len(jobs) - 1)
    t0, sel = jobs[wi]
    item = song.window(t0, W, sel) or song.window(t0, W, np.concatenate([sel] * 4)[:4])
    batch = collate([item], g.max_notes, "cpu")
    notes, mask = batch["notes"], batch["mask"]
    save(d, "t0", np.array([t0], np.float64))
    save(d, "index", np.array([wi], np.int64))
    save(d, "sel", np.asarray(sel, np.int64))
    save(d, "audio", batch["audio"][0])
    save(d, "notes", notes[0])
    save(d, "mask", mask[0].numpy().astype(np.uint8))
    save(d, "feat", batch["feat"][0].numpy())                                          # float16, as collate holds it
    X, P = m.front(batch["audio"])
    save(d, "X", X[0])                                                                 # the reference front end (FFT PCEN)
    save(d, "P", P[0])
    Xl = log_hcqt(m.front, batch["audio"])
    save(d, "X_log", Xl[0])
    save(d, "X_exact", FR.standardise(pcen_exact(Xl))[0])
    save(d, "X_pcen_exact", pcen_exact(Xl)[0])
    # tokens, stage by stage (NoteNet.tokens)
    x = m.pitch_emb(notes[..., 2].round().long().clamp(0, 127)) + m.feat(note_scalars(notes))
    save(d, "x_note", x[0])
    hd_raw = m.hd.raw(P, notes, mask)
    hd = m.hd.enc(hd_raw)
    save(d, "hd_raw", hd_raw[0])
    save(d, "hd", hd[0])
    B, ch, T, Fq = X.shape
    f_on, f_off, lane = FR.note_frames(notes, T)
    fb = 3 * lane
    cs = torch.cumsum(F.pad(X, (0, 0, 1, 0)), 2).permute(0, 2, 3, 1)
    cs2 = torch.cumsum(F.pad(X * X, (0, 0, 1, 0)), 2).permute(0, 2, 3, 1)
    bi = torch.arange(B)[:, None]
    Ln = (f_off - f_on).clamp(min=1).float()[..., None]
    prof = []
    for off in (0, 1):
        fbb = (fb + off).clamp(0, Fq - 1)
        m1 = (cs[bi, f_off, fbb] - cs[bi, f_on, fbb]) / Ln
        m2 = (cs2[bi, f_off, fbb] - cs2[bi, f_on, fbb]) / Ln
        prof += [m1, ((m2 - m1 * m1).clamp(min=0) + 1e-4).sqrt()]
    prof_raw = torch.cat(prof, -1)
    pr = m.prof.net(prof_raw)
    assert torch.equal(pr, m.prof(X, notes))
    save(d, "prof_raw", prof_raw[0])
    save(d, "prof", pr[0])
    lf = m.lanes
    Y = lf.stem(X)
    Tq = Y.shape[2]
    R, on, off, ln = FR.rasters(notes, mask, Tq, lf.P, 100.0 / lf.ts)
    save(d, "rasters", R[0])
    Y = Y + lf.occ(R)
    save(d, "lf_y0", Y[0])
    for i, blk in enumerate(lf.blocks):
        Y = blk(Y)
        if i == 0:
            save(d, "lf_blk0", Y[0])
        if i in lf.attn_at:
            Y = lf.attn[str(i)](Y)
            save(d, f"lf_attn{i}", Y[0])
    Yp = Y.permute(0, 2, 3, 1)
    at_on = Yp[bi, on, ln]
    csy = F.pad(Y.float(), (0, 0, 1, 0)).cumsum(2).permute(0, 2, 3, 1)
    mean = (csy[bi, off, ln] - csy[bi, on, ln]) / (off - on).clamp(min=1).float()[..., None]
    lanes = lf.read(torch.cat([at_on.float(), mean], -1))
    assert torch.equal(lanes, lf(X, notes, mask))
    save(d, "lf_at_on", at_on[0])
    save(d, "lf_mean", mean[0])
    save(d, "lanes", lanes[0])
    xa = 0
    xa = xa + hd
    xa = xa + pr
    xa = xa + lanes
    x = x + xa
    f = batch["feat"][:, :, list(m.s15_taps)].float()
    mk = mask[:, :, None, None].float()
    nn_ = mk.sum(1, keepdim=True).clamp(min=1.0)
    mu = (f * mk).sum(1, keepdim=True) / nn_
    sd = (((f - mu) ** 2) * mk).sum(1, keepdim=True).div(nn_).sqrt().clamp(min=1e-3)
    fs = (f - mu) / sd
    save(d, "s15_in", fs[0])
    s15 = m.s15_net(m.s15_ln(fs).flatten(2))
    save(d, "s15", s15[0])
    tok = x + s15
    assert torch.equal(tok, m.tokens(batch))
    save(d, "tok", tok[0])
    x = m.ln_in(tok)
    save(d, "ln_in", x[0])
    pos = notes[..., 0] * m.pos_scale
    for i, blk in enumerate(m.blocks):
        x = blk(x, pos, mask)
        save(d, f"blk{i}", x[0])
    out = m.heads(m.ln_out(x))
    ref = m(batch)
    for k in ("inst", "emb", "keep", "h"):
        assert torch.equal(out[k], ref[k]), k
        save(d, k, out[k][0])
    print(f"  window {wi}: t0 {t0:.2f} s, {len(sel)} notes (collated to {notes.shape[1]})", flush=True)


def labels_of(groups, n):
    """Decoded groups -> per-note instrument index and track index (-1 = none)."""
    inst, trk = -np.ones(n, np.int64), -np.ones(n, np.int64)
    for t, (k, ids) in enumerate(groups):
        inst[ids] = k
        trk[ids] = t
    return inst, trk


def save_tracks(d: Path, tracks, info=None):
    save(d, "inst", np.array([INSTRUMENTS.index(t["instrument"]) for t in tracks], np.int64))
    save(d, "program", np.array([t["program"] for t in tracks], np.int64))
    save(d, "off", np.cumsum([0] + [len(t["notes"]) for t in tracks]).astype(np.int64))
    save(d, "notes", np.concatenate([t["notes"] for t in tracks]) if tracks else np.zeros((0, 4)))
    if info is not None:
        (d / "info.json").write_text(json.dumps(info, indent=1))


def compare(P0, l0, P1, l1):
    from sklearn.metrics import adjusted_rand_score

    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    return dict(probs_rel=rel(P1["probs"], P0["probs"]), emb_rel=rel(P1["emb"], P0["emb"]),
                argmax_agree=float((P1["probs"].argmax(1) == P0["probs"].argmax(1)).mean()),
                ari=float(adjusted_rand_score(l0[1], l1[1])), same_inst=float((l0[0] == l1[0]).mean()))


def fixture(name: str, x: np.ndarray, out: Path, a):
    from dettato.grouper import Grouper, retune
    from dettato.grouper.core import _Song, assign_slots, gather
    from dettato.transcriber import Transcriber

    t_start = time.time()
    d = out / name
    save(d, "audio16k", x.astype(np.float32))
    notes = input_notes(x, d, a.notes_device)
    print(f"{name}: {len(x) / SR:.1f} s, {len(notes)} input notes", flush=True)
    g = Grouper.from_pretrained("cpu")
    tr = Transcriber.from_pretrained("cpu")
    # ---- Grouper.group, step by step
    delta, strength, shift, r = retune.ratio(x)
    save(d, "tuning", np.array([delta, strength, shift, r], np.float64))
    y = retune.resample(x, r) if shift else x
    if shift:
        save(d, "retuned", y)
    nin = np.array(notes, np.float64).reshape(-1, 3)
    nin[:, :2] /= r
    print(f"  tuning {delta:+.2f} c, strength {strength:.3f}, shift {shift:+.2f} c, r {r:.6f}", flush=True)
    t = time.time()
    maps, H16, H32 = run_taps(tr, y)
    print(f"  pass 2: {time.time() - t:.0f} s", flush=True)
    tn = tr.notes(maps)
    own = np.stack([tn["onset"], tn["offset"], tn["pitch"].astype(np.float64), tn["slot"] - 1.0], 1).reshape(-1, 4)
    nt = assign_slots(nin, own)
    Pm = {k: torch.from_numpy(v) for k, v in maps.items()}
    meta, feat, fsc = gather(Pm, H16["hidden"], H16["stem"], nt, tr.hop)
    X16 = gather_X(Pm, H16["hidden"], H16["stem"], nt, tr.hop)
    q16, s16 = quantise(X16)
    assert np.array_equal(q16, feat) and np.array_equal(s16, fsc), "gather_X does not reproduce gather"
    X32 = gather_X(Pm, H32["hidden"], H32["stem"], nt, tr.hop)
    feat32, fsc32 = quantise(X32)
    del H16, H32
    fd = d / "feats"
    for k, v in maps.items():
        save(fd, f"map_{k}", v)
    save(fd, "own", own)
    save(fd, "nt", nt)
    save(fd, "meta", meta)
    save(fd, "feat", feat)
    save(fd, "fsc", fsc)
    save(fd, "X", X16)
    save(fd, "X_f32acc", X32)
    save(fd, "feat_f32acc", feat32)
    save(fd, "fsc_f32acc", fsc32)
    q_agree = float((feat32 == feat).mean())
    print(f"  {len(nt)} notes; int8 taps equal between float16 and float32 accumulation: {q_agree:.4f}", flush=True)
    on, off = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64)
    pitch = meta[:, 2].round().astype(np.int64)
    conf = np.where(meta[:, 3] > 0.5, meta[:, 5], meta[:, 4])
    nd = (on, off, pitch.astype(np.float64))
    # ---- the reference (the package) and the variants
    res = {}
    for var, taps32, exact in (("ref", False, False), ("f16_exactpcen", False, True), ("f32_fftpcen", True, False),
                               ("exact", True, True)):
        t = time.time()
        song = _Song(y, meta, feat32 if taps32 else feat, fsc32 if taps32 else fsc)
        with exact_pcen(exact):
            if var == "ref" and a.window.get(name) is not None:
                dump_window(g, song, d / "win", a.window[name])
            P = g.predict(song)
        dd = d if var == "ref" else d / var
        groups = dump_decode(dd, P, nd, g.th, g.params, with_dist=(var == "ref"))
        tracks = g._tracks(on * r, off * r, pitch, conf, groups)
        lab = labels_of(groups, len(on))
        save(dd / "tracks", "labels_inst", lab[0])
        save(dd / "tracks", "labels_track", lab[1])
        info = dict(tuning_cents=round(float(delta), 2), tuning_strength=round(float(strength), 3), retune_cents=round(float(shift), 2),
                    retune_ratio=float(r), duration_s=round(len(x) / SR, 2), notes_in=len(on),
                    notes_out=int(sum(len(t_["notes"]) for t_ in tracks)))
        save_tracks(dd / "tracks", tracks, info)
        res[var] = (P, lab)
        print(f"  {var:14s} predict+decode {time.time() - t:5.1f} s: " +
              ", ".join(f"{t_['instrument']} {len(t_['notes'])}" for t_ in tracks), flush=True)
    rep = {"int8_taps_equal_f16_vs_f32": q_agree}
    P0, l0 = res["ref"]
    for var, (P1, l1) in res.items():
        if var != "ref":
            rep[var] = compare(P0, l0, P1, l1)
            print(f"  {var:14s} vs ref: {rep[var]}", flush=True)
    (d / "variants.json").write_text(json.dumps(rep, indent=1))
    if a.verify and name in a.verify:
        # the replication above against the package's own Grouper.group
        t = time.time()
        tracks_pkg, info_pkg = g.group(x, notes, tr)
        ref_notes = np.load(d / "tracks" / "notes.npy")
        mine = np.concatenate([t_["notes"] for t_ in tracks_pkg]) if tracks_pkg else np.zeros((0, 4))
        assert np.array_equal(mine, ref_notes), "Grouper.group differs from the step-by-step replication"
        print(f"  verified against Grouper.group ({time.time() - t:.0f} s)", flush=True)
    (d / "REF_COMMIT").write_text(dettato_ref.ref_commit() + "\n")
    print(f"  {name} done in {time.time() - t_start:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fixtures", default="mix,detuned,song")
    ap.add_argument("--notes-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--windows", default="mix:1,song:9", help="fixture:window index to dump layer by layer")
    ap.add_argument("--verify", default="mix", help="fixtures to check against Grouper.group itself (one more pass 2)")
    ap.add_argument("--cases", action="store_true", help="also write the random scipy / numpy cases (cases/)")
    a = ap.parse_args()
    a.window = {k: int(v) for k, v in (s.split(":") for s in a.windows.split(",") if s)}
    a.verify = set(a.verify.split(",")) if a.verify else set()
    out = Path(a.out)
    root = Path(__file__).resolve().parent.parent
    if a.cases:
        gen_cases(out / "cases")
        print("cases written", flush=True)
    mix = np.load(root / "testdata/refs/mix/audio16k.npy").astype(np.float32)
    for name in a.fixtures.split(","):
        if name == "mix":
            x = mix
        elif name == "detuned":
            import soxr

            r = 2.0 ** (DETUNE_CENTS / 1200.0)
            x = soxr.resample(mix, SR, SR / r, quality="HQ").astype(np.float32)
        elif name == "song":
            x = np.load(root / "testdata/song4min_16k.npy").astype(np.float32)
        else:
            raise SystemExit(f"unknown fixture {name}")
        fixture(name, x, out, a)


if __name__ == "__main__":
    main()
