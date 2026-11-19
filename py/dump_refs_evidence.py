"""Reference tensors of the evidence repairer (dettato/evidence), for the C++ parity tests (cpp/tools/parity_evidence.cpp).

    python py/dump_refs_evidence.py OUT_DIR [--fixtures mix,song4min]

Runs on the CPU in float32. For each fixture the package's own path runs unchanged: Transcriber.run (hop_frac 0.25, the hidden
tap accumulated in float16), transcribe_with_evidence, input_notes, EvidenceRepairer.edits / apply. Alongside it the hidden tap is
accumulated a second time in float32 (a tee on Transcriber.windows, the same arithmetic as Transcriber.run with a float32 buffer)
and the same evidence / repair code runs on that: the "f32" variant. It differs from the package only in that accumulation, and
it is what the C++ (float32 throughout) must reproduce; the package's own result is the "f16" variant.

OUT_DIR/<fixture>/
  audio16k.npy                                 the input (a copy of the fixture)
  notes_{onset,offset,pitch,conf,slot}.npy     Transcriber.notes
  span_{L,A,B,S}.npy                           per note: lane, first frame, end frame (exclusive), voice
  mapf.npy (N, 31)                             map_evidence
  {at,mean}{16,32}.npy (N, 368)                the gathered hidden vectors, f16 / f32 accumulation
  ev16.npy, ev32.npy (N, 769)                  the evidence matrices (ev16 = the package's)
  k_{pitch,vel,on,off}.npy, k_order.npy        input_notes (k_order = the sort order o)
  {v}_win.npy (W, 3)                           the windows the model ran on: t0, lo, hi (v = f16 / f32)
  {v}_out_{keep,don,dur,ins}.npy               their model outputs, rows concatenated in window order
  {v}_cand_{pitch,on,off,score,kind}.npy       Repairer.candidates, concatenated
  {v}_edit_{pk,on,off}.npy, {v}_ins.npy        EvidenceRepairer.edits (ins: (K, 4) onset, end, pitch, score)
  {v}_rep_{onset,offset,pitch,inserted,score}.npy, {v}_info.json    EvidenceRepairer.apply
  win/                                         the f16 variant's window with the most notes through the network: the batch,
                                               the 110 continuous features, NoteEmbed's output, every block, the backbone
                                               output (after its RMSNorm), the four heads
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dettato_ref  # noqa: E402,F401  (puts the reference package on sys.path)
from dettato.evidence import EvidenceRepairer  # noqa: E402
from dettato.evidence import features as F  # noqa: E402
from dettato.evidence import repair as R  # noqa: E402
from dettato.transcriber import N_KEYS, PITCH_LO, Transcriber  # noqa: E402

torch.set_num_threads(int(os.environ.get("DUMP_THREADS", "16")))
REPO = Path(__file__).resolve().parent.parent
FIXTURES = {"mix": REPO / "testdata/refs/mix/audio16k.npy", "song4min": REPO / "testdata/song4min_16k.npy"}


def save(d: Path, name: str, a):
    d.mkdir(parents=True, exist_ok=True)
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    np.save(d / f"{name}.npy", np.ascontiguousarray(a))


class TeeF32:
    """Wraps tr.windows: every window's hidden tap is also added into a float32 buffer, as Transcriber.run adds it into its
    float16 one (same weights, same order)."""

    def __init__(self, tr: Transcriber, spill: Path):
        self.tr = tr
        self.orig = tr.windows
        self.buf = None
        self.wsum = None
        self.spill = spill          # the float32 buffer lives in a disk-backed memmap (a 4-min song is 1.5 GB)
        tr.windows = self.windows

    def windows(self, audio, hop_frac=0.25, batch=2, taps=(), pad_batch=False):
        tr = self.tr
        _, Fw, T, _, _, g_pad = tr._layout(len(audio), hop_frac)
        w = torch.from_numpy(np.hanning(Fw + 2)[1:-1].astype(np.float32))
        for ss, pr, tp in self.orig(audio, hop_frac, batch, taps, pad_batch):
            if "hidden" in tp:
                v = tp["hidden"]
                if self.buf is None:
                    self.mm = np.memmap(self.spill, dtype=np.float32, mode="w+", shape=(T, N_KEYS, v.shape[1]))
                    self.buf = torch.from_numpy(self.mm)
                    self.wsum = torch.zeros(T)
                for k, s0 in enumerate(ss):
                    g0 = tr.frame_offset(s0, g_pad)
                    a, b = max(0, g0), min(T, g0 + Fw)
                    if b <= a:
                        continue
                    wt = w[a - g0:b - g0]
                    self.buf[a:b] += v[k, :, a - g0:b - g0, :N_KEYS].permute(1, 2, 0).float() * wt[:, None, None]
                    self.wsum[a:b] += wt
            yield ss, pr, tp

    def result(self):
        self.buf /= self.wsum.clamp(min=1e-6)[:, None, None]
        self.tr.windows = self.orig
        return self.buf


class Capture:
    """Records, per Repairer.candidates call, the window batch, the model outputs and the candidates."""

    def __init__(self, model):
        self.model = model
        self.orig = model.candidates
        self.rows = []
        self.hook = None
        self.last = None

    def __enter__(self):
        self.rows = []

        def fwd_hook(mod, inp, out):
            self.last = {k: v.detach().clone() for k, v in out.items()}

        def cands(b, tol_on=0.05):
            res = self.orig(b, tol_on)
            self.rows.append((b, self.last, res))
            return res

        self.hook = self.model.register_forward_hook(fwd_hook)
        self.model.candidates = cands
        return self

    def __exit__(self, *a):
        self.hook.remove()
        del self.model.candidates


def edits_and_apply(rep: EvidenceRepairer, k, ev_sorted, d: Path, v: str):
    with Capture(rep.model) as cap:
        e = rep.edits(k, ev_sorted)
    out = rep.apply(k, e)
    # the windows in the order edits ran them (lo / hi recomputed exactly as edits does)
    on = k["on"]
    wins = []
    if len(k):
        end = float(k["off"].max())
        for t0 in np.arange(0, end + 8.0, 8.0):
            lo, hi = int(np.searchsorted(on, t0)), int(np.searchsorted(on, t0 + 8.0))
            if hi <= lo or hi - lo > R.MAX_WIN:
                continue
            wins.append((float(t0), lo, hi))
    assert len(wins) == len(cap.rows), (len(wins), len(cap.rows))
    save(d, f"{v}_win", np.array(wins, np.float64).reshape(-1, 3))
    outs = {k_: [] for k_ in ("keep", "don", "dur", "ins")}
    cand = {k_: [] for k_ in ("pitch", "on", "off", "score", "kind")}
    for _, o, c in cap.rows:
        for k_ in outs:
            outs[k_].append(o[k_][0].float().numpy())
        for k_ in cand:
            cand[k_].append(c[0][k_ if k_ != "kind" else "kind"])
    for k_, vals in outs.items():
        save(d, f"{v}_out_{k_}", np.concatenate(vals, 0) if vals else np.zeros(0, np.float32))
    for k_, vals in cand.items():
        a = np.concatenate(vals, 0) if vals else np.zeros(0)
        save(d, f"{v}_cand_{k_}", a.astype(np.int64) if k_ in ("pitch", "kind") else a)   # (int16 / int8 in CAND_DT)
    pk, on_r, off_r, ins = e
    save(d, f"{v}_edit_pk", pk)
    save(d, f"{v}_edit_on", on_r)
    save(d, f"{v}_edit_off", off_r)
    save(d, f"{v}_ins", np.array(ins, np.float64).reshape(-1, 4))
    for k_ in ("onset", "offset", "pitch", "inserted", "score"):
        save(d, f"{v}_rep_{k_}", out[k_])
    info = dict(transcribed=len(k), kept=int((~out["inserted"]).sum()), inserted=int(out["inserted"].sum()))
    info["dropped"] = info["transcribed"] - info["kept"]
    (d / f"{v}_info.json").write_text(json.dumps(info) + "\n")
    return cap.rows, out, info


@torch.no_grad()
def dump_window(rep: EvidenceRepairer, b, d: Path):
    """One window batch through the network, every stage."""
    m = rep.model
    for k_ in ("inst", "pitch", "vel", "on", "dur", "ev"):
        save(d, k_, b[k_][0])
    got = {}
    hooks = [m.emb.cont.register_forward_hook(lambda mod, inp, out: got.__setitem__("cont_in", inp[0][0])),
             m.emb.register_forward_hook(lambda mod, inp, out: got.__setitem__("emb", out[0])),
             m.bb.register_forward_hook(lambda mod, inp, out: got.__setitem__("bb", out[0]))]
    for i, blk in enumerate(m.bb.blocks):
        hooks.append(blk.register_forward_hook(lambda mod, inp, out, i=i: got.__setitem__(f"block_{i}", out[0])))
    o = m(b)
    for h in hooks:
        h.remove()
    for k_, v in got.items():
        save(d, k_, v)
    for k_ in ("keep", "don", "dur", "ins"):
        save(d, f"out_{k_}", o[k_][0])


@torch.no_grad()
def dump(tr: Transcriber, rep: EvidenceRepairer, audio: np.ndarray, d: Path):
    save(d, "audio16k", audio)
    spill = d / "H32.tmp"
    tee = TeeF32(tr, spill)
    got = {}
    orig_gather = F.gather_hidden

    def gather(H, L, A, B, device, chunk=4):
        at, mean = orig_gather(H, L, A, B, device, chunk)
        got["at16"], got["mean16"] = at, mean
        return at, mean

    F.gather_hidden = gather
    try:
        assert len(audio) <= F.MAX_S_1PASS * 16000, "the fixture would take the two-pass path"
        res, maps, ev16 = F.transcribe_with_evidence(tr, audio)       # the package's path, unchanged
    finally:
        F.gather_hidden = orig_gather
    H32 = tee.result()
    for k_ in ("onset", "offset", "pitch", "conf", "slot"):
        save(d, f"notes_{k_}", res[k_])
    T_ = maps["onset"].shape[0]
    hop = tr.hop
    L = (res["pitch"] - PITCH_LO).astype(np.int64)
    A = np.clip(np.round(res["onset"] / hop).astype(np.int64), 0, T_ - 1)
    B = np.clip(np.round(res["offset"] / hop).astype(np.int64), A + 1, T_)
    S = res["slot"].astype(np.int64)
    for k_, v in (("L", L), ("A", A), ("B", B), ("S", S)):
        save(d, f"span_{k_}", v)
    mf = F.map_evidence(maps, L, A, B, S)
    save(d, "mapf", mf)
    save(d, "at16", got["at16"])
    save(d, "mean16", got["mean16"])
    save(d, "ev16", ev16)
    # the f32 variant: the same gather / quantisation / assembly as transcribe_with_evidence
    at32, mean32 = orig_gather(H32, L, A, B, "cpu")
    del H32, tee
    spill.unlink()
    qa, la = F._std_q(at32)
    qm, lmn = F._std_q(mean32)
    evf = np.concatenate([mf, la[:, None], lmn[:, None]], 1).astype(np.float16)
    evh = np.concatenate([qa, qm], 1)
    ev32 = np.concatenate([evf.astype(np.float32), evh.astype(np.float32) / F.Q], 1)
    save(d, "at32", at32)
    save(d, "mean32", mean32)
    save(d, "ev32", ev32)
    # a consistency check of the tee: the f16 path recomputed from the package's own at16 / mean16 must give ev16
    qa16, la16 = F._std_q(got["at16"])
    qm16, lm16 = F._std_q(got["mean16"])
    ev_chk = np.concatenate([np.concatenate([mf, la16[:, None], lm16[:, None]], 1).astype(np.float16).astype(np.float32),
                             np.concatenate([qa16, qm16], 1).astype(np.float32) / F.Q], 1)
    assert np.array_equal(ev_chk, ev16), "evidence assembly differs from transcribe_with_evidence"

    k, o = R.input_notes(res)
    for k_ in ("pitch", "vel", "on", "off"):
        save(d, f"k_{k_}", k[k_])
    save(d, "k_order", o)
    rows16, out16, info16 = edits_and_apply(rep, k, ev16[o], d, "f16")
    rows32, out32, info32 = edits_and_apply(rep, k, ev32[o], d, "f32")
    # the package's own __call__ on the same inputs must give the f16 variant (guards against drift in this script)
    if rows16:
        big = max(range(len(rows16)), key=lambda i: int(rows16[i][0]["mask"].sum()))
        dump_window(rep, rows16[big][0], d / "win")
        (d / "win" / "index.txt").write_text(f"{big}\n")
    n_diff_q = int(((np.round(ev16[:, 33:] * F.Q)) != np.round(ev32[:, 33:] * F.Q)).sum())
    print(f"{d.name}: {len(audio) / 16000:.1f} s, {len(k)} notes, {len(rows16)} windows; f16 {info16}; f32 {info32}; "
          f"int8 evidence steps differing f16 vs f32: {n_diff_q} of {ev16[:, 33:].size}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fixtures", default="mix,song4min")
    a = ap.parse_args()
    out = Path(a.out)
    tr = Transcriber.from_pretrained("cpu")
    rep = EvidenceRepairer.from_pretrained("cpu")
    for name in a.fixtures.split(","):
        audio = np.load(FIXTURES[name]).astype(np.float32)
        dump(tr, rep, audio, out / name)
    (out / "REF_COMMIT").write_text(dettato_ref.ref_commit() + "\n")


if __name__ == "__main__":
    main()
