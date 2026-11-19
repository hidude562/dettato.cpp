"""Compare the C++ pipeline's MIDI with the Python package's on the same recordings.

    python py/bench/compare_e2e.py CPP_MID PY_MID [CPP_MID PY_MID ...]

Per pair: note counts, pitched-note onset F1 (50 ms, same pitch), drum-hit F1 (50 ms, same key), the two track lists,
and the adjusted Rand index of the track assignment over the matched pitched notes.
To make PY_MID: dettato.transcribe(path).save_midi(PY_MID) (or py/bench/variants.py for benchmark sets).
"""
from __future__ import annotations

import sys

import numpy as np
import pretty_midi


def notes(path):
    pm = pretty_midi.PrettyMIDI(path)
    pitched, drums, names = [], [], []
    for ti, ins in enumerate(pm.instruments):
        names.append(("drums" if ins.is_drum else ins.name, len(ins.notes)))
        for n in ins.notes:
            (drums if ins.is_drum else pitched).append((n.start, n.end, n.pitch, ti))
    return np.array(pitched, float).reshape(-1, 4), np.array(drums, float).reshape(-1, 4), names


def match(a, b, tol=0.05):
    """Greedy one-to-one match on (pitch, onset within tol): -> list of (i, j)."""
    pairs = []
    used = np.zeros(len(b), bool)
    order = np.argsort(a[:, 0]) if len(a) else []
    for i in order:
        cand = np.nonzero((~used) & (b[:, 2] == a[i, 2]) & (np.abs(b[:, 0] - a[i, 0]) <= tol))[0] if len(b) else []
        if len(cand):
            j = cand[np.argmin(np.abs(b[cand, 0] - a[i, 0]))]
            used[j] = True
            pairs.append((i, j))
    return pairs


def f1(n_match, n_a, n_b):
    return 2 * n_match / max(1, n_a + n_b)


def ari(x, y):
    from itertools import combinations  # noqa: F401
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = len(x)
    if n < 2:
        return 1.0
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    M = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(M, (xi, yi), 1)
    c2 = lambda v: v * (v - 1) / 2  # noqa: E731
    s = c2(M).sum()
    a, b = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    exp = a * b / c2(n)
    mx = (a + b) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def main():
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, __doc__
    for cpp, py in zip(args[::2], args[1::2]):
        pa, da, na = notes(cpp)
        pb, db, nb = notes(py)
        m = match(pa, pb)
        md = match(da, db)
        ar = ari([pa[i, 3] for i, _ in m], [pb[j, 3] for _, j in m]) if m else float("nan")
        print(f"{cpp}\n  pitched: C++ {len(pa)}  Python {len(pb)}  onset-F1 {f1(len(m), len(pa), len(pb)):.4f}"
              f"   drums: C++ {len(da)}  Python {len(db)}  F1 {f1(len(md), len(da), len(db)):.4f}   track ARI {ar:.4f}")
        print(f"  tracks C++:    {na}\n  tracks Python: {nb}")


if __name__ == "__main__":
    main()
