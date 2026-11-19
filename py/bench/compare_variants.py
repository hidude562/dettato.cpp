"""Track-assignment agreement between two variant runs (same recordings): mean ARI over matched pitched notes, the share of
recordings with the same track-name multiset, and note-count ratio. Usage: compare_variants.py RUN_A RUN_B"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_e2e import ari, match, notes  # noqa: E402

a_dir, b_dir = sys.argv[1], sys.argv[2]
aris, same_names, ratio, f1s = [], 0, [], []
files = sorted(glob.glob(os.path.join(a_dir, "pipeline", "*", "*.mid")))
for fa in files:
    fb = os.path.join(b_dir, "pipeline", os.path.relpath(fa, os.path.join(a_dir, "pipeline")))
    pa, _, na = notes(fa)
    pb, _, nb = notes(fb)
    m = match(pa, pb)
    if m:
        aris.append(ari([pa[i, 3] for i, _ in m], [pb[j, 3] for _, j in m]))
    f1s.append(2 * len(m) / max(1, len(pa) + len(pb)))
    same_names += sorted(n for n, _ in na) == sorted(n for n, _ in nb)
    ratio.append(len(pb) / max(1, len(pa)))
print(f"{os.path.basename(b_dir)} vs {os.path.basename(a_dir)}: {len(files)} recordings | note onset-F1 {np.mean(f1s):.4f} | "
      f"track ARI mean {np.mean(aris):.4f} (min {np.min(aris):.3f}) | same track names {same_names}/{len(files)}")
