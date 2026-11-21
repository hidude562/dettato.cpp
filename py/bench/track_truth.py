"""Track assignment against the reference instruments of the benchmark sets.

    python py/bench/track_truth.py SETS_DIR RUN_DIR [RUN_DIR ...]

RUN_DIR holds pipeline/<set>/<id>.mid, as run_cpp_sets.sh and run_dettato.py write it. Per recording: the pitched notes
are matched to the reference notes (same pitch, onsets within 50 ms), and the adjusted Rand index is taken between each
matched note's track and its reference note's program. Prints the mean per set, the mean over the sets, the share of the
reference notes that were matched and the number of tracks over the number of reference instruments. A set with one
reference instrument (Dagstuhl ChoirSet) scores 1 for every run.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_e2e import ari, match, notes  # noqa: E402

sets_dir, runs = sys.argv[1], sys.argv[2:]
SETS = ["bach10", "dcs", "phenicx", "rwc_r"]
print(f"{'run':28s} " + " ".join(f"{s:>8s}" for s in SETS) + "     mean  matched  tracks / instruments")
for run in runs:
    per_set, matched, ratio = [], [], []
    for s in SETS:
        vals = []
        for lab in sorted(glob.glob(os.path.join(sets_dir, s, "labels", "*.npz"))):
            mid = os.path.join(run, "pipeline", s, os.path.basename(lab)[:-4] + ".mid")
            if not os.path.exists(mid):
                continue
            z = np.load(lab)
            keep = ~z["is_drum"].astype(bool)
            ref = np.stack([z["onset"][keep], z["offset"][keep], z["pitch"][keep], z["program"][keep]], 1).astype(float)
            pred, _, _ = notes(mid)
            m = match(pred, ref)
            if len(m) < 2:
                continue
            vals.append(ari([pred[i, 3] for i, _ in m], [ref[j, 3] for _, j in m]))
            matched.append(len(m) / max(1, len(ref)))
            ratio.append(len(set(pred[:, 3])) / max(1, len(set(ref[:, 3]))))
        per_set.append(np.mean(vals) if vals else float("nan"))
    print(f"{os.path.basename(run.rstrip('/')):28s} " + " ".join(f"{v:8.3f}" for v in per_set)
          + f"   {np.nanmean(per_set):6.3f}   {np.mean(matched):6.3f}   {np.mean(ratio):5.2f}")
