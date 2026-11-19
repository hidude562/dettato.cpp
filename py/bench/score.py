"""Score MIDI transcriptions on the benchmark sets with MuScriptor's metric definitions (Rouard et al. 2026, sec. 3.3).

    python score.py SETS_DIR MIDI_DIR [--sets bach10,...] [--out results/transcriber] [--jobs 8]

MIDI_DIR holds <set>/<id>.mid (run_dettato.py writes OUT_DIR/transcriber and OUT_DIR/pipeline in this layout). Every MIDI
track counts: notes on drum tracks are drum hits, everything else is a pitched note (instrument names are not scored).

Measures (mir_eval.transcription, pitch tolerance 50 cents), F1 per recording, averaged over a set's recordings:
  onset   pitched notes, onset within 50 ms
  frame   pitched piano rolls in 62.5 ms cells, cell-wise F1
  offset  onset within 50 ms and offset within max(50 ms, 0.2 x the reference note's length)
  drums   drum hits, onset within 50 ms, the exact General MIDI key (only sets whose references have drums)
Reference cleanup, as in the paper: of overlapping notes with the same pitch and instrument, the shorter one is removed.
Composite = the mean of the four measures, each first averaged over the sets that have it. The 95 % interval resamples the
recordings of every set (2000 resamples, seed 0).

Writes OUT/per_recording.csv, OUT/summary.json and prints a table.
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
import pretty_midi
from mir_eval.transcription import match_notes
from mir_eval.util import midi_to_hz

SETS = ["bach10", "dcs", "phenicx", "rwc_p", "rwc_c", "rwc_g", "rwc_j", "rwc_r", "slakh_test"]
MEASURES = ["onset", "frame", "offset", "drums"]


def load_refs(path):
    z = np.load(path)
    return [{"onset": float(a), "offset": max(float(b), float(a) + 1e-3), "pitch": int(p), "program": int(g), "is_drum": bool(d)}
            for a, b, p, g, d in zip(z["onset"], z["offset"], z["pitch"], z["program"], z["is_drum"])]


def load_midi(path):
    out = []
    for ins in pretty_midi.PrettyMIDI(path).instruments:
        for n in ins.notes:
            out.append({"onset": float(n.start), "offset": float(n.end), "pitch": int(n.pitch), "is_drum": bool(ins.is_drum)})
    return out


def drop_overlaps(notes):
    """Of pitched notes with the same (program, pitch) whose spans overlap, keep the longer."""
    by = {}
    for i, n in enumerate(notes):
        by.setdefault((n["is_drum"], n["program"], n["pitch"]), []).append(i)
    drop = set()
    for idx in by.values():
        idx.sort(key=lambda i: notes[i]["onset"])
        keep = None
        for i in idx:
            if keep is not None and not notes[i]["is_drum"] and notes[i]["onset"] < notes[keep]["offset"]:
                short = min(keep, i, key=lambda j: notes[j]["offset"] - notes[j]["onset"])
                drop.add(short)
                keep = i if short == keep else keep
            else:
                keep = i
    return [n for i, n in enumerate(notes) if i not in drop]


def _arr(notes):
    iv = np.array([[n["onset"], max(n["offset"], n["onset"] + 1e-3)] for n in notes], dtype=float).reshape(-1, 2)
    p = np.array([n["pitch"] for n in notes], dtype=float)
    return iv, midi_to_hz(p) if len(p) else p


def n_match(ref, est, with_offset):
    if not ref or not est:
        return 0
    ri, rp = _arr(ref)
    ei, ep = _arr(est)
    if not with_offset:                       # onset only: give both a fixed span so offsets never decide
        ri = np.stack([ri[:, 0], ri[:, 0] + 1.0], 1)
        ei = np.stack([ei[:, 0], ei[:, 0] + 1.0], 1)
        return len(match_notes(ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None))
    return len(match_notes(ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05))


def frame_counts(ref, est, fps=16.0):
    if not ref and not est:
        return 0, 0, 0
    T = int(np.ceil(max(n["offset"] for n in ref + est) * fps)) + 1

    def roll(notes):
        r = np.zeros((128, T), bool)
        for n in notes:
            a, b = int(np.ceil(n["onset"] * fps)), int(np.ceil(n["offset"] * fps))
            if 0 <= n["pitch"] < 128 and b > a:
                r[n["pitch"], a:b] = True
        return r

    R, E = roll(ref), roll(est)
    return int((R & E).sum()), int(R.sum()), int(E.sum())


def f1(tp, nr, ne):
    p, r = (tp / ne if ne else 0.0), (tp / nr if nr else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


def score_recording(job):
    s, sid, ref_path, est_path = job
    ref, est = drop_overlaps(load_refs(ref_path)), load_midi(est_path)
    rp, ep = [n for n in ref if not n["is_drum"]], [n for n in est if not n["is_drum"]]
    rd, ed = [n for n in ref if n["is_drum"]], [n for n in est if n["is_drum"]]
    ftp, fnr, fne = frame_counts(rp, ep)
    return {"set": s, "id": sid, "onset": f1(n_match(rp, ep, False), len(rp), len(ep)), "frame": f1(ftp, fnr, fne),
            "offset": f1(n_match(rp, ep, True), len(rp), len(ep)),
            "drums": f1(n_match(rd, ed, False), len(rd), len(ed)) if rd else None,
            "ref_notes": len(rp), "est_notes": len(ep), "ref_drums": len(rd), "est_drums": len(ed)}


def composite(c, sets):
    """c: {(set, measure): F1 x 100} -> mean over measures of the mean over the sets that have that measure."""
    out = []
    for m in MEASURES:
        v = [c[s, m] for s in sets if c.get((s, m)) is not None]
        if v:
            out.append(np.mean(v))
    return float(np.mean(out)) if out else None


def summarize(per, sets, n_boot=2000):
    """per: {set: {id: {measure: F1 (0-1) or None}}} -> point numbers + bootstrap interval of the composite."""
    mean = {s: {m: (100 * float(np.mean(v)) if v else None) for m in MEASURES
                for v in [[c[m] for c in per[s].values() if c[m] is not None]]} for s in sets}
    c = {(s, m): mean[s][m] for s in sets for m in MEASURES}
    point = {"composite": composite(c, sets), **{m: composite({k: v for k, v in c.items() if k[1] == m}, sets) for m in MEASURES}}
    rng = np.random.default_rng(0)
    ids = {s: sorted(per[s]) for s in sets}
    boot = []
    for _ in range(n_boot):
        b = {}
        for s in sets:
            d = rng.choice(ids[s], len(ids[s]))
            dr = [i for i in ids[s] if per[s][i]["drums"] is not None]
            for m in MEASURES:
                pick = rng.choice(dr, len(dr)) if m == "drums" and dr else (d if m != "drums" else [])
                if len(pick):
                    b[s, m] = 100 * float(np.mean([per[s][i][m] for i in pick]))
        boot.append(composite(b, sets))
    return {"point": point, "ci95": [float(np.percentile(boot, q)) for q in (2.5, 97.5)] if boot else None, "per_set": mean,
            "recordings": {s: len(per[s]) for s in sets}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets_dir")
    ap.add_argument("midi_dir")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--out", default=None, help="where to write per_recording.csv and summary.json")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    sets = a.sets.split(",")
    jobs = []
    for s in sets:
        for line in open(os.path.join(a.sets_dir, s, "manifest.jsonl")):
            sid = json.loads(line)["id"]
            est = os.path.join(a.midi_dir, s, sid + ".mid")
            if not os.path.exists(est):
                raise SystemExit(f"missing {est}")
            jobs.append((s, sid, os.path.join(a.sets_dir, s, "labels", sid + ".npz"), est))
    with Pool(a.jobs) as pool:
        rows = pool.map(score_recording, jobs, chunksize=1)
    per = {s: {} for s in sets}
    for r in rows:
        per[r["set"]][r["id"]] = r
    res = summarize(per, sets)
    eight = [s for s in sets if s != "slakh_test"]
    if len(eight) == 8:
        res["composite_8_sets"] = summarize({s: per[s] for s in eight}, eight, n_boot=0)["point"]["composite"]
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "per_recording.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["set", "id"] + MEASURES + ["ref_notes", "est_notes", "ref_drums", "est_drums"])
            for r in rows:
                w.writerow([r["set"], r["id"]] + ["" if r[m] is None else f"{100 * r[m]:.2f}" for m in MEASURES]
                           + [r["ref_notes"], r["est_notes"], r["ref_drums"], r["est_drums"]])
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    print("| set | recordings | onset | frame | offset | drums |\n|---|---|---|---|---|---|")
    for s in sets:
        v = res["per_set"][s]
        print(f"| {s} | {res['recordings'][s]} | " + " | ".join("-" if v[m] is None else f"{v[m]:.1f}" for m in MEASURES) + " |")
    p = res["point"]
    print(f"\ncomposite {p['composite']:.2f} (95 % {res['ci95'][0]:.2f}-{res['ci95'][1]:.2f}); "
          + ", ".join(f"{m} {p[m]:.2f}" for m in MEASURES if p[m] is not None)
          + (f"; 8 sets without Slakh {res['composite_8_sets']:.2f}" if "composite_8_sets" in res else ""))


if __name__ == "__main__":
    main()
