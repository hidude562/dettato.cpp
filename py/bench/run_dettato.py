"""Run the published dettato models on the benchmark sets -> one MIDI file per recording, in two configurations.

    python run_dettato.py SETS_DIR OUT_DIR [--sets bach10,dcs,...] [--device cuda]

SETS_DIR is what build_sets.py wrote: <set>/manifest.jsonl, <set>/wav/<id>.wav (16 kHz mono, the file every system in the
table is fed) and, for the sets with drums, <set>/full/<id>.flac (the recording at its own rate, for the drum model).

Writes
  OUT_DIR/transcriber/<set>/<id>.mid  the transcriber's notes + the drum model's hits: the configuration on the model cards.
                                      The transcriber can play a key twice at once and a MIDI track cannot hold two overlapping
                                      notes on the same key, so the notes are spread over as few tracks as that allows.
  OUT_DIR/pipeline/<set>/<id>.mid     everything `dettato` does (repaired notes grouped into instrument tracks + the drum hits),
                                      written by the package's own Transcription.to_midi(), the same file the CLI writes.
  OUT_DIR/log.jsonl                   one line per recording (note counts, drum router, run time)
  OUT_DIR/arrays/<set>/<id>.npz       with --arrays: the notes at full precision (transcriber, repaired, each track, drums)
One change from the package: the drum model's spectrograms are computed 60 s at a time instead of for the whole recording at
once, so a 10-18 minute recording fits in 6 GB of memory (the package needs ~8 GB for 18 minutes). Each STFT frame only reads
its own samples, so the features are the same (on a 10-minute recording 1 of 16.5M values differed, by 1e-7).
Both come from one transcriber pass (the repairer runs the transcriber and returns its notes), with the 8 s windows run in
batches of --batch (8, as in the run behind the model cards; the CLI uses 2: in bf16 on a GPU the batch size moves a few notes
by a frame, on a CPU it changes nothing). A recording whose two MIDI files
already exist is skipped, so an interrupted run picks up where it stopped. Weights come from the Hugging Face Hub.
"""
import argparse
import ctypes
import gc
import importlib
import json
import os
import sys
import time

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from dettato import audio as A
from dettato.pipeline import MIN_END_S, Dettato, Track, Transcription

SETS = ["bach10", "dcs", "phenicx", "rwc_p", "rwc_c", "rwc_g", "rwc_j", "rwc_r", "slakh_test"]
RESOLUTION = 960                 # ticks per beat at 120 BPM: 0.52 ms per tick


_drum_features = importlib.import_module("dettato.drums.features")


def logspec_blocks(x, n_fft, hop, fb, block=6000):
    """dettato.drums.features._logspec, `block` frames at a time (centre-padded STFT frames depend only on their own samples)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    xp = np.pad(x, (n_fft // 2, n_fft // 2))
    n = 1 + len(x) // hop
    win = torch.hann_window(n_fft)
    out = []
    for t0 in range(0, n, block):
        t1 = min(n, t0 + block)
        seg = torch.from_numpy(xp[t0 * hop:(t1 - 1) * hop + n_fft])
        s = torch.stft(seg, n_fft, hop_length=hop, win_length=n_fft, window=win, center=False, return_complex=True)
        out.append(torch.log(fb @ (s.abs() ** 2) + _drum_features.FLOOR).T.numpy())
    return np.concatenate(out, 0)


_drum_features._logspec = logspec_blocks


def free_memory(device):
    """Give freed memory back between models: the drum model needs ~3.5 GB extra on a 10-minute recording."""
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def layers(notes):
    """(N, 4) notes -> list of (M, 4) arrays, none holding two overlapping notes on the same key (first fit by onset)."""
    out, last = [], []                      # last[k]: {pitch: end of the latest note on that key in layer k}
    for n in notes[np.lexsort((notes[:, 2], notes[:, 0]))]:
        for k, ends in enumerate(last):
            if ends.get(n[2], -1.0) <= n[0]:
                break
        else:
            k = len(out)
            out.append([]); last.append({})
        out[k].append(n); last[k][n[2]] = n[1]
    return [np.array(o) for o in out]


def midi(tracks, path):
    """tracks: [(name, program, is_drum, (N, 4) onset / offset / pitch / velocity)] -> MIDI file."""
    pm = pretty_midi.PrettyMIDI(resolution=RESOLUTION, initial_tempo=120.0)
    for name, program, is_drum, notes in tracks:
        ins = pretty_midi.Instrument(program=0 if is_drum else int(program), is_drum=is_drum, name=name)
        for on, off, pitch, vel in notes:
            ins.notes.append(pretty_midi.Note(velocity=int(vel), pitch=int(pitch), start=float(on), end=float(off)))
        pm.instruments.append(ins)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm.write(path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def one(model, x16, full, batch):
    """The package's Dettato.transcribe, step by step, keeping the transcriber's own notes as well. Drums run first, while the
    process holds the least memory."""
    drums, dinfo = None, {}
    if full is not None:
        a = A.load(full)
        assert a.rate >= 40000, (full, a.rate)
        hits = model.drums.transcribe(a.at(44100))
        del a
        if hits:
            drums = np.array([[h["onset"], h["offset"], h["pitch"], max(1, min(127, int(round(h["velocity"] * 127))))] for h in hits],
                             np.float64)
        dinfo = dict(drum_hits=len(hits), router=getattr(model.drums, "last_info", {}).get("model"))
        free_memory(model.device)
    rep, res, rinfo = model.repairer(model.transcriber, x16, batch=batch)
    raw = np.stack([res["onset"], res["offset"], res["pitch"], np.full(len(res["onset"]), 100)], 1).reshape(-1, 4)
    raw = [(f"notes {k + 1}", 0, False, layer) for k, layer in enumerate(layers(raw))]
    notes = np.stack([rep["onset"], np.maximum(rep["offset"], rep["onset"] + MIN_END_S), rep["pitch"].astype(np.float64)], 1)
    if str(model.device).startswith("cuda"):
        torch.cuda.empty_cache()
    info = {"transcribed": rinfo["transcribed"], "dropped": rinfo["dropped"], "inserted": rinfo["inserted"]}
    try:
        groups, _ = model.grouper.group(x16, notes, model.transcriber)
        tracks = [Track(g["instrument"].replace("_", " "), int(g["program"]), np.asarray(g["notes"], np.float64)) for g in groups]
        info.update(grouped=int(sum(len(t.notes) for t in tracks)), tracks=len(tracks))
    except torch.OutOfMemoryError:          # the grouper keeps its features for the whole song on the device
        torch.cuda.empty_cache()
        tracks = None
        info["pipeline"] = "grouper out of GPU memory: rerun this recording with --device cpu"
    if drums is not None:
        if tracks is not None:
            tracks.append(Track("Drums", 0, drums, is_drum=True))
        raw.append(("Drums", 0, True, drums))
    info.update(dinfo)
    arrays = {"transcriber": np.stack([res["onset"], res["offset"], res["pitch"], res["slot"]], 1).reshape(-1, 4),
              "repaired": np.stack([rep["onset"], rep["offset"], rep["pitch"]], 1).reshape(-1, 3),
              **{f"track{k}_{t.program if not t.is_drum else 'drums'}": t.notes for k, t in enumerate(tracks or [])}}
    if raw[-1][2]:
        arrays["drums"] = raw[-1][3]
    pipe = None if tracks is None else Transcription("", len(x16) / 16000, tracks, info)
    return raw, pipe, info, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8, help="8 s windows per transcriber batch")
    ap.add_argument("--arrays", action="store_true", help="also save every stage's notes as .npz")
    a = ap.parse_args()
    torch.manual_seed(0)
    model = Dettato.from_pretrained(a.device)
    log = open(os.path.join(a.out_dir, "log.jsonl"), "a") if os.path.isdir(a.out_dir) else None
    if log is None:
        os.makedirs(a.out_dir)
        log = open(os.path.join(a.out_dir, "log.jsonl"), "a")
    for s in a.sets.split(","):
        rows = [json.loads(line) for line in open(os.path.join(a.sets_dir, s, "manifest.jsonl"))]
        for i, r in enumerate(rows):
            outs = [os.path.join(a.out_dir, c, s, r["id"] + ".mid") for c in ("transcriber", "pipeline")]
            if all(os.path.exists(o) for o in outs):
                continue
            t0 = time.time()
            x16, sr = sf.read(os.path.join(a.sets_dir, s, "wav", r["id"] + ".wav"), dtype="float32")
            assert sr == 16000 and x16.ndim == 1
            full = os.path.join(a.sets_dir, s, "full", r["id"] + ".flac")
            raw, pipe, info, arrays = one(model, x16, full if os.path.exists(full) else None, a.batch)
            if a.arrays:
                os.makedirs(os.path.join(a.out_dir, "arrays", s), exist_ok=True)
                np.savez_compressed(os.path.join(a.out_dir, "arrays", s, r["id"] + ".npz"), **arrays)
            if not os.path.exists(outs[0]):            # a rerun for the pipeline file keeps the first transcriber file
                midi(raw, outs[0])
            if pipe is not None:
                os.makedirs(os.path.dirname(outs[1]), exist_ok=True)
                pipe.to_midi().write(outs[1] + ".tmp")
                os.replace(outs[1] + ".tmp", outs[1])
            info.update(set=s, id=r["id"], seconds=round(len(x16) / 16000, 2), run_s=round(time.time() - t0, 1))
            log.write(json.dumps(info) + "\n")
            log.flush()
            print(f"{s} {i + 1}/{len(rows)} {r['id']}: {json.dumps(info)}", flush=True)
            del raw, pipe, arrays
            free_memory(model.device)


if __name__ == "__main__":
    sys.exit(main())
