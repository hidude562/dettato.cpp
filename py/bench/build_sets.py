"""Build the nine benchmark sets from what download.py fetched.

    python build_sets.py DOWNLOADS SETS_DIR [bach10 dcs phenicx rwc_p rwc_c rwc_g rwc_j rwc_r slakh_test]   (default: all)

Each set becomes SETS_DIR/<set>/
  manifest.jsonl     one line per recording (id, duration, ...)
  wav/<id>.wav       16 kHz mono PCM16: the input every system in the table is fed
  labels/<id>.npz    reference notes: onset, offset (s), pitch, program (128 = drums on RWC), is_drum, velocity
  full/<id>.flac     the recording at its own rate, lossless (RWC and Slakh only: the sets with drums; the drum model reads up to 20 kHz)

References (one note per row; program = the General MIDI program given here, only used to drop overlapping duplicates):
- bach10   the mix; notes from <piece>-GTNotes.mat (per part, per note: contiguous frame numbers + per-frame MIDI pitch; onset /
           offset = first / last frame centre, 23 ms + 10 ms hop; pitch = rounded median), violin / clarinet / sax / bassoon.
- dcs      Dagstuhl ChoirSet, the 20 takes with a time-aligned score (full choir + quartets), the room microphone mixdown
           (Stereo_STM, 22.05 kHz mono); notes = annotations_csv_scorerepresentation S / A / T / B.
- phenicx  PHENICX-Anechoic, 4 pieces; the mix = the sum of every player's anechoic stem; notes = annotations/<piece>/<inst>.txt.
- rwc_*    RWC 2.0 audio, notes = rwc-annotations MIDI_aligned. Vocal melody tracks (metadata_mapping_vocals.csv) are moved DOWN by
           their SemitoneOffset (the MIDI melody is written 1-2 octaves above the sung pitch where that column says so).
- slakh    Slakh2100 test (151 songs) as YourMT3 prepares it: only stems rendered into the mix (slakh_test_stems.json, YourMT3's
           index, and audio_rendered in metadata.yaml); bass programs 32-39 one octave down except the slap bass plugin; the
           program and the drum flag come from metadata.yaml.
Audio: mono = the mean of the channels; 16 kHz with soxr. The 8 non-Slakh sets are scaled only if they would clip (peak -> 0.95),
Slakh is resampled with soxr VHQ and clipped to [-1, 1].
"""
import csv
import glob
import io
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import soundfile as sf
import soxr

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["bach10", "dcs", "phenicx", "rwc_p", "rwc_c", "rwc_g", "rwc_j", "rwc_r", "slakh_test"]
NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
BACH10_PROG = {1: 40, 2: 71, 3: 66, 4: 70}
PHENICX_PROG = {"violin": 40, "viola": 41, "cello": 42, "doublebass": 43, "flute": 73, "oboe": 68, "clarinet": 71,
                "bassoon": 70, "horn": 60, "trumpet": 56}
RWC_ZIP = {"rwc_p": "RWC-P", "rwc_c": "RWC-C", "rwc_g": "RWC-G", "rwc_j": "RWC-J", "rwc_r": "RWC-R"}
SLAP_BASS = "scarbee_jay_bass_slap_both.nkm"


def note_number(name):
    """'C#5' / 'Bb3' -> MIDI number (C4 = 60)."""
    i, pc = 1, NOTE_NAMES[name[0].upper()]
    while i < len(name) and name[i] in "#b":
        pc += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + pc


def to_16k(x, sr):
    """float mono -> 16 kHz, scaled only if it would clip."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    gain = min(1.0, 0.95 / peak) if peak > 0 else 1.0
    return np.clip(soxr.resample(x * gain, sr, SR), -1, 1), gain


class Writer:
    def __init__(self, root, name):
        self.name, self.dir = name, os.path.join(root, name)
        for d in ("wav", "labels"):
            os.makedirs(os.path.join(self.dir, d), exist_ok=True)
        self.rows = []

    def add(self, sid, y, gain, notes, **meta):
        """notes: list of (onset, offset, pitch, program, is_drum)."""
        sf.write(os.path.join(self.dir, "wav", sid + ".wav"), y, SR, subtype="PCM_16")
        a = np.array(sorted(notes), dtype=np.float64).reshape(-1, 5)
        n = len(a)
        np.savez(os.path.join(self.dir, "labels", sid + ".npz"), onset=a[:, 0].astype(np.float32),
                 offset=np.maximum(a[:, 1], a[:, 0] + 1e-3).astype(np.float32), pitch=a[:, 2].astype(np.int16),
                 program=a[:, 3].astype(np.int16), is_drum=a[:, 4].astype(np.int8), velocity=np.full(n, 100, np.int16))
        self.rows.append(dict(id=sid, duration=round(len(y) / SR, 3), n_notes=n, gain=round(gain, 4), **meta))
        print(f"{self.name} {sid}: {n} notes ({int(a[:, 4].sum())} drum), {len(y) / SR:.1f} s", flush=True)

    def full(self, sid, data, subtype, sr):
        os.makedirs(os.path.join(self.dir, "full"), exist_ok=True)
        sf.write(os.path.join(self.dir, "full", sid + ".flac"), data, sr, subtype=subtype)

    def close(self):
        with open(os.path.join(self.dir, "manifest.jsonl"), "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{self.name}: {len(self.rows)} recordings, {sum(r['duration'] for r in self.rows) / 3600:.2f} h", flush=True)


def read_mono(f):
    x, sr = sf.read(f, dtype="float32", always_2d=True)
    return x.mean(1), sr


def build_bach10(dl, root):
    import scipy.io as sio
    w = Writer(root, "bach10")
    for d in sorted(glob.glob(os.path.join(dl, "bach10", "[0-9][0-9]-*"))):
        piece = os.path.basename(d)
        y, gain = to_16k(*read_mono(os.path.join(d, piece + ".wav")))
        g = sio.loadmat(os.path.join(d, piece + "-GTNotes.mat"))["GTNotes"]
        notes = []
        for part in range(4):
            cells = g[part, 0]
            for i in range(cells.shape[0]):
                fr, f0 = cells[i, 0]
                t0, t1 = 0.023 + 0.010 * (fr[0] - 1), 0.023 + 0.010 * (fr[-1] - 1)
                notes.append((t0, t1, int(round(float(np.median(f0)))), BACH10_PROG[part + 1], 0))
        w.add(piece, y, gain, notes)
    w.close()


def build_dcs(dl, root):
    w = Writer(root, "dcs")
    ann = os.path.join(dl, "dcs", "annotations_csv_scorerepresentation")
    for take in sorted({os.path.basename(f)[:-6] for f in glob.glob(os.path.join(ann, "*.csv"))}):
        y, gain = to_16k(*read_mono(os.path.join(dl, "dcs", "audio_wav_22050_mono", take + ".wav")))
        notes = []
        for v in "SATB":
            for row in csv.reader(open(os.path.join(ann, f"{take}_{v}.csv"))):
                if len(row) >= 3:
                    notes.append((float(row[0]), float(row[1]), int(round(float(row[2]))), 52, 0))
        w.add(take.replace("_Stereo_STM", ""), y, gain, notes)
    w.close()


def build_phenicx(dl, root):
    w = Writer(root, "phenicx")
    z = zipfile.ZipFile(os.path.join(dl, "phenicx", "PHENICX-Anechoic.zip"))
    names = [n for n in z.namelist() if n.startswith("PHENICX-Anechoic/") and "__MACOSX" not in n]
    for piece in ("beethoven", "bruckner", "mahler", "mozart"):
        mix, sr = None, None
        for s in sorted(n for n in names if n.startswith(f"PHENICX-Anechoic/audio/{piece}/") and n.endswith(".wav")):
            x, sr_ = sf.read(io.BytesIO(z.read(s)), dtype="float32", always_2d=True)
            x = x.mean(1)
            assert sr is None or sr_ == sr
            sr = sr_
            if mix is None:
                mix = np.zeros_like(x)
            if len(x) > len(mix):
                mix = np.pad(mix, (0, len(x) - len(mix)))
            mix[:len(x)] += x
        y, gain = to_16k(mix, sr)
        notes = []
        for inst, prog in PHENICX_PROG.items():
            f = f"PHENICX-Anechoic/annotations/{piece}/{inst}.txt"
            if f in names:
                for line in z.read(f).decode().splitlines():
                    p = line.strip().split(",")
                    if len(p) >= 3:
                        notes.append((float(p[0]), float(p[1]), note_number(p[2].strip()), prog, 0))
        w.add(piece, y, gain, notes)
    w.close()


def build_rwc(dl, root, name):
    import pretty_midi
    coll = RWC_ZIP[name]
    ann = os.path.join(dl, "rwc_annotations", "01_annotations_preprocessed", "MIDI_aligned")
    vocals = {}
    for r in csv.DictReader(open(os.path.join(ann, "metadata_mapping_vocals.csv")), delimiter=";"):
        vocals.setdefault(r["RWCID"], []).append((int(r["MIDITrackNo"]), r["MIDITrackName"].strip(), int(r["SemitoneOffset"] or 0)))
    z = zipfile.ZipFile(os.path.join(dl, "rwc", coll + ".zip"))
    w = Writer(root, name)
    for wav in sorted(n for n in z.namelist() if n.endswith(".wav")):
        rid = os.path.basename(wav)[:-4]
        mid = os.path.join(ann, coll, rid + ".mid")
        if not os.path.exists(mid):
            print(f"{name} {rid}: no aligned MIDI, skipped", flush=True)
            continue
        raw = z.read(wav)
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        y, gain = to_16k(x.mean(1), sr)
        pm = pretty_midi.PrettyMIDI(mid)
        voc = {}
        for trk, tname, off in vocals.get(rid, []):
            idx = [i for i, ins in enumerate(pm.instruments) if ins.name.strip() == tname]
            if len(idx) > 1 and trk - 1 in idx:
                idx = [trk - 1]
            for i in idx:
                voc[i] = off
        notes = []
        for i, ins in enumerate(pm.instruments):
            for n in ins.notes:
                if ins.is_drum:
                    notes.append((n.start, n.end, n.pitch, 128, 1))
                elif i in voc:
                    notes.append((n.start, n.end, n.pitch - voc[i], 53, 0))
                else:
                    notes.append((n.start, n.end, n.pitch, ins.program, 0))
        w.add(rid, y, gain, notes, vocal_tracks=sorted(voc.items()))
        info = sf.info(io.BytesIO(raw))
        pcm, _ = sf.read(io.BytesIO(raw), dtype="int32", always_2d=True)
        w.full(rid, pcm, info.subtype if info.subtype in ("PCM_16", "PCM_24") else "PCM_24", sr)
    w.close()


def build_slakh(dl, root):
    import pretty_midi
    import yaml
    stems = json.load(open(os.path.join(HERE, "slakh_test_stems.json")))
    w = Writer(root, "slakh_test")
    for tid in sorted(stems):
        d = os.path.join(dl, "slakh", tid)
        meta = yaml.safe_load(open(os.path.join(d, "metadata.yaml")))
        notes = []
        for sid in stems[tid]:
            st = meta["stems"][sid]
            if not st.get("audio_rendered", True):
                continue
            p, is_drum = int(st["program_num"]), bool(st["is_drum"])
            shift = -12 if (32 <= p <= 39 and st.get("plugin_name") != SLAP_BASS and not is_drum) else 0
            for ins in pretty_midi.PrettyMIDI(os.path.join(d, "MIDI", sid + ".mid")).instruments:
                for n in ins.notes:
                    notes.append((float(n.start), float(n.end), int(n.pitch) + shift, p, 1 if is_drum else 0, int(n.velocity)))
        x, sr = sf.read(os.path.join(d, "mix.flac"), dtype="float32", always_2d=True)
        y = soxr.resample(x.mean(1), sr, SR, quality="VHQ")
        if float(np.abs(y).max()) > 1.0:
            print(f"slakh_test {tid}: resampled peak {np.abs(y).max():.3f}, clipped", flush=True)
        y = np.clip(y, -1, 1)
        sf.write(os.path.join(w.dir, "wav", tid + ".wav"), y, SR, subtype="PCM_16")
        on = np.array([n[0] for n in notes])
        o = np.argsort(on)                    # Slakh keeps the stems' own velocities and the original note order per onset
        a = np.array([n[:5] for n in notes], np.float64).reshape(-1, 5)[o]
        np.savez(os.path.join(w.dir, "labels", tid + ".npz"), onset=a[:, 0].astype(np.float32), offset=a[:, 1].astype(np.float32),
                 pitch=a[:, 2].astype(np.int16), program=a[:, 3].astype(np.int16), is_drum=a[:, 4].astype(np.int8),
                 velocity=np.array([n[5] for n in notes], np.int16)[o])
        os.makedirs(os.path.join(w.dir, "full"), exist_ok=True)
        shutil.copyfile(os.path.join(d, "mix.flac"), os.path.join(w.dir, "full", tid + ".flac"))
        w.rows.append(dict(id=tid, duration=round(len(y) / SR, 3), n_notes=len(notes), src_sr=int(sr)))
        print(f"slakh_test {tid}: {len(notes)} notes, {len(y) / SR:.1f} s", flush=True)
    w.close()


def main():
    dl, root = sys.argv[1], sys.argv[2]
    for s in sys.argv[3:] or SETS:
        if s in RWC_ZIP:
            build_rwc(dl, root, s)
        else:
            {"bach10": build_bach10, "dcs": build_dcs, "phenicx": build_phenicx, "slakh_test": build_slakh}[s](dl, root)


if __name__ == "__main__":
    main()
