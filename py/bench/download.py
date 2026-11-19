"""Download the nine benchmark sets from their public sources into one folder.

    python download.py DOWNLOADS [bach10 dcs phenicx rwc rwc_annotations slakh]      (default: all of them)

Layout it writes (build_sets.py reads this):
  bach10/<NN-Piece>/...                 Bach10 v1.1 (GitHub flippy-fyp/Bach10_v1.1 @ 2a53cdc6)
  dcs/audio_wav_22050_mono/*_Stereo_STM.wav, dcs/annotations_csv_scorerepresentation/*.csv
                                        Dagstuhl ChoirSet V1.2.3 (Zenodo 4618287; only these two folders of the 6 GB zip)
  phenicx/PHENICX-Anechoic.zip          PHENICX-Anechoic (Zenodo 840025)
  rwc/RWC-{P,C,G,J,R}.zip               RWC Music Database 2.0 audio (Zenodo 18656623, CC BY-NC 4.0, 13 GB)
  rwc_annotations/...                   rwc-annotations (GitHub rwc-music/rwc-annotations @ 0a1a6c31): the aligned MIDI
  slakh/<Track>/{mix.flac, metadata.yaml, MIDI/S**.mid}
                                        Slakh2100 test split, per file: the mixes from HF DreamyWanderer/Slakh2100-FLAC-Redux-Reduced,
                                        the stem MIDI + metadata from HF choihy/slakh2100_midi_by_ymt3 (YourMT3's copy)
Zenodo files are checked against the md5 Zenodo lists. A finished file is never fetched again, so the script can be re-run.
Zenodo throttles each connection to about 1 MB/s, so a big file comes down as 4 ranged parts in parallel.
"""
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "dettato-benchmark/1.0"}
HERE = os.path.dirname(os.path.abspath(__file__))
BACH10 = "https://codeload.github.com/flippy-fyp/Bach10_v1.1/tar.gz/2a53cdc6495bb82f03f943b77ca3b11ddf7f5a31"
RWC_ANN = "https://codeload.github.com/rwc-music/rwc-annotations/tar.gz/0a1a6c31dbe73a7f5d44f7caef8cd0999402a4c2"
DCS = ("4618287", "DagstuhlChoirSet_V1.2.3.zip", r"DagstuhlChoirSet_V1\.2\.3/(audio_wav_22050_mono/.*_Stereo_STM\.wav|annotations_csv_scorerepresentation/.*\.csv)$")
PHENICX = ("840025", "PHENICX-Anechoic.zip")
RWC = ("18656623", ["RWC-P.zip", "RWC-C.zip", "RWC-G.zip", "RWC-J.zip", "RWC-R.zip"])
SLAKH_AUDIO = "https://huggingface.co/datasets/DreamyWanderer/Slakh2100-FLAC-Redux-Reduced/resolve/368add9a2d203ba8a416c44dff8936e720f83704/data/test"
SLAKH_MIDI = "https://huggingface.co/datasets/choihy/slakh2100_midi_by_ymt3/resolve/952ad7c97128d52e8cd0368b04a2907a1e93b69c/slakh2100_yourmt3_16k/test"


def _get(url, rng=None, timeout=120):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def head_size(url):
    with _get(url, (0, 0)) as r:
        cr = r.headers.get("Content-Range")
        return int(cr.split("/")[1]) if cr else int(r.headers["Content-Length"])


def fetch_part(url, path, lo, hi, tries=20):
    """Bytes lo..hi of url into path, resuming from what path already holds."""
    for t in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if lo + have > hi:
            return
        try:
            with _get(url, (lo + have, hi)) as r, open(path, "ab") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
        except Exception as e:  # noqa: BLE001  (dropped connection / 429: back off and resume)
            print(f"  part {os.path.basename(path)}: {e!r}, retry {t + 1}", flush=True)
            time.sleep(min(60, 5 * (t + 1)))
    if os.path.getsize(path) != hi - lo + 1:
        raise RuntimeError(f"part {path} incomplete")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, md5=None, conns=4):
    if os.path.exists(dest) and (md5 is None or md5_of(dest) == md5):
        print(f"have {dest}", flush=True)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    size = head_size(url)
    step = -(-size // conns)
    parts = [(i * step, min(size, (i + 1) * step) - 1) for i in range(conns)]
    ths = [threading.Thread(target=fetch_part, args=(url, f"{dest}.part{i}", lo, hi)) for i, (lo, hi) in enumerate(parts)]
    for th in ths:
        th.start()
    t0 = time.time()
    while any(th.is_alive() for th in ths):
        time.sleep(30)
        got = sum(os.path.getsize(f"{dest}.part{i}") for i in range(conns) if os.path.exists(f"{dest}.part{i}"))
        print(f"  {os.path.basename(dest)} {got / 1e6:.0f} / {size / 1e6:.0f} MB, {got / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
    with open(dest + ".tmp", "wb") as out:
        for i in range(conns):
            with open(f"{dest}.part{i}", "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    out.write(b)
    if os.path.getsize(dest + ".tmp") != size or (md5 and md5_of(dest + ".tmp") != md5):
        raise RuntimeError(f"{dest}: size or md5 mismatch")
    os.replace(dest + ".tmp", dest)
    for i in range(conns):
        os.remove(f"{dest}.part{i}")
    print(f"done {dest} ({size / 1e6:.0f} MB)", flush=True)


def zenodo_file(record, name):
    with _get(f"https://zenodo.org/api/records/{record}") as r:
        f = {f["key"]: f for f in json.load(r)["files"]}[name]
    return f["links"]["self"], f["checksum"].split(":", 1)[1]


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos, self.size = url, 0, head_size(url)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for t in range(10):
            try:
                with _get(self.url, (self.pos, self.pos + n - 1)) as r:
                    data = r.read()
                break
            except Exception as e:  # noqa: BLE001
                print(f"  range read: {e!r}, retry {t + 1}", flush=True)
                time.sleep(5 * (t + 1))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def zip_members(url, dest, pattern, strip):
    """Extract the members of a remote zip matching pattern into dest, without the leading `strip` folder."""
    z, rx = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 22)), re.compile(pattern)
    for zi in z.infolist():
        if not rx.search(zi.filename) or zi.is_dir():
            continue
        out = os.path.join(dest, zi.filename[len(strip):])
        if os.path.exists(out) and os.path.getsize(out) == zi.file_size:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(zi) as src, open(out + ".tmp", "wb") as dst:
            for b in iter(lambda: src.read(1 << 22), b""):
                dst.write(b)
        os.replace(out + ".tmp", out)
        print(f"got {out}", flush=True)


def github_tree(url, dest):
    """A GitHub tarball, extracted into dest without its top folder."""
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"have {dest}", flush=True)
        return
    with _get(url) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1]:
                m.name = parts[1]
                members.append(m)
        tar.extractall(dest, members=members, filter="data")
    print(f"done {dest}", flush=True)


def get_file(url, path, tries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for t in range(tries):
        try:
            with _get(url) as r, open(path + ".tmp", "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
            os.replace(path + ".tmp", path)
            return
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                raise RuntimeError(f"{url}: {e!r}") from e
            time.sleep(5 * (t + 1))


def slakh(dest, jobs=8):
    stems = json.load(open(os.path.join(HERE, "slakh_test_stems.json")))

    def track(tid):
        d = os.path.join(dest, tid)
        get_file(f"{SLAKH_MIDI}/{tid}/metadata.yaml", os.path.join(d, "metadata.yaml"))
        for s in stems[tid]:
            get_file(f"{SLAKH_MIDI}/{tid}/MIDI/{s}.mid", os.path.join(d, "MIDI", s + ".mid"))
        get_file(f"{SLAKH_AUDIO}/{tid}/mix.flac", os.path.join(d, "mix.flac"))
        return tid

    with ThreadPoolExecutor(jobs) as ex:
        for k, tid in enumerate(ex.map(track, sorted(stems)), 1):
            print(f"slakh {k}/{len(stems)} {tid}", flush=True)


def main():
    root = sys.argv[1]
    what = sys.argv[2:] or ["bach10", "dcs", "phenicx", "rwc", "rwc_annotations", "slakh"]
    for w in what:
        if w == "bach10":
            github_tree(BACH10, os.path.join(root, "bach10"))
        elif w == "rwc_annotations":
            github_tree(RWC_ANN, os.path.join(root, "rwc_annotations"))
        elif w == "dcs":
            url, _ = zenodo_file(DCS[0], DCS[1])
            zip_members(url, os.path.join(root, "dcs"), DCS[2], "DagstuhlChoirSet_V1.2.3/")
        elif w == "phenicx":
            url, md5 = zenodo_file(*PHENICX)
            download(url, os.path.join(root, "phenicx", PHENICX[1]), md5)
        elif w == "rwc":
            for name in RWC[1]:
                url, md5 = zenodo_file(RWC[0], name)
                download(url, os.path.join(root, "rwc", name), md5)
        elif w == "slakh":
            slakh(os.path.join(root, "slakh"))
        else:
            raise SystemExit(f"unknown: {w}")


if __name__ == "__main__":
    main()
