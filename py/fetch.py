"""Download the dettato GGUF from Hugging Face (kobimusic/dettato-gguf) at a pinned revision and check its sha256.

    python py/fetch.py DIR [--f32]

Writes DIR/dettato-f16.gguf (or dettato-f32.gguf with --f32). A file that is already there with the right checksum is
kept. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

REPO = "kobimusic/dettato-gguf"
REVISION = "73f622a5173c8c9112d7cf2081a0402d73746fc7"
SHA256 = {
    "dettato-f16.gguf": "73f08bfcad22c401c1e584229f2b49cb8097a9b539cc81fd29dca573d7585480",
    "dettato-f32.gguf": "21dd03f39cfb52b0ced919b78566d28320f24c94ab938143ee7871397fcdaed8",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--f32", action="store_true", help="the float32 file (130 MB) instead of float16 (74 MB)")
    a = ap.parse_args()
    name = "dettato-f32.gguf" if a.f32 else "dettato-f16.gguf"
    os.makedirs(a.dir, exist_ok=True)
    out = os.path.join(a.dir, name)
    if os.path.exists(out) and sha256(out) == SHA256[name]:
        print(f"{out} is already there (sha256 ok)")
        return 0
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    tmp = out + ".part"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(tmp)
    if got != SHA256[name]:
        os.remove(tmp)
        print(f"sha256 mismatch for {name}: got {got}, expected {SHA256[name]}", file=sys.stderr)
        return 1
    os.replace(tmp, out)
    print(f"{out}: sha256 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
