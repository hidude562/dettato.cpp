"""Import the reference Python package (kobimusic/dettato) from a checkout, without installing it.

The checkout is $DETTATO_REF (default ~/dettato-ref), pinned to the commit the GGUF and the reference dumps are made from.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REF = Path(os.environ.get("DETTATO_REF", "~/dettato-ref")).expanduser()
if str(REF / "src") not in sys.path:
    sys.path.insert(0, str(REF / "src"))


def transcriber_module():
    return importlib.import_module("dettato.transcriber.model")


def dettato():
    return importlib.import_module("dettato")


def ref_commit() -> str:
    head = (REF / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        return (REF / ".git" / head.split()[1]).read_text().strip()
    return head
