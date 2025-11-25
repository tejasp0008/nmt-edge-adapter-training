# src/data/manifest_and_io.py
from pathlib import Path
import json
from . import os as _os  # if needed
from ..utils import sha256_of_file, write_json

def write_jsonl(path, sentences):
    """Write list of strings to a newline-delimited jsonl (each line is {"text": "..."})."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for s in sentences:
            fh.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")

def build_manifest_for_dir(directory, out_manifest_path):
    directory = Path(directory)
    manifest = []
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            manifest.append({
                "path": str(p.relative_to(directory)),
                "sha256": sha256_of_file(p),
                "size_bytes": p.stat().st_size
            })
    write_json(out_manifest_path, manifest)
    return manifest

def pad_file_to_alignment(path, align=64):
    """Pad file to a multiple of `align` bytes with zero bytes."""
    p = Path(path)
    size = p.stat().st_size
    pad = (align - (size % align)) % align
    if pad:
        with p.open("ab") as fh:
            fh.write(b"\x00" * pad)
    return p.stat().st_size
