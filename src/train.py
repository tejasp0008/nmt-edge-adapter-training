# src/train.py
import os
import json
from pathlib import Path
from transformers import TrainingArguments, Trainer
from .model.loader import load_model, load_tokenizer, summarize_layer_sizes
from .data.manifest_and_io import write_jsonl, build_manifest_for_dir
from .config import WORKDIR, ARTIFACTS_DIR

def prepare_demo_workspace():
    """If weights/adapters missing, create small demo files to allow packaging (from notebook)."""
    wdir = Path(WORKDIR)
    wdir.mkdir(parents=True, exist_ok=True)
    demo_wbin = wdir / "weights" / "demo.wbin"
    demo_wbin.parent.mkdir(parents=True, exist_ok=True)
    if not demo_wbin.exists():
        demo_wbin.write_bytes(b"DEMO" * 256)
    return demo_wbin

def run_quick_demo_training():
    """A light-weight routine adapted from the notebook to illustrate flow."""
    tokenizer = load_tokenizer()
    model = load_model()
    # Summarize layer sizes (writes CSV)
    summary_path, totals = summarize_layer_sizes(model, WORKDIR)
    print("Layer summary written to:", summary_path)
    print("Totals:", totals)
    # create a tiny synthetic dataset and manifest for demo
    sentences = ["Hello world", "This is a demo", "Translation test"]
    demo_jsonl = Path(WORKDIR) / "demo.jsonl"
    write_jsonl(demo_jsonl, sentences)
    manifest = build_manifest_for_dir(WORKDIR, Path(ARTIFACTS_DIR) / "manifest.json")
    print("Manifest entries:", len(manifest))

if __name__ == "__main__":
    run_quick_demo_training()
