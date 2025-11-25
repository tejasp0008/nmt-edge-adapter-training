# src/config.py
import os
from pathlib import Path

WORKDIR = Path(os.getenv("WORKDIR", "/workspace"))
WEIGHTS_DIR = WORKDIR / "weights"
ADAPTERS_DIR = WORKDIR / "adapters"
EMBEDDINGS_DIR = WORKDIR / "embeddings"
ARTIFACTS_DIR = WORKDIR / "artifacts"
ZIPPATH = ARTIFACTS_DIR / "workspace_package.zip"

# tokenizer / model defaults (can be overridden by cli args)
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/nllb-200-distilled-600M")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", MODEL_NAME)
DEVICE = os.getenv("DEVICE", "cuda")
