# Multilingual NMT Edge Training Pipeline

A modular, production-ready pipeline for training, analyzing, quantizing, and packaging multilingual NMT (Neural Machine Translation) models for edge-friendly deployment.

---

## Quick overview
This repository provides utilities to:
- Load and inspect models and tokenizers.
- Generate layer-wise size and quantization estimates.
- Build manifests and JSONL datasets.
- Package a reproducible workspace for edge deployment.

Designed for extensibility: plug in adapters (LoRA), quantization workflows, ONNX export, or custom training loops.

---

## Key features
- Modular architecture: clear separation of model, data, and tooling.
- Layer analysis: per-layer FP16 and INT4 size estimates and summaries.
- Dataset & manifest utilities: JSONL creation and manifest generation.
- Workspace packaging: produce a deployable zip for edge environments.
- Small, version-controlled metadata (model and tokenizer checks, size estimates).

---

## Quick start

1. Create and activate a virtual environment
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the demo pipeline
```bash
python -m src --demo
```
What this does:
- Loads model & tokenizer
- Produces layer size summaries and metadata
- Creates a sample JSONL dataset and manifest
- Packages the workspace

4. Package the workspace
```bash
python scripts/package_workspace.py
# or run the helper shell script
bash scripts/run_train.sh
```
Output: artifacts/workspace_package.zip

---

## Project structure (high-level)
my_project/
- README.md
- requirements.txt
- notebooks/ (original notebooks, e.g., training_script.ipynb)
- src/
  - main.py, config.py, utils.py
  - data/manifest_and_io.py
  - model/loader.py, layers_summary.py
  - train.py
- scripts/package_workspace.py
- scripts/run_train.sh
- metadata/ (layer summaries, size estimates, etc.)
- artifacts/ (workspace zips, large datasets, weights)

---

## Metadata files (what to expect)
- layer_sizes_summary.csv — per-layer FP16 & estimated INT4 sizes  
- size_estimates.json / .csv — global memory footprint estimates  
- quantizable_layers_list.csv — layers identified as safe for quantization  
- model_metadata.json — model configuration overview  
- tokenizer_sanity.json — tokenizer checks and sample tokens  
- quant_test_report.json — quantization verification summary  
- sample_layers.json — sampled layer statistics  
- dataset_overview.json — small descriptors for large datasets (e.g., FLORES-200)

Note: Large artifacts (weights, adapters, datasets) should live in artifacts/ or external storage and are not versioned in the repo.

---

## Extending the project
Common extensions:
- Adapter-based fine-tuning (LoRA, IA3)
- Quantization-aware training and int4 workflows
- Export to ONNX / TensorRT / FPGA toolchains
- Model compression and evaluation scripts

---

## License & Contributing
MIT License — free to use and modify. Contributions welcome: open issues, PRs, and feature requests.

--- 



