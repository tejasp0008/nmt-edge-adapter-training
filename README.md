# EA-MNMT: Edge-Adaptive Multilingual NMT on FPGA

![Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.8%7C3.9%7C3.10-green)
![Hardware](https://img.shields.io/badge/Target-Xilinx_Alveo_FPGA-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview
This repository implements **Edge-Adaptive Multilingual Neural Machine Translation (EA-MNMT)**, a system designed to bridge the "Quality Cliff" when deploying massive multilingual models (like NLLB-200) on resource-constrained edge devices (FPGAs).

Standard 4-bit quantization destroys translation quality for low-resource languages (e.g., Hindi, Galician). This project solves this by combining:
1.  **INT4 Backbone:** Quantizing the 600M parameter model to fit in FPGA HBM.
2.  **LoftQ Initialization:** Mathematically absorbing quantization error into adapters.
3.  **Hardware-Aware Packing:** Custom binary formats (`.wbin`) aligned for FPGA DMA transfer.

---

##  Key Features

### 1. FPGA-Ready Quantization Pipeline
- Implements symmetric per-channel **INT4 quantization** ($Q \in [-8, 7]$).
- Packs weights into a custom **EAQ1 binary format** (row-major, 2-nibbles-per-byte) designed for high-bandwidth systolic arrays.

### 2. LoftQ & Dynamic Adaptation
- Includes logic to compute quantization residuals ($W_{fp16} - W_{int4}$).
- Initializes LoRA adapters via SVD to recover accuracy lost during quantization.
- Supports dynamic swapping of adapters for zero-overhead language switching.

### 3. Modular Engineering Structure
- Separates core mathematical logic (`src/quantization`) from execution scripts (`scripts/`).
- separating "Lab Code" (Notebooks) from "Production Code" (Python modules).

### 4. Split-Embedding Architecture
- Tools to partition vocabulary into a "Universal Core" (URAM) and "Language Shards" (DRAM) to reduce on-chip memory usage by ~40%.

---

## 📁 Project Structure

```text
ea-mnmt-fpga/
├── artifacts/                  # Generated binaries (ignored by Git)
│   ├── weights/                # Packed .wbin files for FPGA HBM
│   ├── adapters/               # Trained .abin files for FPGA URAM
│   └── *_manifest.json         # Lookup tables for hardware loader
├── configs/                    # Hardware profiles & Layer maps
│   ├── model_metadata.json
│   └── quantizable_layers_list.csv
├── docs/
│   └── fpga_export_plan.md     # Detailed memory map specification
├── notebooks/
│   └── training_script_fpga.ipynb  # Original research experiments
├── scripts/                    # CLI Tools
│   ├── step0_extract_metadata.py
│   ├── step0_prepare_dataset.py
│   ├── run_synthetic_test.py
│   └── utils_manifest.py
├── src/                        # Source Code
│   ├── quantization/           # Math: Metric calc, bit-packing
│   └── utils/                  # Tools: Unpackers, Verifiers
├── .gitignore
├── requirements.txt
└── README.md
🛠️ Installation
Clone the repository:

Bash

git clone [https://github.com/YOUR_USERNAME/ea-mnmt-fpga.git](https://github.com/YOUR_USERNAME/ea-mnmt-fpga.git)
cd ea-mnmt-fpga
Install dependencies:

Bash

pip install -r requirements.txt
▶️ Usage Workflow
Phase 1: Quantization & Metadata
Extract model architecture and generate the "blueprint" for the FPGA.

Bash

# 1. Profile the NLLB model layers
python scripts/step0_extract_metadata.py

# 2. (Optional) Run synthetic test to verify math
python scripts/run_synthetic_test.py
Phase 2: Data & Training
Prepare FLORES-200 data and run LoftQ adapter training.

Bash

# 1. Convert FLORES dataset to JSONL
python scripts/step0_prepare_dataset.py

# 2. Run Training (Future implementation)
# python scripts/step2_train_adapters.py
Phase 3: Hardware Export
Package the weights and verify the binaries.

Bash

# 1. Generate Manifests and Pad files to 64-byte alignment
python scripts/utils_manifest.py

# 2. Verify a specific binary file (sanity check)
python -m src.utils.unpacker artifacts/weights/demo_layer.wbin --print 10
📉 Artifact Format
Weight Binary (.wbin)
The FPGA expects weights in the following Little-Endian format:

Header (16B): Magic (EAQ1), Version, Out_Dim, In_Dim.

Scales: Array of FP16 scalars (one per output channel).

Payload: Packed INT4 data (row-major).

See docs/fpga_export_plan.md for the full register map and memory layout.

🤝 Contribution
This project is part of a research initiative on efficient Edge AI.

Phase 1 (Current): Quantization & Export Logic.

Phase 2: GPU Fine-tuning (LoftQ).

Phase 3: Verilog/HLS Implementation.

📜 License
MIT License.
