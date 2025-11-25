# Multilingual NMT Edge Training Pipeline

A modular and production-ready pipeline for training, analyzing, quantizing, and packaging multilingual NMT (Neural Machine Translation) models for edge-friendly deployment.  
The project provides tools for model inspection, layer-wise size estimation, quantization planning, dataset handling, workspace packaging, and reproducible evaluation.

---

## 🚀 Key Features

### 🔹 **1. Modular Architecture**
A clean, maintainable Python codebase organized into:
- **Model utilities** (loading, tokenizer checks, layer summaries)
- **Dataset and manifest tools** (JSONL creation, manifest generation)
- **Quantization planning** (layer size estimates, quantizable layers list)
- **Workspace packaging** for deployment
- **Training orchestration** with CLI support

### 🔹 **2. Rich Metadata & Analysis Tools**
The project generates a variety of useful analysis files:
- `layer_sizes_summary.csv` – per-layer FP16 & int4 estimated sizes  
- `size_estimates.json/csv` – global memory footprint estimates  
- `quantizable_layers_list.csv` – layers compatible with quantization  
- `model_metadata.json` – model configuration summary  
- `tokenizer_sanity.json` – tokenizer validation  
- `quant_test_report.json` – quantization verification  
- `sample_layers.json` – sampled layer statistics  

These metadata files are small, version-controlled, and crucial for reproducibility.

### 🔹 **3. Dataset Support**
Includes utilities for:
- JSONL dataset creation  
- Manifest building  
- Language list management  
- Support for multilingual evaluation sets such as FLORES-200

*(Large datasets are not stored in the repository; only metadata such as `dataset_overview.json` is included.)*

### 🔹 **4. Edge-Friendly Workflow**
Built to support:
- Low-precision weight formats (e.g., int4)  
- Adapter loading and packaging  
- Exporting and preparing workspaces for FPGA or low-resource deployment  
- Weight/embedding manifests and inspection tools  

---

## 📁 Directory Structure

my_project/
├─ README.md
├─ requirements.txt
│
├─ notebooks/
│  └─ training_script.ipynb
│
├─ src/
│  ├─ __main__.py
│  ├─ config.py
│  ├─ utils.py
│  │
│  ├─ data/
│  │  └─ manifest_and_io.py
│  │
│  ├─ model/
│  │  ├─ loader.py
│  │  └─ layers_summary.py
│  │
│  └─ train.py
│
├─ scripts/
│  ├─ package_workspace.py
│  └─ run_train.sh
│
├─ metadata/
│  ├─ layer_sizes_summary.csv
│  ├─ size_estimates.csv
│  ├─ size_estimates.json
│  ├─ quantizable_layers_list.csv
│  ├─ model_metadata.json
│  ├─ tokenizer_sanity.json
│  ├─ quant_test_report.json
│  ├─ sample_layers.json
│  └─ dataset_overview.json
│
└─ artifacts/
   ├─ workspace_package.zip
   ├─ logs/
   └─ datasets/


yaml
Copy code

> **Note**  
> Large artifacts such as model weights (`*.wbin`), embeddings, adapters, or large datasets are **not versioned** and belong in the `artifacts/` directory or external storage.

---

## 🔧 Installation

### **1. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
▶ Usage
Run the demo training pipeline
bash
Copy code
python -m src --demo
This executes:

Model + tokenizer loading

Layer size summary generation

Creation of sample JSONL dataset

Manifest generation

Storage of metadata in /metadata and /artifacts

Package the entire workspace
bash
Copy code
python scripts/package_workspace.py
Outputs:

bash
Copy code
artifacts/workspace_package.zip
Run from the shell script
bash
Copy code
bash scripts/run_train.sh
📦 Metadata Files Explained
File	Description
layer_sizes_summary.csv	Per-layer FP16 & estimated INT4 sizes
size_estimates.json / csv	Total model memory estimates
quantizable_layers_list.csv	Identified layers safe for quantization
model_metadata.json	General model configuration metadata
tokenizer_sanity.json	Tokenizer validation (vocab size, test samples)
quant_test_report.json	Quantization verification summary
dataset_overview.json	Summary of dataset structure (e.g., FLORES-200)
sample_layers.json	Example of random layer structure stats

These files are intentionally small and version-controlled.

📚 Dataset Usage
The project supports multilingual datasets such as:

FLORES-200

Custom parallel corpora in JSONL format

Place any large dataset in:

bash
Copy code
artifacts/datasets/
Only include small metadata files (e.g., dataset_overview.json) in the repository.

🧩 Extending the Project
You can easily add:

Adapter-based fine-tuning (LoRA, IA3, etc.)

Quantization-aware training

Export to ONNX, TensorRT, or FPGA

Evaluation scripts

Model compression workflows

The modular structure makes this straightforward.

📄 License
MIT License. You are free to use, modify, and distribute this project.

🤝 Contributing
Contributions are welcome!
Feel free to open:

Issues

Pull requests

Feature requests











