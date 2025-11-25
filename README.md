# Multilingual NMT Edge Training Pipeline

This repository contains a production-ready, modular Python project converted from the original notebook  
`training_script.ipynb` (located in `notebooks/`).  
The project implements utilities for multilingual NMT workflows including model loading, adapter utilities,
manifest generation, quantization-ready layer summaries, and workspace packaging for deployment.

---

## 🚀 Features

### ✔ Modular Codebase
All notebook logic is refactored into a clean directory structure:

- `src/model` – model & tokenizer loaders, layer size summaries  
- `src/data` – JSONL writing, manifest generation, padding utilities  
- `src/train.py` – demo training pipeline & workspace preparation  
- `scripts/package_workspace.py` – package entire workspace into a zip  
- `scripts/run_train.sh` – runnable example script  

### ✔ NMT-Friendly Utilities
- Generates manifests for datasets or artifacts  
- Works with HuggingFace Transformers  
- Provides layer-level parameter & size summaries (FP16 & int4 estimates)  
- Supports workspace packaging for deployment on edge devices  

### ✔ Ready for Extension
You can plug in:
- Custom training loops  
- Quantization methods  
- Adapters / LoRA layers  
- Efficient multilingual translation pipelines  

---

## 📁 Project Structure

my_project/
├─ README.md
├─ requirements.txt
├─ notebooks/
│ └─ training_script.ipynb
├─ scripts/
│ ├─ package_workspace.py
│ └─ run_train.sh
├─ src/
│ ├─ main.py
│ ├─ config.py
│ ├─ utils.py
│ ├─ data/
│ │ └─ manifest_and_io.py
│ ├─ model/
│ │ ├─ loader.py
│ │ └─ layers_summary.py
│ └─ train.py
└─ artifacts/
└─ (generated files)

yaml
Copy code

---

## 🛠 Installation

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
▶ Running the Project
Demo Training Run
bash
Copy code
python -m src --demo
This will:

Load the model & tokenizer

Generate layer size summaries

Create sample dataset

Build a manifest

Package the Workspace
bash
Copy code
python scripts/package_workspace.py
Produces:

bash
Copy code
artifacts/workspace_package.zip
🧩 Scripts
run_train.sh
Example shell script for running the demo or extending into full training.

package_workspace.py
Creates reproducible workspace packages for deployment or sharing.

📦 Artifacts
Generated files (manifests, layer summaries, packaged zips, demo files) are stored inside:

Copy code
artifacts/
This keeps the repository clean and ensures reproducibility.

🔧 Customization
You can extend this template to:

Add real training datasets

Implement LoRA / Adapters

Add ONNX or int4 quantization pipelines

Integrate evaluation scripts

Deploy to edge devices

📜 License
MIT License. Feel free to use or modify this repository for research or production.

