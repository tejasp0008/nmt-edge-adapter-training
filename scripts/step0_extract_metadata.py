import sys
import os
import json
import csv
import math
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

# Constants
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "configs"

def generate_metadata():
    print(f"Loading config for {MODEL_NAME}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # 1. Write model_metadata.json
    meta = {
        "model_name": MODEL_NAME,
        "architecture": getattr(cfg, "model_type", str(cfg.__class__.__name__)),
        "d_model": getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None),
        "vocab_size": len(tok),
        "encoder_layers": getattr(cfg, "encoder_layers", None),
        "decoder_layers": getattr(cfg, "decoder_layers", None),
    }
    with open(f"{OUTPUT_DIR}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Created model_metadata.json")

    # 2. Instantiate Empty Model for Layer Profiling
    print("Instantiating empty model structure...")
    model_empty = AutoModelForSeq2SeqLM.from_config(cfg)
    params = list(model_empty.named_parameters())
    
    # 3. Write quantizable_layers_list.csv & layer_sizes_summary.csv
    csv_quant = open(f"{OUTPUT_DIR}/quantizable_layers_list.csv", "w", newline='')
    csv_sizes = open(f"{OUTPUT_DIR}/layer_sizes_summary.csv", "w", newline='')
    
    writer_q = csv.writer(csv_quant)
    writer_s = csv.writer(csv_sizes)
    
    writer_q.writerow(["module_name", "param_name", "shape", "note"])
    writer_s.writerow(["module_name", "shape", "param_count", "bytes_fp16", "bytes_int4", "note"])
    
    for name, p in params:
        shape_str = "x".join(map(str, p.shape))
        param_count = p.numel()
        
        # Check if it's a linear weight (2D) suitable for quantization
        is_linear_weight = (p.dim() == 2 and "weight" in name)
        note = "quantize_candidate" if is_linear_weight else "skip"
        
        writer_q.writerow([name, name.split('.')[-1], shape_str, note])
        
        # Size estimation
        bytes_fp16 = param_count * 2
        bytes_int4 = math.ceil(param_count / 2)
        writer_s.writerow([name, shape_str, param_count, bytes_fp16, bytes_int4, note])

    csv_quant.close()
    csv_sizes.close()
    print("✅ Created layer CSVs")

if __name__ == "__main__":
    generate_metadata()