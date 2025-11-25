# src/model/loader.py
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from ..config import MODEL_NAME, TOKENIZER_NAME, DEVICE

def load_tokenizer(tokenizer_name=None):
    tokenizer_name = tokenizer_name or TOKENIZER_NAME
    print("Loading tokenizer:", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def load_model(model_name=None, use_auth_token=None):
    model_name = model_name or MODEL_NAME
    print("Loading model:", model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=cfg)
    model.to(DEVICE)
    return model

# Layer summary helper (derived from notebook code)
def summarize_layer_sizes(model, workdir):
    import csv, math
    from pathlib import Path
    Path(workdir).mkdir(parents=True, exist_ok=True)
    layer_summary_path = Path(workdir) / "layer_sizes_summary.csv"
    params = list(model.named_parameters())
    int4_total = 0
    fp16_total = 0
    with open(layer_summary_path, "w", newline='') as csvout:
        writer = csv.writer(csvout)
        writer.writerow(["module_name", "shape", "param_count", "bytes_fp16", "bytes_int4_est", "note"])
        for name, p in params:
            shape = list(p.shape)
            param_count = 1
            for d in shape:
                param_count *= d
            bytes_fp16 = param_count * 2  # FP16 = 2 bytes
            # crude int4 estimate: 0.5 byte per param
            bytes_int4_est = math.ceil(param_count * 0.5)
            fp16_total += bytes_fp16
            int4_total += bytes_int4_est
            writer.writerow([name, str(shape), param_count, bytes_fp16, bytes_int4_est, ""])
    totals = {"fp16_total_bytes": fp16_total, "int4_total_est_bytes": int4_total}
    return layer_summary_path, totals
