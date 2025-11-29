import os
import json
import statistics
from pathlib import Path

# Config
# Note: You must manually point this to where you extracted the downloaded flores dataset
RAW_DATA_PATH = Path("step2_flores/flores200_dataset") 
OUTPUT_DIR = Path("src/data")
TARGET_LANGS = ["eng_Latn", "rus_Cyrl", "hin_Deva", "deu_Latn", "spa_Latn"]

def read_lang_file(path):
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def prepare_dataset():
    if not RAW_DATA_PATH.exists():
        print(f"❌ Error: Raw data path {RAW_DATA_PATH} not found.")
        print("   Please extract FLORES dataset there first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stats = {}

    for lang in TARGET_LANGS:
        print(f"Processing {lang}...")
        for split in ["dev", "devtest"]:
            # Construct path (Adjust based on actual extracted folder structure)
            # Standard FLORES structure: dev/<lang>.dev
            file_path = RAW_DATA_PATH / split / f"{lang}.{split}"
            
            sentences = read_lang_file(file_path)
            if not sentences: continue

            # Write JSONL
            out_file = OUTPUT_DIR / f"{lang}_{split}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for s in sentences:
                    f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
            
            # Stats
            if lang not in dataset_stats: dataset_stats[lang] = {}
            dataset_stats[lang][split] = len(sentences)

    # Save stats
    with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print("✅ Dataset preparation complete.")

if __name__ == "__main__":
    prepare_dataset()