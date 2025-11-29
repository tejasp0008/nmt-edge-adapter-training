import os
import json
import hashlib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def pad_to_64_bytes(p: Path):
    """Ensures file size is a multiple of 64 bytes for FPGA alignment."""
    size = p.stat().st_size
    pad = (-size) % 64
    if pad:
        with p.open("ab") as f:
            f.write(b"\x00" * pad)
        return pad
    return 0

def update_manifests():
    print("Updating manifests and padding files...")
    
    # Define what folders to scan and what manifest file to write
    scan_configs = [
        ("weights", "weights_manifest.json"),
        ("adapters", "adapter_manifest.json")
    ]

    for subfolder, manifest_name in scan_configs:
        folder_path = ARTIFACTS_DIR / subfolder
        if not folder_path.exists():
            continue
            
        entries = []
        for p in sorted(folder_path.glob("*")):
            if p.is_file() and p.suffix != ".json":
                # 1. Pad
                pad_amt = pad_to_64_bytes(p)
                if pad_amt:
                    print(f"   Padded {p.name} (+{pad_amt} bytes)")
                
                # 2. Hash & Record
                entries.append({
                    "file": str(p.relative_to(ARTIFACTS_DIR)),
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256_file(p)
                })
        
        # 3. Write Manifest
        manifest_path = ARTIFACTS_DIR / manifest_name
        data = {subfolder: entries}
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Updated {manifest_name} ({len(entries)} items)")

if __name__ == "__main__":
    update_manifests()