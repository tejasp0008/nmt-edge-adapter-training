#!/usr/bin/env python3
"""
unpack_and_dequantize.py
Reads .wbin files produced by the quant pipeline (EAQ1 header) and reconstructs float weights.
"""
import struct
import argparse
import pathlib
import numpy as np
import hashlib
import sys

def nibble_to_signed(x):
    return x - 16 if x >= 8 else x

def read_wbin(path: pathlib.Path):
    b = path.read_bytes()
    if len(b) < 16:
        raise ValueError("File too short")
    magic, version, out_dim, in_dim = struct.unpack_from("<4sIII", b, 0)
    if magic != b"EAQ1":
        raise ValueError(f"Unexpected magic {magic!r}")
    offset = 16
    scales_bytes = b[offset: offset + 2 * out_dim]
    if len(scales_bytes) != 2 * out_dim:
        raise ValueError("Not enough bytes for scales")
    scales = np.frombuffer(scales_bytes, dtype=np.float16).astype(np.float32)
    offset += 2 * out_dim
    packed = b[offset:]
    padded_in = ((in_dim + 1)//2) * 2
    Wq = np.zeros((out_dim, padded_in), dtype=np.int8)
    idx = 0
    for r in range(out_dim):
        for c in range(0, padded_in, 2):
            if idx >= len(packed):
                break # Padding at end of file
            byte = packed[idx]
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            Wq[r, c] = nibble_to_signed(low)
            Wq[r, c+1] = nibble_to_signed(high)
            idx += 1
    Wq = Wq[:, :in_dim]
    Wf = (Wq.astype(np.float32)) * scales.reshape(-1,1)
    return {"version": int(version), "out_dim": int(out_dim), "in_dim": int(in_dim), "scales": scales, "Wq": Wq, "Wf": Wf}

def sha256_of_array(arr: np.ndarray):
    return hashlib.sha256(arr.tobytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wbin", type=pathlib.Path)
    ap.add_argument("--print", type=int, default=0)
    ap.add_argument("--save-npy", action="store_true")
    args = ap.parse_args()
    
    info = read_wbin(args.wbin)
    print("version:", info["version"])
    print("shape:", info["out_dim"], "x", info["in_dim"])
    print("scales (first 8):", info["scales"][:8])
    if args.print:
        nr = min(args.print, info["in_dim"])
        print("Wq first row (first cols):", info["Wq"][0,:nr].tolist())
        print("Wf first row (first cols):", info["Wf"][0,:nr].tolist())
    print("Wf sha256:", sha256_of_array(info["Wf"]))
    if args.save_npy:
        outp = args.wbin.with_suffix(".npy")
        np.save(str(outp), info["Wf"])
        print("Saved Wf to", outp)

if __name__ == '__main__':
    main()