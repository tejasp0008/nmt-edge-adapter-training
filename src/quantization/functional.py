import torch
import numpy as np
import struct
import os
from pathlib import Path

def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def pack_int4_rowwise(Wq: torch.Tensor) -> bytes:
    """
    Pack a 2D int8 tensor with values in [-8..7] into bytes,
    two 4-bit signed values per byte, row-major.
    Wq: (out, in) torch.int8 on CPU
    Returns: bytes
    """
    assert Wq.dtype == torch.int8 or Wq.dtype == torch.int16 or Wq.dtype == torch.int32
    Wq_np = Wq.cpu().numpy().astype('int32')
    out, inp = Wq_np.shape
    # pad if odd columns
    if inp % 2 != 0:
        Wq_np = np.pad(Wq_np, ((0,0),(0,1)), 'constant', constant_values=0)
        inp += 1
        
    out_bytes = bytearray()
    for r in range(out):
        row = Wq_np[r]
        for i in range(0, inp, 2):
            low = int(row[i]) & 0xF
            high = int(row[i+1]) & 0xF
            b = (high << 4) | (low)
            out_bytes.append(b)
    return bytes(out_bytes)

def save_wbin(path: Path, Wq: torch.Tensor, scales: torch.Tensor):
    """
    Write .wbin: header(16 bytes) + scales(fp16) + packed int4 bytes, padded to 64B.
    Header: 4s I I I -> magic, version, out_dim, in_dim
    """
    out_dim, in_dim = Wq.shape
    magic = b"EAQ1"
    version = 1
    header = struct.pack("<4sIII", magic, version, out_dim, in_dim)
    # scales as fp16 little-endian
    scales_fp16 = scales.half().cpu().numpy().tobytes()
    packed = pack_int4_rowwise(Wq)
    data = header + scales_fp16 + packed
    # pad to 64B
    pad_len = (-len(data)) % 64
    if pad_len:
        data += b"\x00" * pad_len
    atomic_write_bytes(path, data)
    return path

def int4_to_signed(x):
    # convert nibble (0..15) to signed in [-8..7]
    if x >= 8:
        return x - 16
    return x

def compute_metrics(Wf: torch.Tensor, QW: torch.Tensor):
    diff = (Wf - QW).double()
    rmse = float(torch.sqrt(torch.mean(diff*diff)).item())
    maxabs = float(torch.max(torch.abs(diff)).item())
    rel_norm = float((diff.norm().item()) / (Wf.norm().item() + 1e-12))
    return {"rmse": rmse, "maxabs": maxabs, "rel_norm": rel_norm}