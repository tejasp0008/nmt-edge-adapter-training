import sys
import os
import torch
import numpy as np

# Add the 'src' folder to the path so we can import 'functional'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantization.functional import compute_metrics

# Constants from Notebook Cell 1
NBITS = 4
QMAX = 2**(NBITS-1) - 1  # 7
QMIN = -2**(NBITS-1)     # -8
LOFTQ_RIDGE = 1e-4

def synthetic_run(out=128, inp=256, seed=42):
    print(f"Running synthetic debug quantize+LoftQ test (Shape {out}x{inp})...")
    torch.manual_seed(seed)
    
    # Generate random weights similar to a real model
    Wf = torch.randn(out, inp) * 0.02  
    
    # 1. Quantization (Standard)
    max_abs = Wf.abs().amax(dim=1)
    scale = max_abs / QMAX
    scale[scale==0]=1e-8
    Wq = torch.round(Wf / scale.unsqueeze(1)).clamp(QMIN, QMAX).to(torch.int8)
    QW = Wq.float() * scale.unsqueeze(1)
    
    metrics_before = compute_metrics(Wf, QW)
    
    # 2. LoftQ Logic (SVD Correction)
    r = min(16, out, inp)
    # SVD on Residual
    U,S,Vt = torch.linalg.svd(Wf - QW, full_matrices=False)
    A = Vt[:r,:]
    AAT = A @ A.T
    inv = torch.linalg.inv(AAT + LOFTQ_RIDGE * torch.eye(AAT.shape[0]))
    B = ((Wf - QW) @ A.T) @ inv
    
    # Corrected reconstruction
    corrected = QW + (B @ A)
    metrics_after = compute_metrics(Wf, corrected)
    
    print("metrics before (INT4 Only):", metrics_before)
    print("metrics after (INT4 + LoftQ):", metrics_after)
    return True

if __name__ == "__main__":
    synthetic_run(128, 256)