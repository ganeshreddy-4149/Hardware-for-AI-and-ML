"""
nn_forward_gpu.py
Forward pass of a small neural network on the GPU.
Architecture: Linear(4->5) -> ReLU -> Linear(5->1)
Batch size: 16
ECE 410/510 Codefest 3 - COPT - Spring 2026
"""

import torch
import torch.nn as nn
import sys

print("=" * 50)
print("  Neural Network Forward Pass - GPU")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    print("\nNo CUDA GPU found.")
    print("Please enable GPU: Runtime -> Change runtime type -> T4 GPU")
    sys.exit(1)

print(f"\n[1] CUDA GPU detected!")
print(f"    Device name : {torch.cuda.get_device_name(0)}")
print(f"    Device index: {torch.cuda.current_device()}")
print(f"    Using device: {device}")

model = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)
model = model.to(device)

print(f"\n[2] Network architecture:")
print(f"    {model}")
print(f"    Model device: {next(model.parameters()).device}")

torch.manual_seed(42)
x = torch.randn(16, 4).to(device)

print(f"\n[3] Input tensor:")
print(f"    Shape  : {list(x.shape)}")
print(f"    Device : {x.device}")
print(f"    Dtype  : {x.dtype}")

with torch.no_grad():
    output = model(x)

print(f"\n[4] Forward pass complete!")
print(f"    Output shape  : {list(output.shape)}")
print(f"    Output device : {output.device}")
print(f"    Output dtype  : {output.dtype}")

print(f"\n[5] Output tensor (first 5 values):")
for i in range(5):
    print(f"    batch[{i}] = {output[i].item():.6f}")

print(f"\n{'=' * 50}")
print(f"  SUMMARY")
print(f"{'=' * 50}")
print(f"  GPU            : {torch.cuda.get_device_name(0)}")
print(f"  Network        : Linear(4->5) -> ReLU -> Linear(5->1)")
print(f"  Batch size     : 16")
print(f"  Input shape    : {list(x.shape)}")
print(f"  Output shape   : {list(output.shape)}")
print(f"  Output device  : {output.device}")
print(f"  GPU confirmed  : {output.device.type == 'cuda'}")
print(f"{'=' * 50}")
