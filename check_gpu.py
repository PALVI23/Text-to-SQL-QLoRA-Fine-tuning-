"""Run this first on your own computer:  python check_gpu.py"""
import platform, shutil, subprocess, sys

print("Python:", sys.version.split()[0], "| OS:", platform.system(), platform.release())
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not found -> no NVIDIA GPU or driver installed.")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1e9
        print(f"PyTorch sees: {p.name}, {gb:.1f} GB VRAM, CUDA {torch.version.cuda}")
        print("VERDICT:", "OK for this project" if gb >= 7.5 else "Too little VRAM (need ~8 GB); use Kaggle")
    else:
        print("PyTorch is installed but cannot use a GPU (CPU-only build, or no NVIDIA GPU).")
        print("Apple Silicon (M1-M4) Macs: not supported for this project (needs CUDA for 4-bit QLoRA); use Kaggle.")
except ImportError:
    print("PyTorch not installed yet. If nvidia-smi showed a GPU, install the CUDA build from https://pytorch.org/get-started/locally/")
