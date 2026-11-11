# Running Project 3 on your own computer (VS Code)

## 1. Do you have a usable GPU?
You need an **NVIDIA GPU with at least 8 GB of VRAM** (for example an RTX 3060 12GB, 3070, 4060 Ti 16GB, 4070 or better).
Laptop GPUs with 4–6 GB (an RTX 3050 or 4050) are too small. Integrated Intel or AMD graphics and Apple M-series Macs can't run 4-bit QLoRA.

**Quick check on Windows:** press Ctrl+Shift+Esc → **Performance** tab → look for "GPU 0/1" with **NVIDIA** in the name, and note its *Dedicated GPU memory*.
**Or in a terminal** (VS Code: *Terminal → New Terminal*): `nvidia-smi`. It shows the GPU name and memory, e.g. `8188MiB`.

## 2. Setup (about 20 min)
```bash
cd path/to/text2sql
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt ipykernel
python check_gpu.py
```
`check_gpu.py` should print `VERDICT: OK for this project`.

## 3. Run
Open the `text2sql` folder in VS Code → open `notebooks/KAGGLE_full_pipeline.ipynb` (it works locally too) →
*Select Kernel* → choose `.venv` → **Run All**. Spider downloads into `text2sql/data/` automatically.
Keep the laptop plugged in and stop it from sleeping. Expect about 4–6 h on an 8–12 GB GPU.
