#!/usr/bin/env bash
# Setup script for fine-tuning on WSL2 with RTX 4090
# Run this on your Windows PC inside WSL2
set -euo pipefail

echo "=== Executive Helper: Training Setup (WSL2 + RTX 4090) ==="
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────

echo "Checking CUDA..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers for WSL2:"
    echo "  https://developer.nvidia.com/cuda/wsl"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
python3 --version

# ── Install system build deps (needed for llama.cpp / GGUF export) ───────────

echo "Installing system build dependencies..."
sudo apt-get update -y
sudo apt-get install -y cmake build-essential git curl libcurl4-openssl-dev python3-dev

# ── Clone/sync project ──────────────────────────────────────────────────────

if [ ! -d ".git" ]; then
    echo ""
    echo "NOTE: Run this script from inside the executive-helper project directory."
    echo "Clone the repo first: git clone <repo-url> executive-helper && cd executive-helper"
    exit 1
fi

# ── Create venv ──────────────────────────────────────────────────────────────

echo ""
echo "Creating virtual environment..."
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip

# ── Install Unsloth + dependencies ───────────────────────────────────────────

echo ""
echo "Installing Unsloth (this may take a few minutes)..."
pip install --no-deps "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets tokenizers sentencepiece protobuf pyyaml

# ── Verify CUDA works ────────────────────────────────────────────────────────

echo ""
echo "Verifying PyTorch + CUDA..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To fine-tune:"
echo "  source .venv-train/bin/activate"
echo "  python -m src.finetune.train --config configs/finetune_ef.yaml"
echo ""
echo "Or use make:"
echo "  make finetune-ef"
