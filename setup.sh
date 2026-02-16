#!/bin/bash
# LLMWS Setup Script - Multi-platform (x86_64/aarch64, CUDA 11/12/13, CPU)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
STARTUP_CONFIG_FILE="$SCRIPT_DIR/startup.conf"

# Defaults (can be overridden in environment or startup.conf)
PYTHON_MIN="${PYTHON_MIN:-3.10}"

if [ -f "$STARTUP_CONFIG_FILE" ]; then
    # shellcheck source=/dev/null
    source "$STARTUP_CONFIG_FILE"
fi

# Colors
if [ -t 1 ]; then
    G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
else
    G=''; C=''; Y=''; R=''; N=''
fi

echo -e "${C}LLMWS Setup${N}"

# ── Detect platform ─────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)
CUDA_VER="none"

# Prefer nvidia-smi (reports driver-supported CUDA, more accurate for PyTorch)
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
fi

# Fallback to nvcc (compile-time toolkit version)
if [ -z "$CUDA_VER" ] || [ "$CUDA_VER" = "none" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
    elif [ -f /usr/local/cuda/bin/nvcc ]; then
        CUDA_VER=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
    fi
fi

if [ -z "$CUDA_VER" ]; then
    CUDA_VER="none"
fi

CUDA_MAJOR="${CUDA_VER%%.*}"

echo -e "${C}  OS: ${OS}  Arch: ${ARCH}  CUDA: ${CUDA_VER}${N}"

# ── System dependencies (Linux only) ────────────────────────────────
if [ "$OS" = "Linux" ] && command -v apt-get &>/dev/null; then
    echo -e "${C}  Installing system deps${N}"
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -qq python3-venv python3-dev build-essential \
        ninja-build libssl-dev git git-lfs 2>/dev/null || true
    git lfs install 2>/dev/null || true
fi

# ── Python version check ────────────────────────────────────────────
PYTHON_BIN="python3"
PY_VER=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)

echo -e "${C}  Python: ${PY_VER}${N}"

if [ -z "$PY_VER" ]; then
    echo -e "${R}  ERROR: python3 not found or not runnable${N}"
    exit 1
fi

if [ "$(printf '%s\n' "$PYTHON_MIN" "$PY_VER" | sort -V | head -n 1)" != "$PYTHON_MIN" ]; then
    echo -e "${R}  ERROR: Python >= ${PYTHON_MIN} required (found ${PY_VER})${N}"
    exit 1
fi

# ── Create venv ──────────────────────────────────────────────────────
echo -e "${C}  Creating venv${N}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${Y}  Removing existing venv${N}"
    rm -rf "$VENV_DIR"
fi

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install -q --upgrade pip setuptools wheel

# ── Install PyTorch ──────────────────────────────────────────────────
echo -e "${C}  Installing PyTorch${N}"

if [ "$CUDA_VER" != "none" ]; then
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        echo -e "${G}  PyTorch for CUDA 13.x (${ARCH})${N}"
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        CUDA_MINOR="${CUDA_VER##*.}"
        if [ "$CUDA_MINOR" -ge 6 ]; then
            echo -e "${G}  PyTorch for CUDA 12.6+${N}"
            pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
        elif [ "$CUDA_MINOR" -ge 4 ]; then
            echo -e "${G}  PyTorch for CUDA 12.4+${N}"
            pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        else
            echo -e "${G}  PyTorch for CUDA 12.1+${N}"
            pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        fi
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        echo -e "${G}  PyTorch for CUDA 11.x${N}"
        pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${Y}  Unknown CUDA ${CUDA_VER} - installing default PyTorch${N}"
        pip install -q torch torchvision torchaudio
    fi
else
    echo -e "${Y}  No CUDA - CPU-only PyTorch${N}"
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ── Verify PyTorch ───────────────────────────────────────────────────
python3 -W ignore::UserWarning -c "
import torch, sys
print(f'  PyTorch {torch.__version__} Python {sys.version.split()[0]}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'  GPU: {torch.cuda.get_device_name(0)} SM {cap[0]}.{cap[1]}')
else:
    print('  CPU mode')
"

# ── Install dependencies ─────────────────────────────────────────────
echo -e "${C}  Installing dependencies${N}"
pip install -q websockets transformers accelerate safetensors \
    pillow sentencepiece protobuf

# ── Flash Attention (optional, skip on Blackwell - use SDPA instead) ─
if [ "$CUDA_VER" != "none" ]; then
    COMPUTE_MAJOR=$(python3 -W ignore::UserWarning -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_capability(0)[0])
else:
    print(0)
" 2>/dev/null)

    if [ "$COMPUTE_MAJOR" -ge 12 ]; then
        echo -e "${G}  Blackwell GPU - using SDPA (cuDNN backend, faster than flash-attn)${N}"
    elif [ "$COMPUTE_MAJOR" -ge 8 ]; then
        echo -e "${C}  Installing Flash Attention 2${N}"
        MAX_JOBS=4 pip install -q flash-attn --no-build-isolation 2>/dev/null || \
            echo -e "${Y}  Flash Attention install failed - SDPA will be used as fallback${N}"
    fi
fi

# ── Create directories ───────────────────────────────────────────────
mkdir -p models var/sessions var/models var/logs

echo -e "${G}  Setup complete${N}"
echo -e "${C}  Run: ./start.sh${N}"
