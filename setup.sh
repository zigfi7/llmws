#!/bin/bash
# LLMWS Setup Script - Auto-detect architecture and setup environment

# Configuration
VENV_DIR="venv"
PYTHON_BIN="python3"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         LLMWS - Environment Setup                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"

# Detect architecture
ARCH=$(uname -m)
echo -e "${CYAN}[INFO] Detected architecture: ${YELLOW}${ARCH}${NC}"

# Detect CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${CYAN}[INFO] CUDA detected: ${YELLOW}${CUDA_VERSION}${NC}"
else
    CUDA_VERSION="none"
    echo -e "${YELLOW}[WARN] CUDA not detected - CPU-only mode${NC}"
fi

# 1. System Dependencies
echo -e "\n${CYAN}[1/6] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    libssl-dev \
    zlib1g-dev \
    libjpeg-dev \
    git \
    git-lfs

git lfs install 2>/dev/null

# 2. Create Virtual Environment
echo -e "${CYAN}[2/6] Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[WARN] Virtual environment exists, recreating...${NC}"
    rm -rf $VENV_DIR
fi

$PYTHON_BIN -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# 3. Upgrade pip
echo -e "${CYAN}[3/6] Upgrading pip and build tools...${NC}"
pip install --quiet --upgrade pip setuptools wheel packaging ninja

# 4. Install PyTorch
echo -e "${CYAN}[4/6] Installing PyTorch...${NC}"

if [ "$CUDA_VERSION" != "none" ]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    
    if [ "$CUDA_MAJOR" -ge "13" ]; then
        echo -e "${GREEN}[INFO] Installing PyTorch for CUDA 13+ (using CUDA 12.4 compatible build)${NC}"
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    elif [ "$CUDA_MAJOR" -eq "12" ]; then
        echo -e "${GREEN}[INFO] Installing PyTorch for CUDA 12.x${NC}"
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -eq "11" ]; then
        echo -e "${GREEN}[INFO] Installing PyTorch for CUDA 11.x${NC}"
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}[WARN] Unknown CUDA version, installing default PyTorch${NC}"
        pip install --quiet torch torchvision torchaudio
    fi
else
    echo -e "${YELLOW}[INFO] Installing PyTorch (CPU-only)${NC}"
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch
echo -e "${CYAN}[INFO] Verifying PyTorch installation...${NC}"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}');"

# 5. Install LLMWS Dependencies
echo -e "${CYAN}[5/6] Installing LLMWS dependencies...${NC}"
pip install --quiet \
    websockets \
    transformers \
    accelerate \
    safetensors \
    pillow \
    sentencepiece \
    protobuf

# 6. Install Flash Attention (if CUDA available)
if [ "$CUDA_VERSION" != "none" ]; then
    echo -e "${CYAN}[6/6] Installing Flash Attention 2...${NC}"
    
    # Check compute capability
    COMPUTE_CAP=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0))" 2>/dev/null | tr -d '(,) ')
    COMPUTE_MAJOR=$(echo $COMPUTE_CAP | awk '{print $1}')
    
    if [ "$COMPUTE_MAJOR" -ge "12" ]; then
        echo -e "${YELLOW}[INFO] Compute capability SM ${COMPUTE_MAJOR}.x (Blackwell GB10)${NC}"
        echo -e "${YELLOW}[WARN] Flash Attention binaries not yet available for SM 12.x${NC}"
        echo -e "${YELLOW}[INFO] Server will use standard attention (still fast on GB10!)${NC}"
    elif [ "$COMPUTE_MAJOR" -ge "8" ]; then
        echo -e "${GREEN}[INFO] Compute capability SM ${COMPUTE_MAJOR}.x - Flash Attention supported${NC}"
        export MAX_JOBS=8
        pip install --quiet flash-attn --no-build-isolation || \
            echo -e "${YELLOW}[WARN] Flash Attention install failed - continuing without it${NC}"
    else
        echo -e "${YELLOW}[WARN] Compute capability SM ${COMPUTE_MAJOR}.x - Flash Attention requires SM 8.0+${NC}"
    fi
else
    echo -e "${CYAN}[6/6] Skipping Flash Attention (no CUDA)${NC}"
fi

# Create directories
mkdir -p models var/sessions var/models var/logs

# Summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup Complete! 🚀                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "${CYAN}To start the server:${NC}"
echo -e "  ${YELLOW}./start.sh${NC}"
echo -e ""
echo -e "${CYAN}Or manually:${NC}"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}python llmws.py${NC}"
echo -e ""
