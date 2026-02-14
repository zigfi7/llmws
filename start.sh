#!/bin/bash
# LLMWS Auto-Start Script
# Automatically runs setup if venv doesn't exist, then starts server

# Configuration
VENV_DIR="venv"
SERVER_SCRIPT="llmws.py"

# CRITICAL: Force JIT compilation for unsupported SM architectures (Blackwell GB10)
# This allows PyTorch to work on SM 12.1 even though it doesn't have pre-compiled kernels
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              LLMWS - Auto Start                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo -e ""

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[!] Virtual environment not found${NC}"
    echo -e "${CYAN}[*] Running setup...${NC}"
    echo -e ""
    
    # Check if setup.sh exists
    if [ ! -f "setup.sh" ]; then
        echo -e "${RED}[ERROR] setup.sh not found!${NC}"
        exit 1
    fi
    
    # Make setup.sh executable and run it
    chmod +x setup.sh
    ./setup.sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Setup failed!${NC}"
        exit 1
    fi
    
    echo -e ""
    echo -e "${GREEN}[✓] Setup completed${NC}"
    echo -e ""
fi

# Check if venv activation script exists
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${RED}[ERROR] Virtual environment is corrupted${NC}"
    echo -e "${YELLOW}[!] Please delete '$VENV_DIR' and run this script again${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${CYAN}[*] Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate

# Check if server script exists
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo -e "${RED}[ERROR] Server script '$SERVER_SCRIPT' not found!${NC}"
    exit 1
fi

# Check Python and dependencies
echo -e "${CYAN}[*] Checking dependencies...${NC}"
python3 -c "import torch, transformers, websockets, safetensors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[WARN] Some dependencies missing${NC}"
    echo -e "${CYAN}[*] Running setup to install missing packages...${NC}"
    ./setup.sh
fi

# Display system info
echo -e ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}System Information:${NC}"
python3 -c "
import torch
import sys
import warnings

# Suppress CUDA compatibility warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*is not compatible.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  Compute: SM {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  VRAM: {mem_gb:.1f} GB')
"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e ""

# Start server
echo -e "${GREEN}[✓] Starting LLMWS server...${NC}"
echo -e ""
python3 $SERVER_SCRIPT

# Deactivate on exit
deactivate
