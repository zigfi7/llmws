#!/bin/bash
# LLMWS Start Script
# Linux-focused startup with configurable environment backend.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SERVER_SCRIPT="llmws.py"
VENV_DIR="venv"
MICROMAMBA_DIR="$SCRIPT_DIR/micromamba"
MICROMAMBA_BIN="$MICROMAMBA_DIR/bin/micromamba"
STARTUP_CONFIG_FILE="$SCRIPT_DIR/startup.conf"

# Defaults (can be overridden in environment or startup.conf)
ENV_BACKEND="${ENV_BACKEND:-auto}"                    # auto | venv | mamba
PYTHON_MIN="${PYTHON_MIN:-3.10}"
MAMBA_ENV="${MAMBA_ENV:-llmws}"
MAMBA_PYTHON_DEFAULT="${MAMBA_PYTHON_DEFAULT:-3.11}"
MAMBA_PYTHON_CUDA13_PLUS="${MAMBA_PYTHON_CUDA13_PLUS:-3.12}"
MAMBA_PYTHON_X86_64="${MAMBA_PYTHON_X86_64:-}"
MAMBA_PYTHON_AARCH64="${MAMBA_PYTHON_AARCH64:-}"
MAMBA_PYTHON="${MAMBA_PYTHON:-}"

# Colors
if [ -t 1 ]; then
    G='\033[0;32m'; C='\033[0;36m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
else
    G=''; C=''; Y=''; R=''; N=''
fi

BACKEND_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backend)
            BACKEND_ARG="${2:-}"
            shift 2
            ;;
        --backend=*)
            BACKEND_ARG="${1#*=}"
            shift
            ;;
        --mamba)
            BACKEND_ARG="mamba"
            shift
            ;;
        --venv)
            BACKEND_ARG="venv"
            shift
            ;;
        --auto)
            BACKEND_ARG="auto"
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: ./start.sh [--backend auto|venv|mamba] [--auto|--venv|--mamba]

Environment backend priority:
1) --backend argument
2) startup.conf (ENV_BACKEND)
3) default: auto
EOF
            exit 0
            ;;
        *)
            echo -e "${R}Unknown argument: $1${N}"
            exit 1
            ;;
    esac
done

if [ -f "$STARTUP_CONFIG_FILE" ]; then
    # shellcheck source=/dev/null
    source "$STARTUP_CONFIG_FILE"
fi

if [ -n "$BACKEND_ARG" ]; then
    ENV_BACKEND="$BACKEND_ARG"
fi

detect_cuda_version() {
    local ver=""

    # Prefer nvidia-smi (reports driver-supported CUDA, more accurate for PyTorch)
    if command -v nvidia-smi &>/dev/null; then
        ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
    fi

    # Fallback to nvcc (compile-time toolkit version)
    if [ -z "$ver" ]; then
        local nvcc_out=""
        if command -v nvcc &>/dev/null; then
            nvcc_out="$(nvcc --version 2>/dev/null || true)"
        elif [ -x /usr/local/cuda/bin/nvcc ]; then
            nvcc_out="$(/usr/local/cuda/bin/nvcc --version 2>/dev/null || true)"
        fi
        if [ -n "$nvcc_out" ]; then
            ver="$(printf '%s\n' "$nvcc_out" | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
        fi
    fi

    if [ -z "$ver" ]; then
        echo "none"
    else
        echo "$ver"
    fi
}

version_ge() {
    # Returns 0 if $1 >= $2
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

system_python_version() {
    if ! command -v python3 &>/dev/null; then
        echo ""
        return
    fi
    python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo ""
}

get_torch_index() {
    if [ "$CUDA_VER" = "none" ]; then
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi

    if [ "$CUDA_MAJOR" -ge 13 ]; then
        echo "https://download.pytorch.org/whl/cu130"
        return
    fi

    if [ "$CUDA_MAJOR" -eq 12 ]; then
        local minor="${CUDA_VER##*.}"
        if [ "$minor" -ge 6 ] 2>/dev/null; then
            echo "https://download.pytorch.org/whl/cu126"
        elif [ "$minor" -ge 4 ] 2>/dev/null; then
            echo "https://download.pytorch.org/whl/cu124"
        else
            echo "https://download.pytorch.org/whl/cu121"
        fi
        return
    fi

    echo "https://download.pytorch.org/whl/cu118"
}

ensure_python_deps() {
    if python3 -c "import torch, transformers, websockets, safetensors" 2>/dev/null; then
        return
    fi

    echo -e "${Y}  Missing dependencies - installing${N}"
    local torch_index
    torch_index="$(get_torch_index)"

    pip install -q torch torchvision torchaudio --index-url "$torch_index"
    pip install -q websockets transformers accelerate safetensors sentencepiece protobuf pillow

    # Install flash-attn for GPUs that support it (SM 8.x-9.x, i.e. Ampere/Hopper)
    # Skip on Blackwell (SM 12.x+) which uses SDPA with cuDNN instead
    if [ "$CUDA_VER" != "none" ]; then
        local compute_major
        compute_major="$(python3 -W ignore::UserWarning -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_capability(0)[0])
else:
    print(0)
" 2>/dev/null)"
        if [ "${compute_major:-0}" -ge 8 ] && [ "${compute_major:-0}" -lt 12 ]; then
            echo -e "${C}  Installing Flash Attention 2 (SM ${compute_major}.x)${N}"
            MAX_JOBS=4 pip install -q flash-attn --no-build-isolation 2>/dev/null || true
        fi
    fi
}

activate_venv() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        return 1
    fi
    echo -e "${C}  Activating venv${N}"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
}

ensure_venv() {
    if activate_venv; then
        return 0
    fi

    if [ ! -x "$SCRIPT_DIR/setup.sh" ]; then
        chmod +x "$SCRIPT_DIR/setup.sh" 2>/dev/null || true
    fi

    if [ ! -f "$SCRIPT_DIR/setup.sh" ]; then
        echo -e "${R}  ERROR: setup.sh not found${N}"
        return 1
    fi

    echo -e "${C}  Creating venv via setup.sh${N}"
    "$SCRIPT_DIR/setup.sh"
    activate_venv
}

ensure_micromamba() {
    if [ -x "$MICROMAMBA_BIN" ]; then
        return 0
    fi

    if [ "$OS" != "Linux" ]; then
        echo -e "${R}  ERROR: micromamba bootstrap is only configured for Linux${N}"
        return 1
    fi

    local mamba_arch
    case "$ARCH" in
        x86_64|amd64) mamba_arch="linux-64" ;;
        aarch64|arm64) mamba_arch="linux-aarch64" ;;
        *)
            echo -e "${R}  ERROR: Unsupported arch for micromamba: $ARCH${N}"
            return 1
            ;;
    esac

    echo -e "${C}  Downloading micromamba (${mamba_arch})${N}"
    mkdir -p "$MICROMAMBA_DIR"
    curl -Ls "https://micro.mamba.pm/api/micromamba/${mamba_arch}/latest" | tar -xvj -C "$MICROMAMBA_DIR"
}

activate_mamba_shell() {
    export MAMBA_ROOT_PREFIX="$MICROMAMBA_DIR/root"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    eval "$("$MICROMAMBA_BIN" shell hook --shell bash)"
    export PATH="$MICROMAMBA_DIR/bin:$PATH"
}

mamba_env_exists() {
    micromamba env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$MAMBA_ENV"
}

resolve_mamba_python_version() {
    if [ -n "$MAMBA_PYTHON" ]; then
        echo "$MAMBA_PYTHON"
        return
    fi

    case "$ARCH" in
        x86_64|amd64)
            if [ -n "$MAMBA_PYTHON_X86_64" ]; then
                echo "$MAMBA_PYTHON_X86_64"
                return
            fi
            ;;
        aarch64|arm64)
            if [ -n "$MAMBA_PYTHON_AARCH64" ]; then
                echo "$MAMBA_PYTHON_AARCH64"
                return
            fi
            ;;
    esac

    if [ "$CUDA_MAJOR" -ge 13 ]; then
        echo "$MAMBA_PYTHON_CUDA13_PLUS"
    else
        echo "$MAMBA_PYTHON_DEFAULT"
    fi
}

ensure_mamba_env() {
    ensure_micromamba
    activate_mamba_shell

    local target_python
    target_python="$(resolve_mamba_python_version)"
    echo -e "${C}  Using micromamba env=${MAMBA_ENV} python=${target_python}${N}"

    if ! mamba_env_exists; then
        echo -e "${C}  Creating micromamba environment${N}"
        micromamba create -y -n "$MAMBA_ENV" "python=${target_python}"
    fi

    micromamba activate "$MAMBA_ENV"
}

ARCH="$(uname -m)"
OS="$(uname -s)"
CUDA_VER="$(detect_cuda_version)"
if [ "$CUDA_VER" = "none" ]; then
    CUDA_MAJOR=0
else
    CUDA_MAJOR="${CUDA_VER%%.*}"
fi

echo -e "${C}LLMWS - Starting${N}"
echo -e "${C}  OS: ${OS}  Arch: ${ARCH}  CUDA: ${CUDA_VER}${N}"
echo -e "${C}  Backend mode: ${ENV_BACKEND}${N}"

ACTIVE_BACKEND=""
case "$ENV_BACKEND" in
    auto)
        if activate_venv; then
            ACTIVE_BACKEND="venv"
        fi

        if [ -z "$ACTIVE_BACKEND" ] && [ -x "$MICROMAMBA_BIN" ]; then
            activate_mamba_shell
            if mamba_env_exists; then
                echo -e "${C}  Activating existing micromamba env${N}"
                micromamba activate "$MAMBA_ENV"
                ACTIVE_BACKEND="mamba"
            fi
        fi

        if [ -z "$ACTIVE_BACKEND" ]; then
            SYS_PY="$(system_python_version)"
            if [ -n "$SYS_PY" ] && version_ge "$SYS_PY" "$PYTHON_MIN"; then
                if ensure_venv; then
                    ACTIVE_BACKEND="venv"
                fi
            else
                echo -e "${Y}  System python too old/missing (${SYS_PY:-none}); using mamba${N}"
            fi
        fi

        if [ -z "$ACTIVE_BACKEND" ]; then
            ensure_mamba_env
            ACTIVE_BACKEND="mamba"
        fi
        ;;
    venv)
        ensure_venv
        ACTIVE_BACKEND="venv"
        ;;
    mamba)
        ensure_mamba_env
        ACTIVE_BACKEND="mamba"
        ;;
    *)
        echo -e "${R}  ERROR: Invalid ENV_BACKEND=${ENV_BACKEND} (expected auto|venv|mamba)${N}"
        exit 1
        ;;
esac

echo -e "${C}  Active backend: ${ACTIVE_BACKEND}${N}"

ensure_python_deps

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

python3 -W ignore::UserWarning -c "
import torch, sys
print(f'  Python {sys.version.split()[0]}  PyTorch {torch.__version__}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  {torch.cuda.get_device_name(0)}  SM {cap[0]}.{cap[1]}  {mem:.0f}GB VRAM  CUDA {torch.version.cuda}')
    print(f'  cuDNN {torch.backends.cudnn.version()}  SDPA backends: flash={torch.backends.cuda.flash_sdp_enabled()} mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} math={torch.backends.cuda.math_sdp_enabled()}')
else:
    print('  CPU-only mode')
"

if [ ! -f "$SERVER_SCRIPT" ]; then
    echo -e "${R}  ERROR: $SERVER_SCRIPT not found${N}"
    exit 1
fi

echo -e "${G}  Starting LLMWS server${N}"
exec python3 "$SERVER_SCRIPT"
