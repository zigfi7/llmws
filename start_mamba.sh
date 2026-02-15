#!/bin/bash
python=3.11
MICROMAMBA_DIR="$(pwd)/micromamba"
MICROMAMBA_BIN="$MICROMAMBA_DIR/bin/micromamba"

if [ ! -f "$MICROMAMBA_BIN" ]; then
    echo "Micromamba not found. Downloading..."
    mkdir -p "$MICROMAMBA_DIR"
    if [ "$(uname -m)" == "aarch64" ]; then
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C "$MICROMAMBA_DIR"
    else
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MICROMAMBA_DIR"
    fi
    echo "Micromamba has been downloaded to $MICROMAMBA_BIN"
fi

export MAMBA_ROOT_PREFIX="$MICROMAMBA_DIR/root"
mkdir -p "$MAMBA_ROOT_PREFIX"

eval "$($MICROMAMBA_BIN shell hook --shell bash)"
export PATH="$MICROMAMBA_DIR/bin:$PATH"

if ! micromamba env list | grep -q "^myenv\s"; then
    echo "Creating the myenv environment with Python $python ..."
    micromamba create -y -n myenv python=$python
    micromamba activate myenv
    pip install --upgrade pip
    pip install torch torchvision torchaudio
    pip install "torch>=2.3.1"
    pip install "accelerate>=0.31.0"
    pip install "transformers>=4.43.0"
    pip install "websockets"
    pip install "flash_attn>=2.5.8"
    pip install protobuf sentencepiece
    pip install peft scipy backoff
    pip install flash-attn --no-build-isolation
else
    micromamba activate myenv
fi

python llmws.py
micromamba deactivate
