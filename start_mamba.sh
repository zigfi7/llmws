#!/bin/bash

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
    echo "Creating the myenv environment with Python 3.12..."
    micromamba create -y -n myenv python=3.12
fi

micromamba activate myenv
/usr/bin/bash start.sh

micromamba deactivate
