#!/bin/bash

set -e

if ! command -v git &>/dev/null; then
    echo "Git is not installed. Please install Git and try again."
    exit 1
fi

if ! git lfs --version &>/dev/null; then
    echo "Git LFS is not installed. Installing..."
    git lfs install
fi

#MODEL_DIR="/opt/llm/models"
MODEL_DIR="$(pwd)/models"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ ! -d "Phi-3.5-mini-instruct" ]; then
    git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct
    echo "Model successfully cloned to $MODEL_DIR"
else
    echo "Model directory already exists at $MODEL_DIR/Phi-3.5-mini-instruct"
fi