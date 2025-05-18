#!/bin/bash
set -e

if ! command -v git &>/dev/null; then
    echo "Git is not installed. Please install Git and try again."
    exit 1
fi

if !  git lfs install  &>/dev/null; then
    echo "Git LFS is not installed. Please install Git and try again. "
    exit 1
fi

MODEL_DIR="$(pwd)/models"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"


if [ ! -d "Llama-3.1-Tulu-3-8B" ]; then
    git clone https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B
    echo "Model successfully cloned to $MODEL_DIR"
else
    echo "Model directory already exists at $MODEL_DIR/Llama-3.1-Tulu-3-8B"
fi
