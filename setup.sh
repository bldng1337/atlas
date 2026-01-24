#!/bin/bash
set -e

echo "Installing system dependencies..."

apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget

echo "Installing UV..."

curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Installing project dependencies..."

uv sync --no-install-project --frozen

uv pip install triton

if [ ! -z "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face..."
    uv run huggingface-cli login --token $HF_TOKEN
fi

echo "Setup complete!"
