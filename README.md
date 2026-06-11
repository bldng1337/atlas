# Atlas

Training pipeline, dataset tools, and inference UI for controlling [MESA](https://huggingface.co/NewtNewt/MESA) via Controlnet a diffusion model for synthesizing satellite terrain imagery and heightmaps.

## Setup

### Requirements

- Python 3.9
- CUDA-capable GPU (CU128)
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
uv sync
```

## Usage

### Inference UI

```bash
uv run app.py --controlnet-path ./weights/controlnet
```

Options: `--device`, `--mixed-precision` (fp16/bf16/fp32), `--share`, `--server-port`.

### Training

```bash
uv run train_controlnet.py
```

Training is configured via module-level variables in the script and/or a `.env` file (see `train_utils.py` → `parse_args`). There are premade configs for training on 16gb, 24gb, and 32gb GPUs (use with `--config ./16gb.json`).

### Dataset Pipeline

1. Download and filter metadata: `dataset/download_meta.py`
2. Download imagery & DEM: `dataset/download_img.py`
3. Merge into a single dataset: `dataset/collect.py`
4. Generate feature maps: `dataset/create_features.py`
5. Add text prompts: `dataset/add_prompts.py`

The dataset combines [Major-TOM](https://huggingface.co/Major-TOM) Sentinel-2 imagery with Copernicus DEM data, filtered by cloud cover and human footprint index.
