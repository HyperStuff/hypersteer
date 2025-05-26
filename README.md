# Hypersteer

Steering models with hypernetworks.

## Setup

Install with [uv](https://docs.astral.sh/uv/):

```bash
# Clone and install
git clone <repo-url>
cd hypersteer
uv sync
```

### Hardware Options

Choose one based on your setup:

```bash
# CPU only
uv sync --extra cpu

# CUDA 12.8
uv sync --extra cu128

# CUDA 12.8 nightly
uv sync --extra cu128-nightly
```

### Optional Extras

```bash
# Development tools
uv sync --extra dev

# ReFT support
uv sync --extra reft

# Flash attention
uv sync --extra flash-attn

# Ray launcher
uv sync --extra ray
```

## Usage

```bash
# Training
uv run python train.py

# Inference
uv run python inference.py

# Evaluation
uv run python evaluate.py

# Generation
uv run python generate.py
```

Configuration via Hydra configs in `config/`.

# TODO

- better data management (Right now only axbench support, and we use the parquets commmitted to GH). Need to migrate to using just Huggingface datasets completely
- Proper implementation of data generation
- robust distributed training support via FSDP and DDP
- Fast Deepspeed inference kernels and revamped inference logic with distributed supprot
- Liger kernel, FA2, etc. for faster training