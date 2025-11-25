# HyperSteer

Steering models at scale with hypernetworks. This repo supports distributed training and inference, along with LLM-judge-in-the-loop evaluation for training your own HyperSteer models.

## Setup

```bash
UV_TORCH_BACKEND=auto
# in root directory
uv sync
# To install development dependencies, including `ipykernel`, `jupyter`, `pre-commit`, and `ai_commit`, run:
uv sync --extra dev
# To install all optional dependencies, including `ray` and `flash-attn`, run:
uv sync --all-extras
```

# Usage

The [Hydra](https://hydra.cc/docs/configure_hydra/intro/) config is structured hierarchically:

```
config
├── config.yaml
├── dataset
│   ├── axbench.yaml
│   └── base.yaml
├── evaluate
│   ├── base.yaml
│   └── hypersteer.yaml
├── experiment
│   ├── base_gh200.yaml
│   ├── base.yaml
│   └── hypersteer.yaml
├── generate
│   └── base.yaml
├── hydra
│   ├── hydra_logging
│   │   └── colorlog.yaml
│   └── job_logging
│       └── colorlog.yaml
├── inference
│   ├── base.yaml
│   └── hypersteer.yaml
├── launcher
│   ├── base.yaml
│   └── ray.yaml
├── model
│   ├── base.yaml
│   └── hypersteer.yaml
├── train
│   ├── base.yaml
│   └── hypersteer.yaml
└── wandb
    └── base.yaml
```

An `experiment` is a pre-configured set of overrides for the default configuration. For example, to use the `hypersteer` experiment:


```bash
python -m hypersteer.scripts.[train|inference|evaluate] experiment=hypersteer ...<hydra overrides>
```

By default, training runs inference and evaluation at the end. This can be configured in the extensive configuration.

## Environment Setup

Put API keys in a `.env` file.

# TODO

- [x] Faster initialization for big networks (pretty easy - just do on device)
- [x] Better data management (Right now only axbench support, and we use the parquets committed to GH). Need to migrate to using just Huggingface datasets completely
- [x] Safetensors
- [ ] Unified/clean checkpointing and robust resume/fault tolerance in training
- [ ] Proper implementation of data generation (TODO: jiuding)
- [ ] Robust distributed training support via FSDP and DDP
- [ ] Optional Fast Deepspeed inference kernels and revamped inference logic with distributed data parallel support
- [ ] Optional Liger kernel, FA2, etc. for faster training
