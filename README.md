# HyperSteer

Steering models at scale with hypernetworks. This repo supports training and inference with LLM-judge evaluation for training your own HyperSteer models. Basic distributed training support is available with DistributedSampler.

## Setup

```bash
export UV_TORCH_BACKEND=auto
# use --locked in sync command to skip dependency resolution
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
│   ├── base.yaml
│   ├── hypersteer.yaml
│   └── hypersteer_sel.yaml
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
├── model
│   ├── base.yaml
│   └── hypersteer.yaml
└── train
    ├── base.yaml
    └── hypersteer.yaml
```

An `experiment` is a pre-configured set of overrides for the default configuration. Available experiments:
- `hypersteer`: Base HyperSteer configuration
- `hypersteer_sel`: HyperSteer with selection head enabled

Example usage:

```bash
python -m hypersteer.scripts.[train|inference|evaluate|generate] experiment=hypersteer ...<hydra overrides>
```

By default, training runs inference and evaluation at the end. This can be configured in the experiment configuration.

## Environment Setup

Put API keys in a `.env` file at the project root. Key environment variables:

- `WANDB_PROJECT` - Weights & Biases project name
- `WANDB_ENTITY` - Weights & Biases entity/team name
- `OPENAI_API_KEY` - OpenAI API key (required for LLM-as-judge evaluation)
- `OPENAI_DRY_RUN=1` - Set to mock OpenAI API calls during development/testing

## Scripts

The project provides four main scripts:

- `train` - Train a HyperSteer model
- `inference` - Run steering inference with a trained model
- `evaluate` - Evaluate model performance using LLM-as-judge
- `generate` - Generate training data from concept descriptions

All scripts use Hydra for configuration and support the same override syntax.

## Architecture

HyperSteer uses hypernetworks to generate steering vectors that modify language model behavior at inference time. Key components:

- **HyperAdditiveIntervention**: pyvene-based intervention that adds scaled steering vectors to hidden states at specified layers. Optional SelectionHead for per-token steering masks.
- **HyperSteerRegression**: Regression-based hypernet that generates steering vectors from concept text embeddings.
- **Dataset Factory**: Registry-based system for loading datasets from HuggingFace (currently supports axbench).
- **LM Judge Evaluator**: Uses LLMs to evaluate generation quality based on concept relevance, instruction relevance, and fluency.

Default dataset: `pyvene/axbench-concept500` from HuggingFace. Concepts are behavioral attributes (e.g., "responds formally", "uses humor") with positive/negative example pairs.

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

# TODO

- [x] Faster initialization for big networks (on-device initialization)
- [x] HuggingFace datasets integration with factory pattern
- [x] Safetensors support for checkpointing
- [ ] Re-implement attention-based hypernet (HyperSteerAttn)
- [ ] Unified checkpointing and robust resume/fault tolerance in training
- [ ] Enhanced data generation pipeline
- [ ] Distributed training support via FSDP and DDP
- [ ] Optional DeepSpeed inference kernels with distributed data parallel
- [ ] Optional Liger kernel, Flash Attention 2 integration for faster training
