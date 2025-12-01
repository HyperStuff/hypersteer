# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HyperSteer trains hypernetworks to steer language models at scale. The system uses concept descriptions to generate steering vectors that modify hidden states during inference, enabling controllable text generation.

## Commands

```bash
# Setup
UV_TORCH_BACKEND=auto uv sync          # Basic install
uv sync --extra dev                     # Development deps (jupyter, pre-commit)
uv sync --all-extras                    # All deps (ray, flash-attn)

# Training
python -m hypersteer.scripts.train experiment=hypersteer

# Inference (steering)
python -m hypersteer.scripts.inference experiment=hypersteer dump_dir=<path_to_trained_model>

# Evaluation
python -m hypersteer.scripts.evaluate experiment=hypersteer dump_dir=<path_to_trained_model>

# Hydra override examples
python -m hypersteer.scripts.train experiment=hypersteer train.lr=1e-4 train.batch_size=8
python -m hypersteer.scripts.train model=hypersteer dataset=axbench  # Mix configs
```

## Architecture

### Core Pipeline

1. **HyperAdditiveIntervention** (`hypersteer/models/modules/interventions.py`): pyvene-based intervention that adds scaled steering vectors to hidden states at specified layers. Optional `SelectionHead` learns per-token masks for selective intervention.

2. **HyperSteer Models** (`hypersteer/models/`):
   - `HyperSteerBase` (`hypersteer/models/hypersteer_base.py`) - Abstract base class with shared logic for training, inference, visualization
   - `HyperSteerRegression` (`hypersteer/models/hypersteer_regression.py`) - Regression-based hypernet (linear projection from concept embedding via base model)

   Note: HyperSteerAttn (attention-based hypernet) was removed. Config files may reference `hypernet_type: "attn"` but this is not currently implemented.

   Model dispatch: `get_model("HyperSteer", model_config=cfg)` attempts to use `cfg.hypernet_type` to select the implementation, but currently only "regression" is available.

3. **Trainer** (`hypersteer/training/trainer.py`): Generic trainer using `TrainerMixin` interface. Models implement `train_step`, `val_step`, `setup_optimizer` methods.

### Key Abstractions

- **pyvene IntervenableModel**: Wraps target model for activation intervention at configurable layers
- **Dataset Factory** (`hypersteer/data/base.py`): Creates training/eval datasets from HuggingFace (axbench)
- **LMJudgeEvaluator** (`hypersteer/evaluators/lm_judge.py`): LLM-as-judge evaluation using concept relevance, instruction relevance, and fluency metrics
- **Model Factory** (`hypersteer/models/__init__.py`): Simple dict-based factory, no decorator magic

### Config System

Hydra-based hierarchical config in `config/`. Key structure:
- `experiment/`: Pre-configured overrides (use `experiment=hypersteer`)
- `model/`: Model architecture configs
- `train/`: Training hyperparameters
- `inference/`: Steering and factor selection settings
- `evaluate/`: Evaluation settings

An experiment config merges into root level via `@package _global_`.

## Environment

Put API keys (OpenAI, WandB) in `.env` file. Key env vars:
- `WANDB_PROJECT`, `WANDB_ENTITY`
- `OPENAI_API_KEY` (for LM judge evaluation)
- `OPENAI_DRY_RUN=1` to mock API calls

## Data

Default dataset: `pyvene/axbench-concept500` from HuggingFace. Concepts are behavioral attributes (e.g., "responds formally", "uses humor") with positive/negative example pairs.
