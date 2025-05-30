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

# TODO


- [x] Faster initialization for big networks (pretty easy - just do on device)
- [x] Better data management (Right now only axbench support, and we use the parquets committed to GH). Need to migrate to using just Huggingface datasets completely
- [] Safetensors, unified/clean checkpointing and robust resume/fault tolerance in training
- [ ] Proper implementation of data generation (TODO: jiuding)
- [ ] Robust distributed training support via FSDP and DDP
- [ ] Fast Deepspeed inference kernels and revamped inference logic with distributed data parallel support
- [ ] Liger kernel, FA2, etc. for faster training