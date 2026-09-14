# MoT-ALIT 128-token budget sweep

This repository runs four matched fixed-budget ALIT/VQGAN experiments in sequence:

| Run | ALIT tokens | VQGAN refinement grids | Total |
|---|---:|---:|---:|
| `32_96` | 32 | 96 | 128 |
| `64_64` | 64 | 64 | 128 |
| `96_32` | 96 | 32 | 128 |
| `128_0` | 128 | 0 | 128 |

Every run starts independently from the same official ALIT-small quantized EMA
and VQGAN checkpoints. No run initializes from another budget's checkpoint.
All four use the same losses, discriminator, optimizer schedule, seed, per-GPU
batch size, and 10-epoch duration.

## Setup

```bash
git clone https://github.com/LCGLiChenge/MoTALIT.git
cd MoTALIT
pip install -r requirements.txt
```

The CUDA/PyTorch environment is assumed to exist. Download and verify the
official ALIT/VQGAN checkpoints and LPIPS weights:

```bash
python scripts/download_weights.py
```

Pre-download the frozen ConvNeXt, DINOv2, VGG16, and CLIP backbones before the
long run:

```bash
export TORCH_HOME="$PWD/weights/torch"
python scripts/prefetch_features.py
```

Log in once if W&B tracking is required:

```bash
wandb login
```

## Run all four experiments

Replace the ImageNet path and run one command from the repository root:

```bash
export TORCH_HOME="$PWD/weights/torch"

python scripts/train_h200_budget_sweep.py \
  --data-path /path/to/imagenet/train \
  --gpus 0,1,2,3,4,5,6,7 \
  --accum-steps 1
```

Before training, the launcher probes `128+0`, the highest-memory budget, using
complete generator and discriminator updates. It searches for the largest
per-GPU batch size that still leaves at least 8 GiB reserved-memory headroom,
then uses that same batch size for all four groups. Override the reserve with
`--reserve-memory-gib N`. Pass an explicit `--batch-size N` to skip probing.
Use `--no-wandb` to keep only local JSONL logs.

Each epoch atomically overwrites one budget-specific checkpoint. No numbered or
step checkpoints are created:

```text
results/alit_budget_sweep_10epoch/32_96/latest_32_96.pt
results/alit_budget_sweep_10epoch/64_64/latest_64_64.pt
results/alit_budget_sweep_10epoch/96_32/latest_96_32.pt
results/alit_budget_sweep_10epoch/128_0/latest_128_0.pt
```

If the launcher is interrupted, rerun the same command. An incomplete group
resumes from its latest completed epoch, and groups already at epoch 10 are
skipped. Keep the same GPU count, batch size, and accumulation setting when
resuming so the data cursor remains valid. To guarantee the exact same batch on
resume, pass the batch reported by the first probe as `--batch-size N`.

The shared schedule is epoch-based: Router-only ends at epoch 0.5, joint
reconstruction begins at epoch 0.5, and discriminator warmup starts at epoch
1.5 for 0.01 epoch before full GAN training. Generator learning rates use
cosine decay across 10 epochs.

## Checks

Inspect the automatic probe and all four generated commands without starting
training:

```bash
python scripts/train_h200_budget_sweep.py \
  --data-path /path/to/imagenet/train \
  --dry-run
```

Run package tests:

```bash
python -m unittest discover -s tests -v
python scripts/download_weights.py --check-only
```

The repository vendors the official ALIT and DINOv2 source needed by this
experiment. No model checkpoint is committed to Git.
