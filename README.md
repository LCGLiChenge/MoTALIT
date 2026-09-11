# MoT-ALIT 64+64

This repository contains the current fixed-budget ALIT experiment only:

- 64 official recurrent ALIT tokens;
- exactly 64 Router-selected native VQGAN grids;
- total budget: 128 tokens;
- L1 coefficient: 3.0;
- effective LPIPS coefficient: 0.6;
- 20 epochs on 8 H200 GPUs;
- batch 24 per GPU, accumulation 1, effective global batch 192.

The training starts from the official ALIT-small quantized EMA and VQGAN
checkpoints. It does not resume the previous five-epoch checkpoint.

## 1. Clone

```bash
git clone https://github.com/LCGLiChenge/MoTALIT.git
cd MoTALIT
```

The CUDA/PyTorch environment is assumed to exist. Install missing Python
packages only if needed:

```bash
pip install -r requirements.txt
```

## 2. Prepare weights

Download and verify the two official ALIT checkpoints and the LPIPS linear
weights used by both ALIT and LlamaGen:

```bash
python scripts/download_weights.py
```

Pre-download the frozen ConvNeXt, DINOv2, VGG16 and CLIP feature backbones:

```bash
export TORCH_HOME="$PWD/weights/torch"
python scripts/prefetch_features.py
```

The second command is optional when the machine already has these model caches
and internet access, but running it before a long job catches missing weights.

## 3. W&B

```bash
wandb login
```

To disable W&B, append `--no-wandb` to the training command.

## 4. Train on 8 H200 GPUs

Run from the repository root and replace the ImageNet path:

```bash
export TORCH_HOME="$PWD/weights/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  experiments/alit_vqgan_mix/train_mot_fixed96_5epoch.py \
  --config configs/h200_alit64_fixed64_l1_3_20epoch.yaml \
  --data-path /path/to/imagenet/train \
  --alit-ckpt weights/alit_vqgan_small_quantized_latent.pth \
  --vqgan-ckpt weights/vqgan.ckpt
```

The schedule is epoch-based. With ImageNet-1K train and global batch 192, it
uses 6,672 optimizer updates per epoch and 133,440 updates in total. Router-only
training ends at epoch 0.5. Joint reconstruction starts at epoch 0.5. The D-only
warmup starts at epoch 1.5 and lasts 0.01 epoch, followed by full GAN and feature
matching. Generator learning rates use cosine decay over all 20 epochs.

`latest.pt` is updated at every epoch. Numbered snapshots are written at epochs
10 and 20 only. Local JSONL logs remain in the output directory even if W&B is
unavailable.

## Resume

Use the same global batch and append:

```bash
--resume results/alit64_vqgan_mix_fixed64_l1_3_h200_20epoch/latest.pt
```

Changing the global batch changes `steps_per_epoch`, so exact data-cursor resume
is intentionally rejected.

## Quick checks

```bash
python -m unittest discover -s tests -v
python scripts/download_weights.py --check-only
```

For a no-checkpoint single-GPU smoke after preparing weights, use:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  experiments/alit_vqgan_mix/train_mot_fixed96_5epoch.py \
  --config configs/h200_alit64_fixed64_l1_3_20epoch.yaml \
  --data-path /path/to/imagenet/train \
  --alit-ckpt weights/alit_vqgan_small_quantized_latent.pth \
  --vqgan-ckpt weights/vqgan.ckpt \
  --batch-size 1 --accum-steps 1 --limit-samples 8 --epochs 1 \
  --smoke-steps 4 --smoke-no-save --router-only-epochs 0.125 \
  --gan-start-epoch 0.25 --d-warmup-epochs 0.125 \
  --native-decode-chunk 1 --audit-gradients --no-wandb
```

The source tree vendors the official ALIT and DINOv2 code needed by this
experiment. The LlamaGen LPIPS implementation is included with attribution.
No model checkpoint is committed to Git.
