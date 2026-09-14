# H200 ALIT/VQGAN 128-Token Budget Sweep

Date: 2026-09-15

## Objective

Train four fixed-budget variants sequentially with one H200 launcher:

- 32 ALIT + 96 VQGAN grids
- 64 ALIT + 64 VQGAN grids
- 96 ALIT + 32 VQGAN grids
- 128 ALIT + 0 VQGAN grids

Every variant has a total budget of 128 tokens and starts independently from
the same official ALIT-small EMA and VQGAN checkpoints.

## Shared training contract

- 10 epochs per variant, eight GPUs by default.
- Before training, probe the highest-memory 128+0 configuration with a complete
  GAN step and select the maximum safe per-GPU batch that leaves at least 8 GiB.
- Use the one probed batch size for every variant to preserve comparability.
- Same seed, accumulation, losses, discriminator, and optimizer schedule for
  every variant.
- L1 3.0, effective LPIPS 0.6, DINO feature 0.5, CLIP feature 0.5,
  discriminator feature matching 0.5, GAN 0.12.
- Router-only through epoch 0.5; joint reconstruction afterward.
- Discriminator warmup begins at epoch 1.5 and lasts 0.01 epoch.
- Generator learning rates use cosine decay over 10 epochs.
- `latest_every=0` and `save_epoch_every=0`.
- At every epoch boundary, each run atomically overwrites only its own
  `latest_<1d>_<2d>.pt` checkpoint.

## Launcher and outputs

Launcher: `scripts/train_h200_budget_sweep.py`

Batch probe: `scripts/probe_h200_batch.py`

The probe grows and then binary-searches batch size. A run is considered safe
only when `peak_reserved_gib <= gpu_total_gib - 8`; a successful CUDA step above
that cap is treated as an unsafe upper bound, not as the final recommendation.
An explicit `--batch-size N` bypasses probing.

Outputs:

```text
results/alit_budget_sweep_10epoch/32_96/latest_32_96.pt
results/alit_budget_sweep_10epoch/64_64/latest_64_64.pt
results/alit_budget_sweep_10epoch/96_32/latest_96_32.pt
results/alit_budget_sweep_10epoch/128_0/latest_128_0.pt
```

Rerunning the launcher resumes an incomplete variant from its latest completed
epoch and skips variants already at epoch 10. It never resumes one budget from
another budget's checkpoint.

## Verification

- Python compilation after 8-GiB probe integration: passed.
- Automatic-probe dry-run and all four command expansions: passed.
- Explicit `--batch-size 24` probe bypass: passed.
- Unit tests: 8/8 passed.
- Real single-GPU 96+32 full-GAN forward/backward smoke with official
  checkpoints: passed; hard mask count was exactly 32.
- Real atomic-save smoke: only `latest_96_32.pt` and `log.txt` were produced;
  no numbered checkpoint was created. Saved metadata was epoch 1, step 1,
  budget 96+32.

No formal 10-epoch sweep has been run yet.
