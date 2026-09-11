# ALIT Upstream

The source files in this directory are copied without behavioral changes from:

- Repository: https://github.com/ShivamDuggal4/adaptive-length-tokenizer
- Commit: `8c0b4ec29c76963ad81ffdd91c9afd901f5215f5e`
- Paper: Adaptive Length Image Tokenization via Recurrent Allocation (ICLR 2025)

Only the `adaptive_tokenizers`, `base_tokenizers`, `modules`, and `utils`
directories required to instantiate the official ALIT/VQGAN model are vendored.
Experiment-specific fixed-rollout and mixed-decoding logic lives outside this
directory under `experiments/alit_vqgan_mix`.
