import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import torch

from experiments.alit_vqgan_mix.model import fixed_topk_ste_mask
from experiments.alit_vqgan_mix.train_mot_fixed96_5epoch import (
    parse_args,
    reconstruction_loss,
)
from scripts.probe_h200_batch import midpoint, parse_csv_ints, within_memory_budget
from scripts.train_h200_budget_sweep import (
    BUDGETS,
    PROBE_BUDGET,
    TARGET_EPOCHS,
    build_probe_command,
    build_train_command,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/h200_alit64_fixed64_l1_3_20epoch.yaml"
ABLATION_CONFIG = ROOT / "configs/h200_alit128_fixed0_l1_3_20epoch.yaml"
SWEEP_CONFIG = ROOT / "configs/h200_budget_sweep_l1_3_10epoch.yaml"


class H200PackageTest(unittest.TestCase):
    def test_config_and_effective_weights(self):
        config = yaml.safe_load(CONFIG.read_text())
        argv = [
            "train",
            "--config", str(CONFIG),
            "--data-path", "unused/train",
            "--alit-ckpt", "unused/alit.pth",
            "--vqgan-ckpt", "unused/vqgan.ckpt",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual((args.alit_tokens, args.refine_tokens), (64, 64))
        self.assertEqual((args.batch_size, args.accum_steps), (24, 1))
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.lambda_l1, 3.0)
        self.assertAlmostEqual(args.lambda_mix * args.lambda_lpips, 0.6)
        self.assertAlmostEqual(float(reconstruction_loss(args, 2.0, 5.0)), 9.0)
        self.assertEqual(config["save_epoch_every"], 0)

    def test_vendored_runtime_sources_exist(self):
        required = [
            ROOT / "third_party/alit/adaptive_tokenizers/adaptive_vqgan.py",
            ROOT / "third_party/dinov2/hubconf.py",
            ROOT / "third_party/llamagen/tokenizer/tokenizer_image/lpips.py",
        ]
        self.assertTrue(all(path.is_file() for path in required))

    def test_batch_probe_helpers(self):
        self.assertEqual(parse_csv_ints("0,1,7", minimum=0), [0, 1, 7])
        self.assertEqual(midpoint(24, 48), 36)
        self.assertEqual(midpoint(36, 48), 42)
        self.assertTrue(within_memory_budget(132.0, 140.0, 8.0))
        self.assertFalse(within_memory_budget(132.01, 140.0, 8.0))

    def test_alit128_ablation_changes_only_budget_and_identity(self):
        baseline = yaml.safe_load(CONFIG.read_text())
        ablation = yaml.safe_load(ABLATION_CONFIG.read_text())
        differences = {
            key for key in baseline.keys() | ablation.keys()
            if baseline.get(key) != ablation.get(key)
        }
        self.assertEqual(
            differences,
            {"alit_tokens", "refine_tokens", "output_dir", "wandb_name"},
        )
        argv = [
            "train",
            "--config", str(ABLATION_CONFIG),
            "--data-path", "unused/train",
            "--alit-ckpt", "unused/alit.pth",
            "--vqgan-ckpt", "unused/vqgan.ckpt",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual((args.alit_tokens, args.refine_tokens), (128, 0))

    def test_budget_sweep_contract(self):
        config = yaml.safe_load(SWEEP_CONFIG.read_text())
        self.assertEqual(config["epochs"], TARGET_EPOCHS)
        self.assertEqual(config["latest_every"], 0)
        self.assertEqual(config["save_epoch_every"], 0)
        self.assertEqual(BUDGETS, ((32, 96), (64, 64), (96, 32), (128, 0)))
        for alit_tokens, refine_tokens in BUDGETS:
            command = build_train_command(
                config=SWEEP_CONFIG, data_path=Path("train"),
                alit_ckpt=Path("alit.pt"), vqgan_ckpt=Path("vqgan.pt"),
                output_dir=Path("results") / f"{alit_tokens}_{refine_tokens}",
                alit_tokens=alit_tokens, refine_tokens=refine_tokens,
                batch_size=24, accum_steps=1, world_size=8, resume=None, wandb=True,
            )
            self.assertEqual(alit_tokens + refine_tokens, 128)
            self.assertIn(f"latest_{alit_tokens}_{refine_tokens}.pt", command)
            self.assertEqual(command[command.index("--save-epoch-every") + 1], "0")
            self.assertEqual(command[command.index("--epochs") + 1], "10")

    def test_sweep_probe_uses_hardest_budget_and_eight_gib_reserve(self):
        command = build_probe_command(
            config=SWEEP_CONFIG,
            data_path=Path("train"),
            alit_ckpt=Path("alit.pt"),
            vqgan_ckpt=Path("vqgan.pt"),
            gpu_ids=list(range(8)),
            initial_batch_size=96,
            min_batch_size=1,
            max_batch_size=256,
            reserve_memory_gib=8.0,
            output_root=Path("/tmp/probe"),
        )
        self.assertEqual(PROBE_BUDGET, (128, 0))
        self.assertEqual(command[command.index("--alit-tokens") + 1], "128")
        self.assertEqual(command[command.index("--refine-tokens") + 1], "0")
        self.assertEqual(command[command.index("--reserve-memory-gib") + 1], "8.0")

    def test_96_token_cli_and_named_latest(self):
        argv = [
            "train", "--config", str(SWEEP_CONFIG),
            "--data-path", "unused/train", "--alit-ckpt", "unused/alit.pth",
            "--vqgan-ckpt", "unused/vqgan.ckpt", "--alit-tokens", "96",
            "--refine-tokens", "32", "--latest-filename", "latest_96_32.pt",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual((args.alit_tokens, args.refine_tokens), (96, 32))
        self.assertEqual(args.latest_filename, "latest_96_32.pt")

    def test_zero_refinement_mask_keeps_router_ste_gradient(self):
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        mask, hard, soft = fixed_topk_ste_mask(logits, tokens=0, tau=0.7)
        self.assertTrue(bool((mask == 0).all()))
        self.assertEqual(float(hard.sum()), 0.0)
        self.assertGreater(float(soft.detach().max()), 0.0)
        weights = torch.arange(256, dtype=mask.dtype).view(1, 1, 16, 16)
        (mask * weights).sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
