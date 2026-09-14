#!/usr/bin/env python3
"""Train four matched 128-token ALIT/VQGAN budgets sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = "experiments/alit_vqgan_mix/train_mot_fixed96_5epoch.py"
BUDGETS = ((32, 96), (64, 64), (96, 32), (128, 0))
TARGET_EPOCHS = 10
PROBE_BUDGET = (128, 0)


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(item < 0 for item in gpu_ids):
        raise ValueError("--gpus must contain distinct non-negative GPU IDs")
    return gpu_ids


def budget_name(alit_tokens: int, refine_tokens: int) -> str:
    return f"{alit_tokens}_{refine_tokens}"


def build_probe_command(
    *,
    config: Path,
    data_path: Path,
    alit_ckpt: Path,
    vqgan_ckpt: Path,
    gpu_ids: list[int],
    initial_batch_size: int,
    min_batch_size: int,
    max_batch_size: int,
    reserve_memory_gib: float,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/probe_h200_batch.py",
        "--config", str(config),
        "--data-path", str(data_path),
        "--alit-ckpt", str(alit_ckpt),
        "--vqgan-ckpt", str(vqgan_ckpt),
        "--gpus", ",".join(str(item) for item in gpu_ids),
        "--initial-batch-size", str(initial_batch_size),
        "--min-batch-size", str(min_batch_size),
        "--max-batch-size", str(max_batch_size),
        "--reserve-memory-gib", str(reserve_memory_gib),
        "--alit-tokens", str(PROBE_BUDGET[0]),
        "--refine-tokens", str(PROBE_BUDGET[1]),
        "--output-root", str(output_root),
    ]


def build_train_command(
    *,
    config: Path,
    data_path: Path,
    alit_ckpt: Path,
    vqgan_ckpt: Path,
    output_dir: Path,
    alit_tokens: int,
    refine_tokens: int,
    batch_size: int,
    accum_steps: int,
    world_size: int,
    resume: Path | None,
    wandb: bool,
) -> list[str]:
    name = budget_name(alit_tokens, refine_tokens)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        TRAIN_SCRIPT,
        "--config",
        str(config),
        "--data-path",
        str(data_path),
        "--alit-ckpt",
        str(alit_ckpt),
        "--vqgan-ckpt",
        str(vqgan_ckpt),
        "--output-dir",
        str(output_dir),
        "--alit-tokens",
        str(alit_tokens),
        "--refine-tokens",
        str(refine_tokens),
        "--epochs",
        str(TARGET_EPOCHS),
        "--batch-size",
        str(batch_size),
        "--accum-steps",
        str(accum_steps),
        "--latest-every",
        "0",
        "--save-epoch-every",
        "0",
        "--latest-filename",
        f"latest_{name}.pt",
        "--wandb-name",
        f"alit_{name}_l1_3_h200_{TARGET_EPOCHS}epoch",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if not wandb:
        command.append("--no-wandb")
    return command


def checkpoint_epoch(path: Path, alit_tokens: int, refine_tokens: int) -> float:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    actual_budget = (
        int(config.get("alit_tokens", -1)),
        int(config.get("refine_tokens", -1)),
    )
    expected_budget = (alit_tokens, refine_tokens)
    if actual_budget != expected_budget:
        raise RuntimeError(
            f"{path} belongs to budget {actual_budget}, expected {expected_budget}"
        )
    return float(checkpoint.get("epoch", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h200_budget_sweep_l1_3_10epoch.yaml"),
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--alit-ckpt",
        type=Path,
        default=Path("weights/alit_vqgan_small_quantized_latent.pth"),
    )
    parser.add_argument("--vqgan-ckpt", type=Path, default=Path("weights/vqgan.ckpt"))
    parser.add_argument("--output-root", type=Path, default=Path("results/alit_budget_sweep_10epoch"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="per-GPU batch size; 0 probes the largest safe value automatically",
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--reserve-memory-gib", type=float, default=8.0)
    parser.add_argument("--probe-initial-batch-size", type=int, default=96)
    parser.add_argument("--probe-min-batch-size", type=int, default=1)
    parser.add_argument("--probe-max-batch-size", type=int, default=256)
    parser.add_argument(
        "--probe-output-root", type=Path,
        default=Path("/tmp/motalit_h200_batch_probe_128_0"),
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    if args.batch_size < 0 or args.accum_steps < 1:
        parser.error("--batch-size must be non-negative and --accum-steps must be positive")
    if args.reserve_memory_gib <= 0:
        parser.error("--reserve-memory-gib must be positive")
    if not 1 <= args.probe_min_batch_size <= args.probe_initial_batch_size <= args.probe_max_batch_size:
        parser.error("require probe min <= initial <= max and probe min >= 1")

    config = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    data_path = args.data_path.resolve()
    alit_ckpt = (ROOT / args.alit_ckpt).resolve() if not args.alit_ckpt.is_absolute() else args.alit_ckpt
    vqgan_ckpt = (ROOT / args.vqgan_ckpt).resolve() if not args.vqgan_ckpt.is_absolute() else args.vqgan_ckpt
    output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    probe_output_root = args.probe_output_root.resolve()

    if not args.dry_run:
        for path, label in ((config, "config"), (data_path, "ImageNet train"),
                            (alit_ckpt, "ALIT checkpoint"), (vqgan_ckpt, "VQGAN checkpoint")):
            if not path.exists():
                raise FileNotFoundError(f"missing {label}: {path}")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_ids)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment.setdefault("OMP_NUM_THREADS", "4")

    batch_size = args.batch_size
    probe = None
    if batch_size == 0:
        probe = build_probe_command(
            config=config,
            data_path=data_path,
            alit_ckpt=alit_ckpt,
            vqgan_ckpt=vqgan_ckpt,
            gpu_ids=gpu_ids,
            initial_batch_size=args.probe_initial_batch_size,
            min_batch_size=args.probe_min_batch_size,
            max_batch_size=args.probe_max_batch_size,
            reserve_memory_gib=args.reserve_memory_gib,
            output_root=probe_output_root,
        )
        if args.dry_run:
            batch_size = args.probe_initial_batch_size
        else:
            print(
                f"Probing maximum safe batch with {PROBE_BUDGET[0]}+{PROBE_BUDGET[1]} "
                f"and {args.reserve_memory_gib:g} GiB headroom",
                flush=True,
            )
            subprocess.run(probe, cwd=ROOT, env=environment, check=True)
            recommendation = json.loads((probe_output_root / "recommended.json").read_text())
            actual_budget = (
                int(recommendation["alit_tokens"]),
                int(recommendation["refine_tokens"]),
            )
            if actual_budget != PROBE_BUDGET:
                raise RuntimeError(f"probe used budget {actual_budget}, expected {PROBE_BUDGET}")
            if int(recommendation["world_size"]) != len(gpu_ids):
                raise RuntimeError("probe result world size does not match this launch")
            batch_size = int(recommendation["batch_size"])
            print(
                f"Using batch_size={batch_size} for every budget "
                f"({float(recommendation['memory_headroom_gib']):.2f} GiB probed headroom)",
                flush=True,
            )

    plan = []
    if args.dry_run and probe is not None:
        plan.append({
            "status": "automatic_batch_probe",
            "probe_budget": budget_name(*PROBE_BUDGET),
            "reserve_memory_gib": args.reserve_memory_gib,
            "preview_batch_size": batch_size,
            "command": probe,
        })

    for alit_tokens, refine_tokens in BUDGETS:
        name = budget_name(alit_tokens, refine_tokens)
        output_dir = output_root / name
        latest = output_dir / f"latest_{name}.pt"
        resume = None
        status = "fresh"
        if latest.is_file() and not args.dry_run:
            epoch = checkpoint_epoch(latest, alit_tokens, refine_tokens)
            if epoch >= TARGET_EPOCHS:
                print(f"[{name}] already complete at epoch {epoch:g}; skipping", flush=True)
                plan.append({"budget": name, "status": "complete", "checkpoint": str(latest)})
                continue
            resume = latest
            status = f"resume_epoch_{epoch:g}"

        command = build_train_command(
            config=config,
            data_path=data_path,
            alit_ckpt=alit_ckpt,
            vqgan_ckpt=vqgan_ckpt,
            output_dir=output_dir,
            alit_tokens=alit_tokens,
            refine_tokens=refine_tokens,
            batch_size=batch_size,
            accum_steps=args.accum_steps,
            world_size=len(gpu_ids),
            resume=resume,
            wandb=not args.no_wandb,
        )
        plan.append({"budget": name, "status": status, "checkpoint": str(latest), "command": command})
        if args.dry_run:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] starting ({status})", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        epoch = checkpoint_epoch(latest, alit_tokens, refine_tokens)
        if epoch < TARGET_EPOCHS:
            raise RuntimeError(f"[{name}] stopped at epoch {epoch:g}, expected {TARGET_EPOCHS}")
        print(f"[{name}] complete: {latest}", flush=True)

    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
    else:
        print("All four budget runs completed.", flush=True)


if __name__ == "__main__":
    main()
