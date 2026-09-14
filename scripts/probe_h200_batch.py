#!/usr/bin/env python3
"""Probe the largest per-GPU batch that preserves a fixed memory reserve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cudaerror: out of memory",
    "outofmemoryerror",
)


def parse_csv_ints(value: str, minimum: int = 1) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < minimum for item in values):
        raise ValueError(f"expected comma-separated integers >= {minimum}, got {value!r}")
    return values


def midpoint(low: int, high: int) -> int:
    return (low + high) // 2


def within_memory_budget(peak_gib: float, total_gib: float, reserve_gib: float) -> bool:
    return peak_gib <= total_gib - reserve_gib


def selected_gpu_total_gib(gpu_ids: list[int]) -> float:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    totals = {}
    for line in process.stdout.splitlines():
        index, total_mib = (item.strip() for item in line.split(",", maxsplit=1))
        totals[int(index)] = float(total_mib) / 1024.0
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in totals]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report GPU IDs: {missing}")
    return min(totals[gpu_id] for gpu_id in gpu_ids)


def peak_reserved_gib(output_dir: Path) -> float:
    records = [json.loads(line) for line in (output_dir / "log.txt").read_text().splitlines()]
    return max(float(record["gpu_reserved_gib"]) for record in records)


def probe_command(args, gpu_ids: list[int], batch_size: int, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        "experiments/alit_vqgan_mix/train_mot_fixed96_5epoch.py",
        "--config",
        str(Path(args.config).resolve()),
        "--data-path",
        str(Path(args.data_path).resolve()),
        "--alit-ckpt",
        str(Path(args.alit_ckpt).resolve()),
        "--vqgan-ckpt",
        str(Path(args.vqgan_ckpt).resolve()),
        "--output-dir",
        str(output_dir),
        "--alit-tokens",
        str(args.alit_tokens),
        "--refine-tokens",
        str(args.refine_tokens),
        "--batch-size",
        str(batch_size),
        "--accum-steps",
        "1",
        "--limit-samples",
        str(batch_size * len(gpu_ids)),
        "--epochs",
        "1",
        "--smoke-steps",
        "1",
        "--smoke-no-save",
        "--router-only-epochs",
        "0",
        "--gan-start-epoch",
        "0",
        "--d-warmup-epochs",
        "0",
        "--audit-gradients",
        "--no-wandb",
        "--log-every",
        "1",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h200_budget_sweep_l1_3_10epoch.yaml")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--alit-ckpt", default="weights/alit_vqgan_small_quantized_latent.pth")
    parser.add_argument("--vqgan-ckpt", default="weights/vqgan.ckpt")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--initial-batch-size", type=int, default=96)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--reserve-memory-gib", type=float, default=8.0)
    parser.add_argument("--alit-tokens", type=int, choices=(32, 64, 96, 128), default=128)
    parser.add_argument("--refine-tokens", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/motalit_h200_batch_probe"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = parse_csv_ints(args.gpus, minimum=0)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU IDs must be distinct")
    if not 1 <= args.min_batch_size <= args.initial_batch_size <= args.max_batch_size:
        raise ValueError("require 1 <= min_batch_size <= initial_batch_size <= max_batch_size")
    if args.refine_tokens < 0 or args.alit_tokens + args.refine_tokens != 128:
        raise ValueError("alit_tokens + refine_tokens must equal 128")
    if args.reserve_memory_gib <= 0:
        raise ValueError("reserve_memory_gib must be positive")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_ids)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment.setdefault("OMP_NUM_THREADS", "4")
    run_root = args.output_root / f"run_{os.getpid()}"

    if args.dry_run:
        print(json.dumps({
            "initial_batch_size": args.initial_batch_size,
            "reserve_memory_gib": args.reserve_memory_gib,
            "budget": f"{args.alit_tokens}+{args.refine_tokens}",
            "command": probe_command(
                args,
                gpu_ids,
                args.initial_batch_size,
                run_root / f"bs{args.initial_batch_size}",
            ),
        }))
        return

    total_gib = selected_gpu_total_gib(gpu_ids)
    if args.reserve_memory_gib >= total_gib:
        raise ValueError(
            f"reserve_memory_gib={args.reserve_memory_gib:g} must be smaller than "
            f"the selected GPU capacity ({total_gib:.2f} GiB)"
        )
    target_peak_gib = total_gib - args.reserve_memory_gib
    lower_safe = 0
    upper_unsafe = None
    batch_size = args.initial_batch_size
    best = None
    tested = set()

    while batch_size not in tested:
        tested.add(batch_size)
        output_dir = run_root / f"bs{batch_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "probe.log"
        command = probe_command(args, gpu_ids, batch_size, output_dir)
        print(f"Probing batch_size={batch_size} on {len(gpu_ids)} GPUs", flush=True)
        with log_path.open("w") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        output = log_path.read_text(errors="replace")

        if process.returncode == 0:
            peak_gib = peak_reserved_gib(output_dir)
            utilization = peak_gib / total_gib
            headroom_gib = total_gib - peak_gib
            if within_memory_budget(peak_gib, total_gib, args.reserve_memory_gib):
                best = {
                    "batch_size": batch_size,
                    "accum_steps": 1,
                    "world_size": len(gpu_ids),
                    "global_batch": batch_size * len(gpu_ids),
                    "alit_tokens": args.alit_tokens,
                    "refine_tokens": args.refine_tokens,
                    "peak_reserved_gib": peak_gib,
                    "gpu_total_gib": total_gib,
                    "memory_headroom_gib": headroom_gib,
                    "reserve_memory_gib": args.reserve_memory_gib,
                    "target_peak_reserved_gib": target_peak_gib,
                    "memory_utilization": utilization,
                    "probe_log": str(log_path),
                }
                lower_safe = max(lower_safe, batch_size)
                print(
                    f"batch_size={batch_size} is safe: {peak_gib:.2f}/{total_gib:.2f} GiB "
                    f"reserved, {headroom_gib:.2f} GiB headroom",
                    flush=True,
                )
                if upper_unsafe is not None:
                    if upper_unsafe - lower_safe <= 1:
                        break
                    batch_size = midpoint(lower_safe, upper_unsafe)
                else:
                    next_batch = min(
                        args.max_batch_size,
                        max(batch_size + 1, batch_size * 5 // 4),
                    )
                    if next_batch == batch_size:
                        break
                    batch_size = next_batch
                continue

            upper_unsafe = batch_size if upper_unsafe is None else min(upper_unsafe, batch_size)
            print(
                f"batch_size={batch_size} ran but leaves only {headroom_gib:.2f} GiB; "
                f"need {args.reserve_memory_gib:.2f} GiB",
                flush=True,
            )
            if lower_safe:
                if upper_unsafe - lower_safe <= 1:
                    break
                batch_size = midpoint(lower_safe, upper_unsafe)
            else:
                if batch_size <= args.min_batch_size:
                    break
                batch_size = max(args.min_batch_size, batch_size // 2)
            continue

        lowered = output.lower()
        if not any(marker in lowered for marker in OOM_MARKERS):
            tail = "\n".join(output.splitlines()[-40:])
            raise RuntimeError(f"batch probe failed for a non-OOM reason; inspect {log_path}\n{tail}")
        upper_unsafe = batch_size if upper_unsafe is None else min(upper_unsafe, batch_size)
        print(f"batch_size={batch_size} OOM; reducing in a fresh process", flush=True)
        if lower_safe:
            if upper_unsafe - lower_safe <= 1:
                break
            batch_size = midpoint(lower_safe, upper_unsafe)
        else:
            if batch_size <= args.min_batch_size:
                break
            batch_size = max(args.min_batch_size, batch_size // 2)

    if best is None:
        raise RuntimeError("even the minimum batch size cannot preserve the requested memory reserve")
    result_path = args.output_root / "recommended.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    print(f"Use these overrides for formal training: --batch-size {best['batch_size']} --accum-steps 1")


if __name__ == "__main__":
    main()
