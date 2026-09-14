"""Train fixed-budget ALIT/VQGAN refinement with the MoT five-epoch schedule."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.alit_vqgan_mix.model import (
    MoTAlignedALITVQGANMix,
    grid_mse_gain_target,
    load_official_alit,
)
from train_titok_llamagen_recon import (
    ProjectedConvNeXtDiscriminator,
    TrainableEMA,
    adamw_param_groups,
    build_clip_feature_loss,
    build_dino_feature_loss,
    build_perceptual_loss,
    compute_lecam_loss,
    compute_perceptual_loss,
    discriminator_feature_matching_loss,
    exclude_from_weight_decay,
    gan_g_loss_from_logits,
    hinge_d_loss,
    make_transform,
    set_requires_grad,
    weighted_logits_mean,
)


def distributed_setup():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
    return False, 0, 1, torch.device("cuda")


def autocast_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported mixed precision: {name}")


def cycle(loader, sampler=None, start_microbatch=0):
    epoch, skip = divmod(start_microbatch, len(loader))
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if index >= skip:
                yield batch
        skip = 0
        epoch += 1


def optimizer_steps_per_epoch(loader_steps, accum_steps):
    return max(1, int(loader_steps) // max(1, int(accum_steps)))


def epoch_fraction_to_steps(value, steps_per_epoch, add_one_after=False):
    if float(value) <= 0.0:
        return 0
    steps = max(1, int(round(float(value) * steps_per_epoch)))
    return steps + (1 if add_one_after else 0)


def cosine_lr(step, max_steps, base_lr, min_lr=0.0, warmup_steps=0):
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_role_lrs(optimizer, args, step):
    role_lrs = {
        "adapter": args.lr,
        "vqgan_encoder": args.lr_vqgan_encoder,
        "vqgan_codebook": args.lr_vqgan_encoder,
        "vqgan_decoder": args.lr_vqgan_decoder,
        "router": args.lr_router,
    }
    for group in optimizer.param_groups:
        base_lr = role_lrs[group["lr_role"]]
        group["lr"] = cosine_lr(step, args.max_steps, base_lr, args.min_lr, args.warmup_steps)


def set_module_trainable(module, enabled):
    if module is not None:
        module.requires_grad_(bool(enabled))


def configure_full_trainability(core):
    core.requires_grad_(False)
    set_module_trainable(core.alit.decoder, True)
    core.alit.decoder_positional_embedding.requires_grad_(True)
    core.alit.decoder_class_embedding.requires_grad_(True)
    core.alit.decoder_mask_token.requires_grad_(True)
    vqgan = core.alit.base_tokenizer.vqgan
    set_module_trainable(vqgan.encoder, True)
    set_module_trainable(vqgan.quantize, True)
    set_module_trainable(vqgan.decoder, True)
    set_module_trainable(core.router, True)
    core.frozen_1d_vqgan_encoder.eval().requires_grad_(False)


def apply_phase_trainability(core, router_only):
    configure_full_trainability(core)
    if router_only:
        set_module_trainable(core.alit.decoder, False)
        core.alit.decoder_positional_embedding.requires_grad_(False)
        core.alit.decoder_class_embedding.requires_grad_(False)
        core.alit.decoder_mask_token.requires_grad_(False)
        vqgan = core.alit.base_tokenizer.vqgan
        set_module_trainable(vqgan.encoder, False)
        set_module_trainable(vqgan.quantize, False)
        set_module_trainable(vqgan.decoder, False)


def set_mot_train_mode(core, router_only):
    core.eval()
    core.router.train()
    if not router_only:
        core.alit.decoder.train()
        vqgan = core.alit.base_tokenizer.vqgan
        vqgan.encoder.train()
        vqgan.quantize.train()
        vqgan.decoder.train()
    core.frozen_1d_vqgan_encoder.eval()


def optimizer_groups(core, args):
    buckets = {
        "adapter": [],
        "vqgan_encoder": [],
        "vqgan_codebook": [],
        "vqgan_decoder": [],
        "router": [],
    }
    for name, parameter in core.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("router."):
            role = "router"
        elif name.startswith("alit.base_tokenizer.vqgan.encoder."):
            role = "vqgan_encoder"
        elif name.startswith("alit.base_tokenizer.vqgan.quantize."):
            role = "vqgan_codebook"
        elif name.startswith("alit.base_tokenizer.vqgan.decoder."):
            role = "vqgan_decoder"
        else:
            role = "adapter"
        buckets[role].append((name, parameter))

    groups = []
    role_lrs = {
        "adapter": args.lr,
        "vqgan_encoder": args.lr_vqgan_encoder,
        "vqgan_codebook": args.lr_vqgan_encoder,
        "vqgan_decoder": args.lr_vqgan_decoder,
        "router": args.lr_router,
    }
    role_wd = {
        "adapter": args.weight_decay,
        "vqgan_encoder": 0.0,
        "vqgan_codebook": 0.0,
        "vqgan_decoder": 0.0,
        "router": 0.0,
    }
    for role, named_parameters in buckets.items():
        no_decay = []
        decay = []
        for name, parameter in named_parameters:
            target = no_decay if exclude_from_weight_decay(name, parameter) else decay
            target.append(parameter)
        for parameters, weight_decay in ((no_decay, 0.0), (decay, role_wd[role])):
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "weight_decay": weight_decay,
                        "lr": role_lrs[role],
                        "lr_role": role,
                    }
                )
    return groups


def collect_trainable_state(core):
    return {
        name: parameter.detach().cpu()
        for name, parameter in core.named_parameters()
        if name in core.mot_trainable_names
    }


def load_trainable_state(core, state, strict=True):
    parameters = dict(core.named_parameters())
    unexpected = [name for name in state if name not in parameters]
    missing = [name for name, parameter in parameters.items() if parameter.requires_grad and name not in state]
    if strict and (missing or unexpected):
        raise RuntimeError(f"resume mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, value in state.items():
            if name in parameters:
                parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def build_projected_discriminator(args, device):
    return ProjectedConvNeXtDiscriminator(
        input_size=args.projected_input_size,
        hidden_channels=args.projected_head_hidden,
        backbone_name=args.projected_backbone,
        head_depth=args.projected_head_depth,
        loss_weights=args.projected_loss_weights,
        pretrained=args.projected_pretrained,
    ).to(device)


def reconstruction_loss(args, l1_loss, lpips_loss):
    """Keep pixel and perceptual coefficients independently configurable."""
    return (
        args.lambda_l1 * l1_loss
        + args.lambda_mix * args.lambda_lpips * lpips_loss
    )


def reduce_metrics(metrics, device, world_size):
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values)
        values /= world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def save_checkpoint(path, core, discriminator, optimizer, optimizer_d, ema, step, epoch, args, full_state,
                    lecam_real=None, lecam_fake=None):
    payload = {
        "step": int(step),
        "epoch": float(epoch),
        "model": collect_trainable_state(core),
        "model_ema": ema.state_dict(),
        "discriminator": discriminator.state_dict(),
        "config": vars(args),
        "source_alit_ckpt": str(Path(args.alit_ckpt).resolve()),
        "source_vqgan_ckpt": str(Path(args.vqgan_ckpt).resolve()),
        "mot_aligned": True,
        "lecam_real": lecam_real.detach().cpu() if lecam_real is not None else torch.tensor(0.0),
        "lecam_fake": lecam_fake.detach().cpu() if lecam_fake is not None else torch.tensor(0.0),
    }
    if full_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["optimizer_d"] = optimizer_d.state_dict()
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def init_wandb(args, is_main):
    if not is_main or not args.wandb:
        return None
    from experiments.alit_vqgan_mix.wandb_logging import start_wandb

    return start_wandb(args.output_dir, args.wandb_project, args.wandb_name, vars(args))


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.native_decode_chunk < 1:
        raise ValueError("native_decode_chunk must be positive")
    distributed, rank, world_size, device = distributed_setup()
    is_main = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = make_transform(args.image_size, args.random_crop, args.random_flip)
    dataset = ImageFolder(args.data_path, transform=transform)
    if args.limit_samples > 0:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if not len(loader):
        raise ValueError("dataset must contain at least one full batch per rank")
    steps_per_epoch = optimizer_steps_per_epoch(len(loader), args.accum_steps)
    if args.max_steps <= 0:
        args.max_steps = epoch_fraction_to_steps(args.epochs, steps_per_epoch)
    args.steps_per_epoch = steps_per_epoch
    args.router_only_steps = epoch_fraction_to_steps(args.router_only_epochs, steps_per_epoch)
    args.gan_start_step = epoch_fraction_to_steps(
        args.gan_start_epoch, steps_per_epoch, add_one_after=True
    )
    args.d_warmup_steps = epoch_fraction_to_steps(args.d_warmup_epochs, steps_per_epoch)

    official_alit, source_checkpoint = load_official_alit(
        args.alit_root, args.vqgan_ckpt, args.alit_ckpt, device
    )
    source_state_name = "ema" if "ema" in source_checkpoint else "model"
    del source_checkpoint
    core = MoTAlignedALITVQGANMix(
        official_alit,
        router_hidden_dim=args.router_hidden_dim,
        router_depth=args.router_depth,
        router_detach_inputs=args.router_detach_inputs,
        alit_tokens=args.alit_tokens,
    ).to(device)
    configure_full_trainability(core)
    core.mot_trainable_names = {name for name, p in core.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(optimizer_groups(core, args), betas=(0.9, 0.999))
    ema = TrainableEMA(core, decay=args.ema_decay)

    discriminator = build_projected_discriminator(args, device)
    optimizer_d = torch.optim.AdamW(
        adamw_param_groups(discriminator, 0.0), lr=args.lr_d, betas=(0.5, 0.9)
    )
    perceptual = build_perceptual_loss(args, device)
    dino_loss_module = build_dino_feature_loss(args, device)
    clip_loss_module = build_clip_feature_loss(args, device)

    start_step = 0
    lecam_real = torch.zeros((), device=device)
    lecam_fake = torch.zeros((), device=device)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key, default in (("alit_tokens", 32), ("refine_tokens", 96)):
            if int(resume["config"].get(key, default)) != getattr(args, key):
                raise ValueError(f"resume changes {key}; use the matching experiment config")
        load_trainable_state(core, resume["model"], strict=True)
        discriminator.load_state_dict(resume["discriminator"], strict=True)
        start_step = int(resume.get("step", 0))
        if int(resume["config"]["steps_per_epoch"]) != steps_per_epoch:
            raise ValueError("resume requires the same global batch to preserve the data cursor")
        lecam_real = resume["lecam_real"].to(device)
        lecam_fake = resume["lecam_fake"].to(device)
        if not args.reset_optimizer:
            optimizer.load_state_dict(resume["optimizer"])
            optimizer_d.load_state_dict(resume["optimizer_d"])
        if "model_ema" in resume:
            ema.load_state_dict(resume["model_ema"], device=device)
        if is_main:
            print(f"resumed {args.resume} at step {start_step}", flush=True)
        del resume
    data_iter = cycle(loader, sampler, start_microbatch=start_step * args.accum_steps)
    stop_step = args.max_steps
    if args.stop_after_epoch > 0:
        stop_step = min(stop_step, epoch_fraction_to_steps(args.stop_after_epoch, steps_per_epoch))
    if args.smoke_steps > 0:
        stop_step = min(stop_step, start_step + args.smoke_steps)
    if stop_step <= start_step:
        raise ValueError(f"stop_step {stop_step} must exceed resume step {start_step}")

    trainable_count = sum(parameter.numel() for parameter in core.parameters() if parameter.requires_grad)
    if is_main:
        print(
            f"dataset={len(dataset)} world_size={world_size} batch={args.batch_size} accum={args.accum_steps} "
            f"global_batch={args.batch_size * args.accum_steps * world_size} trainable={trainable_count:,} "
            f"ALIT_state={source_state_name} alit_tokens={args.alit_tokens} fixed_tokens={args.refine_tokens} "
            f"steps_per_epoch={steps_per_epoch} max_steps={args.max_steps} "
            f"router_only_steps={args.router_only_steps} gan_start_step={args.gan_start_step} "
            f"d_warmup_steps={args.d_warmup_steps}",
            flush=True,
        )

    if distributed:
        core = DDP(core, device_ids=[device.index], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False)
    model = core
    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    latest_path = out_dir / args.latest_filename
    log_path = out_dir / "log.txt"
    wandb_run = init_wandb(args, is_main)
    pbar = tqdm(
        range(start_step + 1, stop_step + 1),
        desc=f"Epoch {start_step / steps_per_epoch:.2f}/{args.epochs:g}",
        disable=not is_main,
        dynamic_ncols=True,
        mininterval=2.0,
    )
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    d_warmup_end_step = args.gan_start_step + max(args.d_warmup_steps, 0) - 1

    for step in pbar:
        model_core = model.module if distributed else model
        router_only = args.router_only_steps > 0 and step <= args.router_only_steps
        apply_phase_trainability(model_core, router_only)
        set_mot_train_mode(model_core, router_only)
        model_core.alit.dynamic_halting = False
        set_optimizer_role_lrs(optimizer, args, step)
        for group in optimizer.param_groups:
            if router_only and group["lr_role"] != "router":
                group["lr"] = 0.0
        d_active = not router_only and step >= args.gan_start_step
        d_warmup_active = d_active and step <= d_warmup_end_step
        g_gan_active = d_active and not d_warmup_active
        for group in optimizer_d.param_groups:
            group["lr"] = args.lr_d if d_active else 0.0
        discriminator.train(d_active)
        optimizer.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        totals = {
            key: 0.0
            for key in (
                "loss", "l1", "lpips", "dino", "clip", "quant", "router_aux", "router_budget",
                "router_binary", "gan_raw", "gan", "disc_fm", "d_loss", "lecam", "real", "fake",
                "sep", "psnr", "mask", "mask_std", "router_soft", "gain",
                "gan_input_grad_rms", "gan_recon_grad_pct",
            )
        }
        for micro_step in range(args.accum_steps):
            images, _ = next(data_iter)
            images = images.to(device, non_blocking=True)
            sync = micro_step + 1 == args.accum_steps
            model_for_forward = model_core if router_only else model
            model_context = model.no_sync() if distributed and not router_only and not sync else contextlib.nullcontext()
            disc_context = discriminator.no_sync() if distributed and not sync else contextlib.nullcontext()
            with model_context, disc_context, torch.autocast(
                "cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"
            ):
                outputs = model_for_forward(images, refine_tokens=args.refine_tokens, router_tau=args.router_tau)
                x_mix = outputs["x_mix"]
                with torch.no_grad():
                    native_latents = outputs["z_2d"].detach()
                    x_native = torch.cat(
                        [model_core.decode(chunk) for chunk in native_latents.split(args.native_decode_chunk)],
                        dim=0,
                    )
                    gain_target, gain_score = grid_mse_gain_target(
                        outputs["x_base"], x_native, images, tokens=args.refine_tokens
                    )

                pred_11 = x_mix * 2.0 - 1.0
                target_11 = images * 2.0 - 1.0
                l1_loss = F.l1_loss(pred_11.float(), target_11.float())
                lpips_loss = compute_perceptual_loss(perceptual, "lpips", pred_11, target_11)
                dino_loss = dino_loss_module(x_mix, images, pred_range="zero_1", target_range="zero_1")
                clip_loss = clip_loss_module(x_mix, images, pred_range="zero_1", target_range="zero_1")
                router_aux = F.binary_cross_entropy_with_logits(outputs["router_logits"].float(), gain_target.float())
                router_budget = (outputs["soft_mask"].float().mean() - args.refine_tokens / 256.0).pow(2)
                router_binary = (outputs["soft_mask"].float() * (1.0 - outputs["soft_mask"].float())).mean()
                quant_loss = outputs["alit_quant_loss"].float()

                reconstruction = reconstruction_loss(args, l1_loss, lpips_loss)
                gan_input_grad_rms = 0.0
                gan_recon_grad_pct = 0.0
                gan_raw = x_mix.new_zeros(())
                disc_fm = x_mix.new_zeros(())
                if g_gan_active:
                    if args.audit_gradients:
                        rec_grad = torch.autograd.grad(reconstruction, x_mix, retain_graph=True)[0].float()
                        rec_grad_norm = rec_grad.norm().clamp_min(1e-12)
                        del rec_grad
                    set_requires_grad(discriminator, False)
                    fake_logits_for_g = discriminator(x_mix.float())
                    gan_raw = gan_g_loss_from_logits(fake_logits_for_g)
                    if args.audit_gradients:
                        gan_grad = torch.autograd.grad(args.lambda_gan * gan_raw, x_mix, retain_graph=True)[0].float()
                        gan_input_grad_rms = float(gan_grad.square().mean().sqrt())
                        gan_recon_grad_pct = float(100.0 * gan_grad.norm() / rec_grad_norm)
                        if not math.isfinite(gan_input_grad_rms) or gan_input_grad_rms <= 0.0:
                            raise RuntimeError("GAN does not provide finite nonzero reconstruction gradients")
                        del gan_grad, rec_grad_norm
                    disc_fm = discriminator_feature_matching_loss(discriminator, x_mix.float(), images.float())
                loss = (
                    reconstruction
                    + args.lambda_dino_feat * dino_loss
                    + args.lambda_clip_feat * clip_loss
                    + args.lambda_alit_quant * quant_loss
                    + args.lambda_router_aux * router_aux
                    + args.lambda_router_budget * router_budget
                    + args.lambda_router_binary * router_binary
                    + args.lambda_disc_feature_matching * disc_fm
                    + args.lambda_gan * gan_raw
                )
                (loss / args.accum_steps).backward()

            d_loss = x_mix.new_zeros(())
            lecam_loss = x_mix.new_zeros(())
            real_mean = x_mix.new_zeros(())
            fake_mean = x_mix.new_zeros(())
            if d_active and step % args.d_every == 0:
                set_requires_grad(discriminator, True)
                with torch.autocast(
                    "cuda", dtype=autocast_dtype(args.mixed_precision), enabled=args.mixed_precision != "none"
                ):
                    logits_real = discriminator(images.detach().float())
                    logits_fake = discriminator(x_mix.detach().float())
                    real_mean = weighted_logits_mean(logits_real)
                    fake_mean = weighted_logits_mean(logits_fake)
                    d_loss = hinge_d_loss(logits_real, logits_fake)
                    if args.lecam_regularization_weight > 0.0:
                        lecam_loss = compute_lecam_loss(real_mean, fake_mean, lecam_real, lecam_fake)
                        d_loss = d_loss + args.lecam_regularization_weight * lecam_loss
                (d_loss / args.accum_steps).backward()
                lecam_real = lecam_real * args.lecam_ema_decay + real_mean.detach() * (1.0 - args.lecam_ema_decay)
                lecam_fake = lecam_fake * args.lecam_ema_decay + fake_mean.detach() * (1.0 - args.lecam_ema_decay)

            with torch.no_grad():
                mse = (x_mix.float().clamp(0, 1) - images.float()).pow(2).flatten(1).mean(1)
                psnr = (-10.0 * torch.log10(mse.clamp_min(1e-12))).mean()
                hard_counts = outputs["hard_mask"].float().flatten(1).sum(1)
                weighted_gan = args.lambda_gan * gan_raw
                values = {
                    "loss": loss, "l1": l1_loss, "lpips": lpips_loss, "dino": dino_loss,
                    "clip": clip_loss, "quant": quant_loss, "router_aux": router_aux,
                    "router_budget": router_budget, "router_binary": router_binary,
                    "gan_raw": gan_raw, "gan": weighted_gan,
                    "disc_fm": args.lambda_disc_feature_matching * disc_fm, "d_loss": d_loss,
                    "lecam": lecam_loss, "real": real_mean, "fake": fake_mean,
                    "sep": real_mean - fake_mean, "psnr": psnr, "mask": hard_counts.mean(),
                    "mask_std": hard_counts.std(unbiased=False), "router_soft": outputs["soft_mask"].float().mean(),
                    "gain": gain_score.mean(),
                }
                for key, value in values.items():
                    totals[key] += float(value.detach())
                totals["gan_input_grad_rms"] += gan_input_grad_rms
                totals["gan_recon_grad_pct"] += gan_recon_grad_pct
                if args.audit_gradients and not torch.all(hard_counts == args.refine_tokens):
                    raise RuntimeError("hard Router budget changed")

        if distributed and router_only:
            for parameter in model_core.router.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_core.parameters(), args.max_grad_norm,
                                                  error_if_nonfinite=True)
        if args.audit_gradients:
            for name, parameter in model_core.named_parameters():
                if name not in model_core.mot_trainable_names and parameter.grad is not None:
                    raise RuntimeError(f"frozen 1D parameter received a gradient: {name}")
            roles_without_expected_grad = {"vqgan_codebook"}
            if args.refine_tokens == 0:
                roles_without_expected_grad.add("vqgan_encoder")
            for group in optimizer.param_groups:
                if router_only and group["lr_role"] != "router":
                    if any(p.grad is not None for p in group["params"]):
                        raise RuntimeError("non-Router gradient during Router-only")
                elif group["lr_role"] not in roles_without_expected_grad:
                    if not any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in group["params"]):
                        raise RuntimeError(f"missing gradients for {group['lr_role']}")
        optimizer.step()
        ema.update(model_core)
        d_grad_norm = 0.0
        if d_active and step % args.d_every == 0:
            d_parameters = [parameter for parameter in discriminator.parameters() if parameter.grad is not None]
            d_grad_norm = float(torch.nn.utils.clip_grad_norm_(d_parameters, args.max_grad_norm)) if d_parameters else 0.0
            optimizer_d.step()

        metrics = {key: value / args.accum_steps for key, value in totals.items()}
        metrics["grad"] = float(grad_norm)
        metrics["d_grad"] = d_grad_norm
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["lr_d"] = optimizer_d.param_groups[0]["lr"]
        metrics["epoch"] = step / steps_per_epoch
        metrics["phase_router_only"] = float(router_only)
        metrics["phase_d_warmup"] = float(d_warmup_active)
        metrics["phase_joint_gan"] = float(g_gan_active)
        metrics["gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        metrics["gpu_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        reconstruction_value = (
            args.lambda_l1 * metrics["l1"]
            + args.lambda_mix * args.lambda_lpips * metrics["lpips"]
        )
        metrics["gr_pct"] = 100.0 * abs(metrics["gan"]) / max(abs(reconstruction_value), 1e-8)
        metrics = reduce_metrics(metrics, device, world_size)

        if is_main:
            if (step - 1) % steps_per_epoch == 0 or step == start_step + 1:
                pbar.set_description(f"Epoch {(step - 1) // steps_per_epoch + 1}/{args.epochs:g}", refresh=False)
            pbar.set_postfix(
                psnr=f"{metrics['psnr']:.2f}", l1=f"{metrics['l1']:.3f}", lp=f"{metrics['lpips']:.3f}",
                mask=f"{metrics['mask']:.0f}", sep=f"{metrics['sep']:.3f}", gr=f"{metrics['gr_pct']:.1f}%",
                epoch=f"{metrics['epoch']:.2f}",
                refresh=False,
            )
            if step % args.log_every == 0 or step == start_step + 1:
                record = {"step": step, "elapsed": time.time() - start_time, **metrics}
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

            epoch_end = step % steps_per_epoch == 0
            if not args.smoke_no_save and args.latest_every > 0 and step % args.latest_every == 0:
                save_checkpoint(
                    latest_path, model_core,
                    discriminator.module if distributed else discriminator,
                    optimizer, optimizer_d, ema, step, step / steps_per_epoch, args, True, lecam_real, lecam_fake,
                )
            if epoch_end:
                epoch_index = step // steps_per_epoch
                if not args.smoke_no_save:
                    if args.save_epoch_every > 0 and epoch_index % args.save_epoch_every == 0:
                        save_checkpoint(
                            out_dir / f"epoch_{epoch_index:04d}.pt", model_core,
                            discriminator.module if distributed else discriminator,
                            optimizer, optimizer_d, ema, step, float(epoch_index), args, True, lecam_real, lecam_fake,
                        )
                    save_checkpoint(
                        latest_path, model_core,
                        discriminator.module if distributed else discriminator,
                        optimizer, optimizer_d, ema, step, float(epoch_index), args, True, lecam_real, lecam_fake,
                    )
        if distributed and (step % steps_per_epoch == 0 or (args.latest_every > 0 and step % args.latest_every == 0)):
            dist.barrier()

    if is_main:
        if not args.smoke_no_save:
            save_checkpoint(
                latest_path, model.module if distributed else model,
                discriminator.module if distributed else discriminator,
                optimizer, optimizer_d, ema, stop_step, stop_step / steps_per_epoch, args, True, lecam_real, lecam_fake,
            )
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", required=True)
    known, _ = bootstrap.parse_known_args()
    with open(known.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    add = parser.add_argument
    add("--data-path", default=config.get("data_path"), required=config.get("data_path") is None)
    add("--output-dir", default=config.get("output_dir", "results/alit_vqgan_mix_fixed96"))
    add("--alit-root", default=config.get("alit_root", "third_party/alit"))
    add("--alit-ckpt", default=config.get("alit_ckpt"), required=config.get("alit_ckpt") is None)
    add("--vqgan-ckpt", default=config.get("vqgan_ckpt"), required=config.get("vqgan_ckpt") is None)
    add("--resume", default=config.get("resume", ""))
    add("--reset-optimizer", action=argparse.BooleanOptionalAction, default=config.get("reset_optimizer", False))
    add("--image-size", type=int, default=config.get("image_size", 256))
    add("--batch-size", type=int, default=config.get("batch_size", 1))
    add("--accum-steps", type=int, default=config.get("accum_steps", 4))
    add("--epochs", type=float, default=config.get("epochs", 5.0))
    add("--stop-after-epoch", type=float, default=0.0)
    add("--smoke-steps", type=int, default=0)
    add("--smoke-no-save", action="store_true")
    add("--audit-gradients", action="store_true")
    add("--max-steps", type=int, default=config.get("max_steps", 0))
    add("--num-workers", type=int, default=config.get("num_workers", 8))
    add("--limit-samples", type=int, default=config.get("limit_samples", 0))
    add("--random-crop", action=argparse.BooleanOptionalAction, default=config.get("random_crop", False))
    add("--random-flip", action=argparse.BooleanOptionalAction, default=config.get("random_flip", False))
    add("--mixed-precision", choices=["bf16", "fp16", "none"], default=config.get("mixed_precision", "bf16"))
    add("--seed", type=int, default=config.get("seed", 0))
    add("--alit-tokens", type=int, choices=(32, 64, 96, 128), default=config.get("alit_tokens", 32))
    add("--refine-tokens", type=int, default=config.get("refine_tokens", 96))
    add("--native-decode-chunk", type=int, default=config.get("native_decode_chunk", 4))
    add("--router-hidden-dim", type=int, default=config.get("router_hidden_dim", 128))
    add("--router-depth", type=int, default=config.get("router_depth", 3))
    add("--router-detach-inputs", action=argparse.BooleanOptionalAction, default=config.get("router_detach_inputs", True))
    add("--router-tau", type=float, default=config.get("router_tau", 0.7))
    add("--lr", type=float, default=config.get("lr", 1e-6))
    add("--lr-vqgan-encoder", type=float, default=config.get("lr_vqgan_encoder", 2e-7))
    add("--lr-vqgan-decoder", type=float, default=config.get("lr_vqgan_decoder", 8e-7))
    add("--lr-router", type=float, default=config.get("lr_router", 1e-5))
    add("--lr-d", type=float, default=config.get("lr_d", 5e-5))
    add("--min-lr", type=float, default=config.get("min_lr", 0.0))
    add("--warmup-steps", type=int, default=config.get("warmup_steps", 0))
    add("--weight-decay", type=float, default=config.get("weight_decay", 0.01))
    add("--max-grad-norm", type=float, default=config.get("max_grad_norm", 1.0))
    add("--ema-decay", type=float, default=config.get("ema_decay", 0.999))
    add("--perceptual-loss", default=config.get("perceptual_loss", "lpips"))
    add("--lambda-perceptual", type=float, default=config.get("lambda_perceptual", None))
    add("--lambda-lpips", type=float, default=config.get("lambda_lpips", 0.3))
    add("--lambda-mix", type=float, default=config.get("lambda_mix", 2.0))
    add("--lambda-l1", type=float, default=config.get("lambda_l1", config.get("lambda_mix", 2.0)))
    add("--lambda-dino-feat", type=float, default=config.get("lambda_dino_feat", 0.5))
    add("--lambda-clip-feat", type=float, default=config.get("lambda_clip_feat", 0.5))
    add("--lambda-alit-quant", type=float, default=config.get("lambda_alit_quant", 0.0))
    add("--lambda-router-aux", type=float, default=config.get("lambda_router_aux", 1.0))
    add("--lambda-router-budget", type=float, default=config.get("lambda_router_budget", 20.0))
    add("--lambda-router-binary", type=float, default=config.get("lambda_router_binary", 0.01))
    add("--lambda-disc-feature-matching", type=float, default=config.get("lambda_disc_feature_matching", 0.5))
    add("--lambda-gan", type=float, default=config.get("lambda_gan", 0.12))
    add("--router-only-epochs", type=float, default=config.get("router_only_epochs", 0.5))
    add("--gan-start-epoch", type=float, default=config.get("gan_start_epoch", 1.5))
    add("--d-warmup-epochs", type=float, default=config.get("d_warmup_epochs", 0.01))
    add("--d-every", type=int, default=config.get("d_every", 1))
    add("--lecam-regularization-weight", type=float, default=config.get("lecam_regularization_weight", 0.001))
    add("--lecam-ema-decay", type=float, default=config.get("lecam_ema_decay", 0.999))
    add("--projected-input-size", type=int, default=config.get("projected_input_size", 224))
    add("--projected-head-hidden", type=int, default=config.get("projected_head_hidden", 512))
    add("--projected-head-depth", type=int, default=config.get("projected_head_depth", 4))
    add("--projected-backbone", default=config.get("projected_backbone", "convnext_base"))
    add("--projected-loss-weights", nargs="+", type=float, default=config.get("projected_loss_weights", [1, 1, 1, 1]))
    add("--projected-pretrained", action=argparse.BooleanOptionalAction, default=config.get("projected_pretrained", True))
    add("--dino-repo", default=config.get("dino_repo", "../.cache/torch/hub/facebookresearch_dinov2_main"))
    add("--dino-model", default=config.get("dino_model", "dinov2_vits14"))
    add("--dino-input-size", type=int, default=config.get("dino_input_size", 224))
    add("--dino-feat-input-size", type=int, default=config.get("dino_feat_input_size", 224))
    add("--dino-feat-use-patch-tokens", action=argparse.BooleanOptionalAction, default=config.get("dino_feat_use_patch_tokens", True))
    add("--dino-feat-loss", default=config.get("dino_feat_loss", "l1"))
    add("--dino-feat-normalize", action=argparse.BooleanOptionalAction, default=config.get("dino_feat_normalize", True))
    add("--clip-model", default=config.get("clip_model", "ViT-B-32"))
    add("--clip-pretrained", default=config.get("clip_pretrained", "openai"))
    add("--clip-cache-dir", default=config.get("clip_cache_dir", "../.cache/open_clip"))
    add("--clip-prefer-hf-hub", action=argparse.BooleanOptionalAction, default=config.get("clip_prefer_hf_hub", False))
    add("--clip-weights-only", action=argparse.BooleanOptionalAction, default=config.get("clip_weights_only", False))
    add("--clip-feat-input-size", type=int, default=config.get("clip_feat_input_size", 224))
    add("--clip-feat-loss", default=config.get("clip_feat_loss", "l1"))
    add("--clip-feat-normalize", action=argparse.BooleanOptionalAction, default=config.get("clip_feat_normalize", True))
    add("--llamagen-root", default=config.get("llamagen_root", "/home/heyefei/lichenge/LlamaGen"))
    add("--latest-every", type=int, default=config.get("latest_every", 200))
    add("--latest-filename", default=config.get("latest_filename", "latest.pt"))
    add("--save-epoch-every", type=int, default=config.get("save_epoch_every", 1))
    add("--log-every", type=int, default=config.get("log_every", 20))
    add("--wandb", action=argparse.BooleanOptionalAction, default=config.get("wandb", False))
    add("--wandb-project", default=config.get("wandb_project", "MoT"))
    add("--wandb-name", default=config.get("wandb_name", "alit_vqgan_mix_fixed96_mot_5epoch"))
    args = parser.parse_args()
    if args.lambda_l1 < 0.0:
        parser.error("--lambda-l1 must be non-negative")
    if not 0 <= args.refine_tokens <= 256:
        parser.error("--refine-tokens must be between 0 and 256")
    if args.save_epoch_every < 0:
        parser.error("--save-epoch-every must be non-negative")
    latest_path = Path(args.latest_filename)
    if latest_path.name != args.latest_filename or latest_path.suffix != ".pt":
        parser.error("--latest-filename must be a basename ending in .pt")
    if args.smoke_no_save and args.smoke_steps <= 0:
        parser.error("--smoke-no-save requires --smoke-steps")
    return args


if __name__ == "__main__":
    try:
        main(parse_args())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
