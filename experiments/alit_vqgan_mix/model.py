"""Official ALIT single-rollout features with a fixed-cardinality spatial Router."""

from __future__ import annotations

import contextlib
import copy
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_official_alit(alit_root: str | Path, vqgan_ckpt: str | Path, alit_ckpt: str | Path, device):
    """Instantiate the official model and load its full EMA, including VQGAN."""
    alit_root = Path(alit_root).resolve()
    vqgan_ckpt = Path(vqgan_ckpt).resolve()
    alit_ckpt = Path(alit_ckpt).resolve()
    if not vqgan_ckpt.is_file():
        raise FileNotFoundError(f"missing base VQGAN checkpoint: {vqgan_ckpt}")
    if not alit_ckpt.is_file():
        raise FileNotFoundError(f"missing ALIT checkpoint: {alit_ckpt}")
    if str(alit_root) not in sys.path:
        sys.path.insert(0, str(alit_root))

    # The official package resolves its YAML files relative to the process cwd.
    with _working_directory(alit_root):
        import adaptive_tokenizers

        model = adaptive_tokenizers.alit_small(
            base_tokenizer_args={
                "id": "vqgan",
                "pretrained_ckpt_path": str(vqgan_ckpt),
                "is_requires_grad": True,
            },
            quantize_latent=True,
            factorize_latent=True,
            train_stage="latent_distillation_pretrain",
        )

    checkpoint = torch.load(alit_ckpt, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("official ALIT checkpoint must be a dictionary")
    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("official ALIT checkpoint has no model/ema state dictionary")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
        if not (key[7:] if key.startswith("module.") else key).startswith("gan_losses.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("gan_losses.")]
    unexpected = [key for key in unexpected if not key.startswith("gan_losses.")]
    if missing or unexpected:
        raise RuntimeError(f"ALIT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device), checkpoint


class FixedCardinalityRouter(nn.Module):
    """MoT CNN Router without a density head because the budget is fixed."""

    def __init__(self, latent_channels=256, hidden_dim=128, depth=3, detach_inputs=True):
        super().__init__()
        self.detach_inputs = bool(detach_inputs)
        self.feat_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.f2d_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.abs_delta_proj = nn.Conv2d(latent_channels, hidden_dim, kernel_size=1)
        self.base_proj = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_dim, 16, 16))
        blocks = []
        for _ in range(int(depth)):
            blocks.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
        self.trunk = nn.Sequential(*blocks)
        self.score_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.normal_(self.score_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score_head.bias)
        for projection in (self.f2d_proj, self.delta_proj, self.abs_delta_proj):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, f_1d, x_base, f_2d):
        if self.detach_inputs:
            f_1d = f_1d.detach()
            x_base = x_base.detach()
            f_2d = f_2d.detach()
        x_low = F.adaptive_avg_pool2d(x_base.float(), (16, 16)).to(dtype=f_1d.dtype)
        delta = f_2d - f_1d
        hidden = (
            self.feat_proj(f_1d)
            + self.f2d_proj(f_2d)
            + self.delta_proj(delta)
            + self.abs_delta_proj(delta.abs())
            + self.base_proj(x_low)
            + self.pos_embed.to(dtype=f_1d.dtype)
        )
        return self.score_head(self.trunk(hidden))


def fixed_topk_ste_mask(logits: torch.Tensor, tokens: int, tau: float):
    """Use exact top-k in the forward pass and a dense sigmoid STE backward."""
    batch, channels, height, width = logits.shape
    if channels != 1 or height * width != 256:
        raise ValueError(f"expected Router logits Bx1x16x16, got {tuple(logits.shape)}")
    tokens = max(0, min(int(tokens), height * width))
    flat = logits.flatten(1)
    hard = torch.zeros_like(flat, dtype=torch.float32)
    if tokens > 0:
        hard.scatter_(1, torch.topk(flat.float(), tokens, dim=1).indices, 1.0)
    hard = hard.view(batch, 1, height, width)

    normalized = (logits.float() - logits.float().mean((2, 3), keepdim=True)) / (
        logits.float().std((2, 3), keepdim=True, unbiased=False) + 1e-6
    )
    ratio = min(max(tokens / float(height * width), 1e-4), 1.0 - 1e-4)
    ratio_bias = math.log(ratio / (1.0 - ratio))
    soft = torch.sigmoid((normalized + ratio_bias) / max(float(tau), 1e-4))
    mask = hard + soft - soft.detach()
    return mask.to(dtype=logits.dtype), hard, soft


class ALITVQGANMix(nn.Module):
    """Run only ALIT's first recurrent rollout and mix its grid with native VQGAN."""

    def __init__(self, alit: nn.Module, router_hidden_dim=128, router_depth=3, router_detach_inputs=True):
        super().__init__()
        self.alit = alit
        self.router = FixedCardinalityRouter(
            latent_channels=alit.base_tokenizer.codebook_emb_dim,
            hidden_dim=router_hidden_dim,
            depth=router_depth,
            detach_inputs=router_detach_inputs,
        )

    def encode_first_rollout(self, images: torch.Tensor):
        z_pre, z_native, vq_loss, token_info = self.alit.base_tokenizer.vqgan.encode(images)
        patch_tokens = self.alit.patch_embed(images)
        image_tokens = torch.cat((z_pre, patch_tokens), dim=1)
        image_tokens = image_tokens.flatten(2).transpose(1, 2)
        image_tokens = F.normalize(image_tokens, dim=-1)

        masked_2d_tokens = self.alit.preprocess_decoder(image_tokens)
        image_tokens, initial_latents = self.alit.preprocess_encoder(image_tokens)
        encoder_input = torch.cat(
            [image_tokens, initial_latents + self.alit.timestep_embedding[0]], dim=1
        )
        encoder_output = self.alit.encoder(self.alit.encoder_ln_pre(encoder_input))
        latent_tokens = self.alit.encoder_ln_post(encoder_output[:, image_tokens.shape[1] :])
        latent_tokens = self.alit.pre_quantizer_mlp(latent_tokens)
        quantized_1d, quant_result = self.alit.quantize(latent_tokens, is_quantize=True)
        decoded_logits = self.alit.decoder(quantized_1d, masked_2d_tokens)
        probabilities = decoded_logits.softmax(dim=-1)
        codebook = self.alit.base_tokenizer.vqgan.quantize.embedding.weight
        z_1d = torch.einsum("bnk,kc->bnc", probabilities, codebook)
        z_1d = z_1d.reshape(images.shape[0], 16, 16, codebook.shape[1]).permute(0, 3, 1, 2).contiguous()
        code_indices = token_info[2].reshape(images.shape[0], 16, 16).long()
        return {
            "z_1d": z_1d,
            "z_2d": z_native,
            "vqgan_quant_loss": vq_loss,
            "alit_quant_loss": quant_result["quantizer_loss"],
            "code_indices": code_indices,
            "latent_tokens": quantized_1d,
        }

    def decode(self, latent_grid: torch.Tensor):
        return self.alit.base_tokenizer.vqgan.decode(latent_grid)

    def forward(self, images: torch.Tensor, refine_tokens=96, router_tau=0.7):
        paths = self.encode_first_rollout(images)
        with torch.no_grad():
            x_base = self.decode(paths["z_1d"].detach())
        logits = self.router(paths["z_1d"], x_base, paths["z_2d"])
        mask, hard_mask, soft_mask = fixed_topk_ste_mask(logits, refine_tokens, router_tau)
        z_mix = (1.0 - mask) * paths["z_1d"] + mask * paths["z_2d"]
        paths.update(
            {
                "x_base": x_base,
                "x_mix": self.decode(z_mix),
                "router_logits": logits,
                "mask": mask,
                "hard_mask": hard_mask,
                "soft_mask": soft_mask,
                "z_mix": z_mix,
            }
        )
        return paths


class MoTAlignedALITVQGANMix(ALITVQGANMix):
    """ALIT/VQGAN mix with independent frozen-1D and trainable-2D encoders.

    TiTok and LlamaGen have separate encoders in MoT. ALIT normally reuses the
    VQGAN encoder both as an input to its 1D branch and for native 2D codes.
    Keeping a frozen snapshot for the former preserves the same ownership split.
    """

    def __init__(self, alit: nn.Module, router_hidden_dim=128, router_depth=3,
                 router_detach_inputs=True, alit_tokens=32):
        if alit_tokens not in (32, 64, 96, 128):
            raise ValueError("MoT-aligned ALIT supports native 32-, 64-, 96-, or 128-token budgets")
        super().__init__(
            alit,
            router_hidden_dim=router_hidden_dim,
            router_depth=router_depth,
            router_detach_inputs=router_detach_inputs,
        )
        self.alit_tokens = int(alit_tokens)
        self.alit.dynamic_halting = False
        self.frozen_1d_vqgan_encoder = copy.deepcopy(alit.base_tokenizer.vqgan.encoder)
        self.frozen_1d_vqgan_encoder.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_1d_vqgan_encoder.eval()
        return self

    def forward(self, images: torch.Tensor, refine_tokens=96, router_tau=0.7):
        paths = self.encode_first_rollout(images)
        # The first decoder call must not cache no-grad casts reused by x_mix.
        with torch.no_grad(), torch.autocast(
            images.device.type, enabled=torch.is_autocast_enabled(images.device.type),
            dtype=torch.get_autocast_dtype(images.device.type), cache_enabled=False,
        ):
            x_base = self.decode(paths["z_1d"].detach())
        logits = self.router(paths["z_1d"], x_base * 2.0 - 1.0, paths["z_2d"])
        mask, hard_mask, soft_mask = fixed_topk_ste_mask(logits, refine_tokens, router_tau)
        z_mix = (1.0 - mask) * paths["z_1d"] + mask * paths["z_2d"]
        x_mix = self.decode(z_mix)
        embedding = self.alit.base_tokenizer.vqgan.quantize.embedding.weight
        if embedding.requires_grad:
            # Match MoT's zero-grad DDP connection for the native quantizer.
            x_mix = x_mix + embedding.float().sum().to(x_mix.dtype) * 0.0
        paths.update(x_base=x_base, x_mix=x_mix, router_logits=logits,
                     mask=mask, hard_mask=hard_mask, soft_mask=soft_mask, z_mix=z_mix)
        return paths

    def encode_first_rollout(self, images: torch.Tensor):
        vqgan = self.alit.base_tokenizer.vqgan
        self.frozen_1d_vqgan_encoder.eval()
        with torch.no_grad():
            z_pre = self.frozen_1d_vqgan_encoder(images)

        z_native_pre = vqgan.encoder(images)
        z_native, vq_loss, token_info = vqgan.quantize(z_native_pre)
        patch_tokens = self.alit.patch_embed(images)
        image_tokens = torch.cat((z_pre, patch_tokens), dim=1)
        image_tokens = image_tokens.flatten(2).transpose(1, 2)
        image_tokens = F.normalize(image_tokens, dim=-1)

        masked_2d_tokens = self.alit.preprocess_decoder(image_tokens)
        image_tokens, initial_latents = self.alit.preprocess_encoder(image_tokens)
        encoder_input = torch.cat(
            [image_tokens, initial_latents + self.alit.timestep_embedding[0]], dim=1
        )
        encoder_output = self.alit.encoder(self.alit.encoder_ln_pre(encoder_input))
        # Official AdaptiveVQGAN.forward recurrence: retain all processed tokens,
        # append the next 32 latent seeds, then apply recursive normalization.
        # Image-context halting stays disabled, matching the 32+96 MoT run.
        for rollout in range(1, self.alit_tokens // initial_latents.shape[1]):
            encoder_output = torch.cat(
                (encoder_output, initial_latents + self.alit.timestep_embedding[rollout]), dim=1
            )
            encoder_output = self.alit.encoder(self.alit.encoder_ln_recursive(encoder_output))
        latent_tokens = self.alit.encoder_ln_post(encoder_output[:, image_tokens.shape[1] :])
        latent_tokens = self.alit.pre_quantizer_mlp(latent_tokens)
        quantized_1d, quant_result = self.alit.quantize(latent_tokens, is_quantize=True)
        if quantized_1d.shape[1] != self.alit_tokens:
            raise RuntimeError(f"expected {self.alit_tokens} ALIT tokens, got {quantized_1d.shape[1]}")
        decoded_logits = self.alit.decoder(quantized_1d, masked_2d_tokens)
        probabilities = decoded_logits.softmax(dim=-1)
        # MoT's mixed reconstruction does not backpropagate into the native
        # quantizer embedding through the 1D adapter path.
        codebook = vqgan.quantize.embedding.weight.detach()
        z_1d = torch.einsum("bnk,kc->bnc", probabilities, codebook)
        z_1d = z_1d.reshape(images.shape[0], 16, 16, codebook.shape[1]).permute(0, 3, 1, 2).contiguous()
        code_indices = token_info[2].reshape(images.shape[0], 16, 16).long()
        return {
            "z_1d": z_1d,
            "z_2d": z_native,
            "vqgan_quant_loss": vq_loss.detach(),
            "alit_quant_loss": quant_result["quantizer_loss"].detach(),
            "code_indices": code_indices,
            "latent_tokens": quantized_1d,
        }


@torch.no_grad()
def grid_mse_gain_target(x_base, x_native, target, tokens=96):
    base_error = (x_base.float() - target.float()).pow(2).mean(dim=1, keepdim=True)
    native_error = (x_native.float() - target.float()).pow(2).mean(dim=1, keepdim=True)
    gain = F.adaptive_avg_pool2d(base_error - native_error, (16, 16)).clamp_min(0.0)
    flat = gain.flatten(1)
    target_mask = torch.zeros_like(flat)
    tokens = max(0, min(int(tokens), flat.shape[1]))
    if tokens > 0:
        target_mask.scatter_(1, torch.topk(flat, tokens, dim=1).indices, 1.0)
    return target_mask.view_as(gain), gain
