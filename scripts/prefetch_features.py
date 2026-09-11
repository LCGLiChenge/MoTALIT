#!/usr/bin/env python3
"""Populate the pretrained feature-model caches before an offline training run."""

from __future__ import annotations

import os
from pathlib import Path

import open_clip
import torch
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights/torch"))


def main() -> None:
    models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch.hub.load(
        str(ROOT / "third_party/dinov2"),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
    cfg = open_clip.get_pretrained_cfg("ViT-B-32", "openai")
    if cfg is None:
        raise RuntimeError("open_clip does not provide ViT-B-32/openai")
    checkpoint = open_clip.pretrained.download_pretrained(
        cfg,
        prefer_hf_hub=False,
        cache_dir=str(ROOT / "weights/open_clip"),
    )
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=checkpoint,
        weights_only=False,
    )
    print("feature-model caches are ready")


if __name__ == "__main__":
    main()

