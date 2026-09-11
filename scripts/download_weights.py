#!/usr/bin/env python3
"""Download the exact official checkpoints used by the MoT-ALIT experiment."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import gdown
import requests
from tqdm import tqdm


ALIT_URL = (
    "https://www.dropbox.com/scl/fi/u8jyro5wysttp5mvs6gir/"
    "alit_vqgan_small_quantized_latent.pth?rlkey=1py1s9755gjhdyw5ifd9h2wff&dl=1"
)
VQGAN_GDRIVE_ID = "13S_unB87n6KKuuMdyMnyExW0G1kplTbP"
LPIPS_URL = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"

FILES = {
    "alit": (
        "alit_vqgan_small_quantized_latent.pth",
        "45841d752669dc3c0591f08c853af0734bc20ebe6d8e91f2a48323f3f8f73bed",
    ),
    "vqgan": (
        "vqgan.ckpt",
        "bac193bccc890db48c498e36ac5a6f2b0e067d3eb360669bd4c2250bf1ae4c6d",
    ),
    "lpips": (
        "vgg.pth",
        "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    print(f"verified {path}")


def download_http(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, destination)


def download_gdrive(destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=VQGAN_GDRIVE_ID, output=str(temporary), quiet=False)
    if result is None or not temporary.is_file():
        raise RuntimeError("VQGAN download failed")
    os.replace(temporary, destination)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def ensure(path: Path, expected: str, downloader, check_only: bool) -> None:
    if path.is_file():
        verify(path, expected)
        return
    if check_only:
        raise FileNotFoundError(path)
    downloader(path)
    verify(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--lpips-path",
        type=Path,
        default=Path("third_party/llamagen/tokenizer/tokenizer_image/cache/vgg.pth"),
    )
    parser.add_argument(
        "--alit-lpips-path",
        type=Path,
        default=Path("third_party/alit/base_tokenizers/taming/modules/autoencoder/lpips/vgg.pth"),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    alit_name, alit_hash = FILES["alit"]
    vqgan_name, vqgan_hash = FILES["vqgan"]
    _lpips_name, lpips_hash = FILES["lpips"]
    ensure(args.weights_dir / alit_name, alit_hash, lambda p: download_http(ALIT_URL, p), args.check_only)
    ensure(
        args.weights_dir / vqgan_name,
        vqgan_hash,
        download_gdrive,
        args.check_only,
    )
    ensure(args.lpips_path, lpips_hash, lambda p: download_http(LPIPS_URL, p), args.check_only)
    ensure(args.alit_lpips_path, lpips_hash, lambda p: copy_file(args.lpips_path, p), args.check_only)


if __name__ == "__main__":
    main()
