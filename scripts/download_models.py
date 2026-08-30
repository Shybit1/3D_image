#!/usr/bin/env python3
"""
Download pretrained model weights for DepthWizard-X.

Never downloads silently: prints model name, license, size, and target
path before fetching, per the master build spec.
"""
import sys
import urllib.request
from pathlib import Path

MODELS = [
    {
        "name": "MiDaS_small (v2.1)",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
        "license": "MIT (Intel ISL)",
        "approx_size_mb": 82,
        "target": "models/checkpoints/midas_v21_small_256.pt",
    },
]


def download(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> downloading to {target} ...")
    urllib.request.urlretrieve(url, target)
    print(f"  -> done ({target.stat().st_size / 1e6:.1f} MB)")


def main():
    root = Path(__file__).resolve().parents[1]
    for m in MODELS:
        target = root / m["target"]
        print(f"Model: {m['name']}")
        print(f"  License: {m['license']}")
        print(f"  Approx size: {m['approx_size_mb']} MB")
        print(f"  Target path: {target}")
        if target.exists():
            print("  Already present, skipping.")
            continue
        try:
            download(m["url"], target)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(
                "  Automatic download failed. Manual download instructions:\n"
                f"    1. Visit: {m['url']}\n"
                f"    2. Save the file to: {target}\n"
            )
            sys.exit(1)
    print("All models ready.")


if __name__ == "__main__":
    main()
