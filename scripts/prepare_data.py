#!/usr/bin/env python3
"""Create the three-class inventory and leakage-aware manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.config import load_config  # noqa: E402
from p1_cxr.data import prepare_manifests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/e0.json")
    parser.add_argument("--data-root", type=Path, help="Override the config data root")
    parser.add_argument("--output-dir", type=Path, help="Override the config output directory")
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Skip PIL verify; dimensions and hashes are still collected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, data_root=args.data_root, output_dir=args.output_dir)
    summaries = prepare_manifests(
        config["data"]["root"],
        config["output_dir"],
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["training"]["seed"]),
        verify_images=not args.skip_image_verify,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
