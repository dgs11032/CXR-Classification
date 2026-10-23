#!/usr/bin/env python3
"""Run a fresh full E2 auxiliary-subtype experiment after explicit confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e2_training import fit_e2, load_e2_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "e2_training.yaml"
    )
    parser.add_argument("--confirm-full-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_full_training:
        raise SystemExit(
            "Full E2 training is blocked. Pass --confirm-full-training only when intentional."
        )
    config = load_e2_training_config(args.config, project_root=PROJECT_ROOT)
    print(json.dumps(fit_e2(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
