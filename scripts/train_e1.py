#!/usr/bin/env python3
"""Run full E1 CE + pneumonia subtype-only SupCon training after explicit confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e1_training import fit_e1, load_e1_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e1_training.yaml")
    parser.add_argument("--confirm-full-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_full_training:
        raise SystemExit(
            "Full E1 training is blocked. Pass --confirm-full-training only when intentional."
        )
    config = load_e1_training_config(args.config, project_root=PROJECT_ROOT)
    print(json.dumps(fit_e1(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
