#!/usr/bin/env python3
"""Run E13 RAD-DINO frozen feature extraction and 3-class linear probing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e13_rad_dino import run_e13, load_e13_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "e13_rad_dino_linear_probe.yaml",
    )
    parser.add_argument(
        "--confirm-full-experiment",
        action="store_true",
        help="Required because this downloads/loads RAD-DINO and extracts all train/validation features.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Allow replacing E13's own final model, metrics, and prediction files; feature caches may still be reused.",
    )
    args = parser.parse_args()
    if not args.confirm_full_experiment:
        raise SystemExit(
            "Full E13 execution is blocked. Pass --confirm-full-experiment only when intentional."
        )
    config = load_e13_config(args.config, project_root=PROJECT_ROOT)
    result = run_e13(config, overwrite_results=args.overwrite_results)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
