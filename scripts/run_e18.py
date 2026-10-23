#!/usr/bin/env python3
"""Validate, GPU-smoke-test, or intentionally run E18 residual CNN fusion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e18_residual_fusion import (  # noqa: E402
    gpu_smoke_check,
    load_e18_config,
    run_e18,
    static_smoke_check,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e18_frozen_rad_dino_cnn_residual_fusion.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu-smoke", action="store_true", help="Run strict E15/CNN restoration and two images without artifacts.")
    mode.add_argument("--confirm-full-experiment", action="store_true", help="Run E18 cache extraction, fixed training, and one validation evaluation.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), help="Optional device override for --gpu-smoke only.")
    args = parser.parse_args()
    config = load_e18_config(args.config, project_root=PROJECT_ROOT)
    if args.confirm_full_experiment:
        print(json.dumps(run_e18(config), ensure_ascii=False, indent=2))
    elif args.gpu_smoke:
        print(json.dumps(gpu_smoke_check(config, device_request=args.device), ensure_ascii=False, indent=2))
    else:
        if args.device is not None:
            raise SystemExit("--device can only be used with --gpu-smoke.")
        print(json.dumps(static_smoke_check(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
