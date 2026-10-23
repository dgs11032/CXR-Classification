#!/usr/bin/env python3
"""CPU static or tiny GPU encoder/head validation for E14 without E14 artifact writes."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e14_patch_specialist import gpu_smoke_check, load_e14_config, static_smoke_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e14_rad_dino_patch_specialist.yaml")
    parser.add_argument("--gpu-smoke", action="store_true", help="Load RAD-DINO and encode only a few B/V training images; no E14 cache or results are written.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_e14_config(args.config, project_root=PROJECT_ROOT)
    result = gpu_smoke_check(config, device_request=args.device) if args.gpu_smoke else static_smoke_check(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
