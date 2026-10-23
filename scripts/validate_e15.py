#!/usr/bin/env python3
"""CPU static or one-batch GPU backward validation for E15; no E15 artifacts are written."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e15_finetune import gpu_backward_smoke, load_e15_config, static_smoke_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e15_rad_dino_last_block_finetune.yaml")
    parser.add_argument("--gpu-backward-smoke", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_e15_config(args.config, project_root=PROJECT_ROOT)
    result = gpu_backward_smoke(config, device_request=args.device) if args.gpu_backward_smoke else static_smoke_check(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
