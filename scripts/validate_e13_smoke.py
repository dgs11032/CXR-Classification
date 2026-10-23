#!/usr/bin/env python3
"""Run E13 static validation, or a tiny no-artifact RAD-DINO encoder smoke check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e13_rad_dino import encoder_smoke_check, load_e13_config, static_smoke_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "e13_rad_dino_linear_probe.yaml",
    )
    parser.add_argument(
        "--encoder-probe",
        action="store_true",
        help="Download/load RAD-DINO and encode only the configured samples per class without writing E13 artifacts.",
    )
    parser.add_argument("--device", default=None, help="Optional device override for --encoder-probe, e.g. cuda.")
    args = parser.parse_args()
    config = load_e13_config(args.config, project_root=PROJECT_ROOT)
    result = encoder_smoke_check(config, device_request=args.device) if args.encoder_probe else static_smoke_check(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
