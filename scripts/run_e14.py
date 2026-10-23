#!/usr/bin/env python3
"""Run E14 RAD-DINO frozen spatial-patch Bacterial/Viral specialist."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e14_patch_specialist import load_e14_config, run_e14  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e14_rad_dino_patch_specialist.yaml")
    parser.add_argument("--confirm-full-experiment", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true", help="Replace only E14 final outputs; valid E14 patch caches may be reused.")
    args = parser.parse_args()
    if not args.confirm_full_experiment:
        raise SystemExit("Full E14 execution is blocked. Pass --confirm-full-experiment only when intentional.")
    config = load_e14_config(args.config, project_root=PROJECT_ROOT)
    print(json.dumps(run_e14(config, overwrite_results=args.overwrite_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
