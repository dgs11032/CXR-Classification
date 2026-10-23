#!/usr/bin/env python3
"""Run full E12 inverse-frequency weighted focal training only after confirmation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e12_training import fit_e12, load_e12_training_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e12_training.yaml")
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_training: raise SystemExit("Full E12 training is blocked. Pass --confirm-full-training only when intentional.")
    print(json.dumps(fit_e12(load_e12_training_config(args.config, project_root=PROJECT_ROOT)), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
