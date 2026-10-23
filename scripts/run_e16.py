#!/usr/bin/env python3
"""Run E16 frozen-E15 global/local dual-view fusion."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e16_fusion import load_e16_config, run_e16  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e16_global_local_fusion.yaml")
parser.add_argument("--confirm-full-experiment", action="store_true")
parser.add_argument("--overwrite-results", action="store_true")
args = parser.parse_args()
if not args.confirm_full_experiment:
    raise SystemExit("Full E16 execution is blocked. Pass --confirm-full-experiment only when intentional.")
print(json.dumps(run_e16(load_e16_config(args.config, project_root=PROJECT_ROOT), overwrite_results=args.overwrite_results), ensure_ascii=False, indent=2))
