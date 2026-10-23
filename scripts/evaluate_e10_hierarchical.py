#!/usr/bin/env python3
"""Evaluate E6+E9 hard hierarchical routing on one fixed validation stream."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e10_hierarchical import evaluate_e10_hierarchical, load_e10_evaluation_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e10_evaluation.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing only E10's own report/trace files.")
    args = parser.parse_args()
    config = load_e10_evaluation_config(args.config, project_root=PROJECT_ROOT)
    result = evaluate_e10_hierarchical(config, device_request=args.device, overwrite=args.overwrite)
    print(json.dumps(result["console_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
