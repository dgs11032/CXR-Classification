#!/usr/bin/env python3
"""Run a read-only, fixed-validation E15 versus E18 comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e15_e18_analysis import load_config, run_analysis, static_smoke_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e15_e18_readonly_analysis.yaml")
    parser.add_argument("--run", action="store_true", help="Read completed result files and write only a new timestamped derived-analysis directory.")
    parser.add_argument("--timestamp", help="Optional unique UTC-like output suffix for reproducible analysis output naming.")
    args = parser.parse_args()
    config = load_config(args.config, project_root=PROJECT_ROOT)
    if args.run:
        print(json.dumps(run_analysis(config, timestamp=args.timestamp), ensure_ascii=False, indent=2))
    else:
        if args.timestamp is not None:
            raise SystemExit("--timestamp requires --run.")
        print(json.dumps(static_smoke_check(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
