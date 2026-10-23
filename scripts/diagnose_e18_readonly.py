#!/usr/bin/env python3
"""Read-only diagnosis of completed E18 results; writes no files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.e15_e18_analysis import load_config as load_analysis_config  # noqa: E402
from p1_cxr.e18_readonly_diagnosis import run_diagnosis, static_smoke_check  # noqa: E402
from p1_cxr.e18_residual_fusion import load_e18_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e18-config", type=Path, default=PROJECT_ROOT / "configs" / "e18_frozen_rad_dino_cnn_residual_fusion.yaml")
    parser.add_argument("--analysis-config", type=Path, default=PROJECT_ROOT / "configs" / "e15_e18_readonly_analysis.yaml")
    parser.add_argument("--run", action="store_true", help="Read completed E15/E18 results and print diagnostics to stdout only.")
    args = parser.parse_args()
    e18_config = load_e18_config(args.e18_config, project_root=PROJECT_ROOT)
    analysis_config = load_analysis_config(args.analysis_config, project_root=PROJECT_ROOT)
    result = run_diagnosis(e18_config, analysis_config) if args.run else static_smoke_check(e18_config, analysis_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
