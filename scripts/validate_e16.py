#!/usr/bin/env python3
"""CPU static or tiny GPU E15-restore/dual-view smoke validation for E16."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from p1_cxr.e16_fusion import gpu_smoke_check, load_e16_config, static_smoke_check  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "e16_global_local_fusion.yaml")
parser.add_argument("--gpu-smoke", action="store_true")
parser.add_argument("--device", default=None)
args = parser.parse_args()
config = load_e16_config(args.config, project_root=PROJECT_ROOT)
print(json.dumps(gpu_smoke_check(config, device_request=args.device) if args.gpu_smoke else static_smoke_check(config), ensure_ascii=False, indent=2))
