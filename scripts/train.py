#!/usr/bin/env python3
"""고정 STEP 6 파이프라인으로 전체 E0 학습을 실행하는 향후 진입점.

STEP 7에서는 실행하지 않는다. 실수로 전체 학습하는 일을 막기 위해 명시적 확인 플래그가 필요하다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_cxr.training import fit_e0, load_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "e0_training.yaml",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="last.pt 등 v2 checkpoint에서 다음 Epoch부터 재개합니다.",
    )
    parser.add_argument(
        "--confirm-full-training",
        action="store_true",
        help="전체 E0 epoch 학습을 의도적으로 실행함을 확인합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_full_training:
        raise SystemExit(
            "전체 E0 학습은 아직 승인되지 않았습니다. 다음 단계에서 사용자 확인 후 "
            "--confirm-full-training을 명시하세요."
        )
    config = load_training_config(args.config, project_root=PROJECT_ROOT)
    result = fit_e0(config, resume_from=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
