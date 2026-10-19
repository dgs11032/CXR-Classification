"""Configuration loading and reproducible JSON serialization."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a JSON config and resolve filesystem paths against the project root."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    resolved = deepcopy(config)
    resolved["config_path"] = str(path)
    resolved["project_root"] = str(PROJECT_ROOT)
    resolved["data"]["root"] = str(
        _resolve_project_path(data_root if data_root is not None else config["data"]["root"])
    )
    resolved["output_dir"] = str(
        _resolve_project_path(output_dir if output_dir is not None else config["output_dir"])
    )
    return resolved


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with deterministic formatting."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
