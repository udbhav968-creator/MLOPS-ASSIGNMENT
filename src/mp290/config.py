"""Settings for the whole pipeline, read from params.yaml with yaml.safe_load."""

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: Path | str | None = None) -> dict:
    """Load params.yaml (repository root by default)."""
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(relative: str) -> Path:
    """Turn a path from params.yaml into an absolute path inside the repo."""
    return ROOT / relative


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's raw bytes (logged as data_sha256 in every MLflow run)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def feature_lists(params: dict) -> tuple[list[str], list[str], str]:
    """Return (numeric features, categorical features, target)."""
    f = params["features"]
    return f["numeric"], f["categorical"], f["target"]
