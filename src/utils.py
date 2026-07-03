"""Shared utility helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logger(name: str, log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a module logger with optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def normalize_team_name(name: str | float | None) -> str:
    if name is None:
        return ""
    if isinstance(name, float) and pd.isna(name):
        return ""
    text = name if isinstance(name, str) else str(name)
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return " ".join(text.strip().split())


def outcome_to_label(outcome: int) -> str:
  mapping = {0: "team_a_win", 1: "draw", 2: "team_b_win"}
  return mapping.get(outcome, "unknown")


def ensure_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)
    return frame
