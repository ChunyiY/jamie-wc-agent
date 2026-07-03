"""Historical international football data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.team_mapping import canonical_team_name
from src.utils import normalize_team_name

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}

OPTIONAL_COLUMNS = {
    "tournament",
    "city",
    "country",
    "neutral",
    "stage",
    "days_since_last_match_home",
    "days_since_last_match_away",
}


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _parse_neutral(series: pd.Series) -> pd.Series:
    def to_bool(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return bool(value)

    return series.map(to_bool)


def load_results_csv(path: Path) -> pd.DataFrame:
    """Load Kaggle-style international results CSV."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["date"] = _parse_date(frame["date"])
    frame = frame.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    frame["home_team"] = frame["home_team"].astype(str).map(canonical_team_name)
    frame["away_team"] = frame["away_team"].astype(str).map(canonical_team_name)
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce").astype(int)
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce").astype(int)

    if "neutral" not in frame.columns:
        frame["neutral"] = False
    frame["neutral"] = _parse_neutral(frame["neutral"])

    if "tournament" not in frame.columns:
        frame["tournament"] = "Friendly"
    frame["tournament"] = frame["tournament"].fillna("Friendly").astype(str)

    if "stage" not in frame.columns:
        frame["stage"] = "group"
    frame["stage"] = frame["stage"].fillna("group").astype(str)

    frame["outcome"] = frame.apply(_match_outcome, axis=1)
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def clean_fixture_field(value: object) -> str:
    """Normalize optional fixture CSV fields; empty cells must not render as 'nan'."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def load_fixtures_csv(path: Path) -> pd.DataFrame:
    """Load upcoming fixtures for analysis."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "team_a",
                "team_b",
                "neutral",
                "tournament",
                "stage",
            ]
        )

    frame = pd.read_csv(path)
    frame = frame.copy()
    if "date" in frame.columns:
        frame["date"] = _parse_date(frame["date"])
    for column in ("team_a", "team_b"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str).map(canonical_team_name)
    if "neutral" not in frame.columns:
        frame["neutral"] = True
    if "tournament" not in frame.columns:
        frame["tournament"] = "FIFA World Cup"
    if "stage" not in frame.columns:
        frame["stage"] = "group"
    frame["stage"] = frame["stage"].fillna("group").astype(str).map(clean_fixture_field)
    if "group" in frame.columns:
        frame["group"] = frame["group"].map(clean_fixture_field)
    if "round" in frame.columns:
        frame["round"] = frame["round"].map(clean_fixture_field)
    return frame


def _match_outcome(row: pd.Series) -> int:
    if row["home_score"] > row["away_score"]:
        return 0
    if row["home_score"] < row["away_score"]:
        return 2
    return 1


def merge_uploaded_results(existing: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, uploaded], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team", "home_score", "away_score"],
        keep="last",
    )
    return combined.sort_values("date").reset_index(drop=True)


def list_teams(frame: pd.DataFrame) -> list[str]:
    teams = set(frame["home_team"].astype(str)) | set(frame["away_team"].astype(str))
    return sorted(normalize_team_name(team) for team in teams if team and team != "nan")
