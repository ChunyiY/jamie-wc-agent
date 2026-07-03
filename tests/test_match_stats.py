"""Tests for international match quality stats (StatsBomb + WC2026)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.match_stats import (
    MATCH_QUALITY_COLUMNS,
    build_team_match_quality,
    load_intl_match_stats,
    match_quality_feature_diff,
)
from src.wc2026_data import build_wc2026_match_stats


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_intl_match_stats_has_rows_when_file_exists():
    path = DATA_DIR / "intl_match_stats.csv"
    if not path.exists():
        return
    frame = load_intl_match_stats(path)
    assert not frame.empty
    assert "xg" in frame.columns
    assert frame["team"].notna().all()


def test_build_team_match_quality_weighted():
    stats = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-01", tz="UTC"),
                "team": "Spain",
                "opponent": "Italy",
                "goals_for": 2,
                "xg": 1.5,
                "shots": 12,
                "sot_pct": 0.4,
                "tournament": "UEFA Euro",
            },
            {
                "date": pd.Timestamp("2024-07-01", tz="UTC"),
                "team": "Spain",
                "opponent": "France",
                "goals_for": 1,
                "xg": 1.8,
                "shots": 14,
                "sot_pct": 0.5,
                "tournament": "UEFA Euro",
            },
        ]
    )
    profile = build_team_match_quality(
        stats, "Spain", pd.Timestamp("2025-01-01", tz="UTC"), prediction_context="world_cup"
    )
    assert profile.sample_matches == 2
    assert profile.xg_per_match > 1.5
    assert profile.sot_pct > 0.4


def test_match_quality_feature_diff_keys():
    stats = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-06-01", tz="UTC"),
                "team": "Spain",
                "opponent": "Italy",
                "goals_for": 2,
                "xg": 2.0,
                "shots": 15,
                "sot_pct": 0.5,
                "tournament": "FIFA World Cup",
            },
            {
                "date": pd.Timestamp("2024-06-01", tz="UTC"),
                "team": "Austria",
                "opponent": "France",
                "goals_for": 0,
                "xg": 0.8,
                "shots": 8,
                "sot_pct": 0.25,
                "tournament": "FIFA World Cup",
            },
        ]
    )
    rankings = pd.DataFrame(
        [{"team": "Spain", "fifa_rank": 3}, {"team": "Austria", "fifa_rank": 22}]
    )
    diff = match_quality_feature_diff(
        stats,
        rankings,
        "Spain",
        "Austria",
        pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert set(diff.keys()) == set(MATCH_QUALITY_COLUMNS)
    assert diff["xg_quality_diff"] > 0
    assert diff["fifa_rank_diff"] > 0


def test_wc2026_match_stats_from_downloaded_files():
    wc = build_wc2026_match_stats(DATA_DIR)
    if wc.empty:
        return
    assert "xg" in wc.columns
    assert (wc["source"] == "fifa.com").any()
