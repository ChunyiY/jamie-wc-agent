"""Tests for team name normalization edge cases."""

from __future__ import annotations

import math

import pandas as pd

from src.player_stats import load_goalscorers
from src.team_mapping import canonical_team_name
from src.utils import normalize_team_name


def test_normalize_team_name_handles_nan():
    assert normalize_team_name(None) == ""
    assert normalize_team_name(float("nan")) == ""
    assert normalize_team_name("nan") == ""
    assert normalize_team_name("  Spain  ") == "Spain"


def test_canonical_team_name_handles_nan():
    assert canonical_team_name(None) == ""
    assert canonical_team_name(float("nan")) == ""
    assert canonical_team_name("Korea Republic") == "South Korea"


def test_load_goalscorers_tolerates_missing_team(tmp_path):
    path = tmp_path / "goalscorers.csv"
    pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "team": math.nan,
                "scorer": "Someone",
                "minute": 10,
                "own_goal": False,
                "penalty": False,
            },
            {
                "date": "2020-01-02",
                "home_team": "Brazil",
                "away_team": "Argentina",
                "team": "Brazil",
                "scorer": "Neymar",
                "minute": 20,
                "own_goal": False,
                "penalty": False,
            },
        ]
    ).to_csv(path, index=False)

    frame = load_goalscorers(path)
    assert len(frame) == 1
    assert frame.iloc[0]["team"] == "Brazil"
