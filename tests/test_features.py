"""Tests for point-in-time feature generation."""

import pandas as pd

from src.elo import EloSystem
from src.features import build_elo_lookup, build_training_frame_point_in_time


def test_point_in_time_elo_diff_changes_over_time():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="ME", tz="UTC"),
            "home_team": ["Brazil", "Brazil", "Brazil", "Argentina", "Argentina", "Argentina"],
            "away_team": ["Argentina", "Argentina", "Argentina", "Brazil", "Brazil", "Brazil"],
            "home_score": [3, 2, 1, 0, 1, 2],
            "away_score": [0, 0, 1, 2, 2, 1],
            "tournament": ["Friendly"] * 6,
            "neutral": [True] * 6,
            "stage": ["group"] * 6,
            "outcome": [0, 0, 0, 2, 2, 0],
        }
    )
    training = build_training_frame_point_in_time(frame)
    assert not training.empty
    assert training["elo_diff"].nunique() > 1


def test_elo_lookup_matches_history():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"], utc=True),
            "home_team": ["Brazil", "Argentina"],
            "away_team": ["Argentina", "Brazil"],
            "home_score": [2, 1],
            "away_score": [1, 0],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
            "stage": ["group", "group"],
            "outcome": [0, 0],
        }
    )
    lookup = build_elo_lookup(frame)
    key = (frame.iloc[1]["date"], "Argentina", "Brazil")
    assert key in lookup
