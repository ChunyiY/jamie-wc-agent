"""Tests for Elo rating system."""

import pandas as pd

from src.elo import EloSystem, expected_score, goal_difference_multiplier


def test_expected_score_symmetry():
    assert abs(expected_score(1600, 1500) + expected_score(1500, 1600) - 1.0) < 1e-9


def test_goal_difference_multiplier():
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(4) > 1.5


def test_elo_updates_after_match():
    elo = EloSystem()
    home_before = elo.get_rating("Brazil")
    away_before = elo.get_rating("Argentina")
    elo.update_match("Brazil", "Argentina", 2, 0, neutral=True)
    assert elo.get_rating("Brazil") > home_before
    assert elo.get_rating("Argentina") < away_before


def test_elo_fit_history():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="ME"),
            "home_team": ["Brazil", "France", "Germany", "Spain", "England"],
            "away_team": ["Argentina", "Italy", "Portugal", "Netherlands", "Belgium"],
            "home_score": [2, 1, 0, 2, 1],
            "away_score": [1, 1, 2, 0, 0],
            "neutral": [True] * 5,
            "tournament": ["Friendly"] * 5,
        }
    )
    elo = EloSystem()
    history = elo.fit_history(frame)
    assert len(history) == 5
    assert not elo.to_dataframe().empty
