"""Tests for international tactical profiles."""

from pathlib import Path

import pandas as pd

from src.elo import EloSystem
from src.football_data import load_results_csv
from src.player_stats import load_goalscorers
from src.team_tactics import build_international_tactical_profile, international_tactical_diff


def test_argentina_tactical_profile():
    results = load_results_csv(Path("data/results.csv"))
    gs = load_goalscorers(Path("data/goalscorers.csv"))
    elo = EloSystem()
    elo.fit_history(results.sort_values("date").tail(5000))
    ref = pd.Timestamp("2026-07-01", tz="UTC")
    p = build_international_tactical_profile(results, gs, "Argentina", ref, elo, prediction_context="world_cup")
    assert 0 <= p.mentality_index <= 3
    assert p.clinical_index > 0


def test_tactical_diff_keys():
    results = load_results_csv(Path("data/results.csv"))
    gs = load_goalscorers(Path("data/goalscorers.csv"))
    elo = EloSystem()
    elo.fit_history(results.sort_values("date").tail(3000))
    ref = pd.Timestamp("2026-07-01", tz="UTC")
    d = international_tactical_diff(results, gs, "Argentina", "Cape Verde", ref, elo)
    assert "wc_form_diff" in d
    assert "mentality_diff" in d
