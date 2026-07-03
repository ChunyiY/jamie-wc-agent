"""Mismatch score refinement for heavy favorites."""

from pathlib import Path

import pandas as pd

from src.dixon_coles import fit_dixon_coles, predict_dixon_coles
from src.football_data import load_results_csv
from src.score_prediction import refine_score_for_context
from src.squad_data import load_per90_csv, load_squads_csv, merge_squads_per90


def test_argentina_cape_verde_boosts_favorite_xg():
    results = load_results_csv(Path("data/results.csv"))
    params = fit_dixon_coles(results)
    _, base_sp = predict_dixon_coles(params, "Argentina", "Cape Verde", neutral=True)

    squads = load_squads_csv(Path("data/wc_squads.csv"))
    per90 = load_per90_csv(Path("data/wc_per90_stats.csv"))
    merged = merge_squads_per90(squads, per90)

    refined = refine_score_for_context(
        base_sp,
        team_a="Argentina",
        team_b="Cape Verde",
        prob_team_a_win=0.78,
        prob_draw=0.14,
        prob_team_b_win=0.08,
        elo_diff=465.0,
        merged_squads=merged,
    )
    assert refined.expected_goals_a > base_sp.expected_goals_a * 0.95
    assert refined.expected_goals_a >= 1.55
    assert refined.most_likely.goals_a >= 1
