"""Tests for Dixon–Coles score model."""

import pandas as pd
import pytest

from src.dixon_coles import fit_dixon_coles, predict_dixon_coles


def _mini_results() -> pd.DataFrame:
    rows = []
    teams = ["Brazil", "Argentina", "France", "Germany"]
    for i in range(80):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        hs, aw = (2, 1) if i % 4 else (1, 1)
        rows.append(
            {
                "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aw,
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def test_dixon_coles_probs_sum_to_one():
    params = fit_dixon_coles(_mini_results())
    probs, sp = predict_dixon_coles(params, "Brazil", "Argentina", neutral=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert sp.most_likely.probability > 0
    assert sp.expected_goals_a > 0


def test_dixon_coles_score_matrix_aligned():
    params = fit_dixon_coles(_mini_results())
    probs, sp = predict_dixon_coles(params, "France", "Germany", neutral=False)
    matrix_sum = sum(sum(row) for row in sp.score_matrix)
    assert matrix_sum == pytest.approx(1.0, abs=1e-5)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


def test_dixon_coles_outcome_bounds():
    params = fit_dixon_coles(_mini_results())
    probs, _ = predict_dixon_coles(params, "Brazil", "France", neutral=True)
    assert all(0 <= p <= 1 for p in probs)
