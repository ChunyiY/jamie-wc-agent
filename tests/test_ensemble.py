"""Tests for prediction ensemble."""

import numpy as np
import pandas as pd
import pytest

from src.dixon_coles import DixonColesParams, fit_dixon_coles
from src.ensemble import PredictionEnsemble
from src.features import FEATURE_COLUMNS, TRAINING_FEATURE_COLUMNS
from src.model import FootballOutcomeModel
from src.ordered_logit import OrderedLogitModel


def _training_frame(n: int = 100) -> pd.DataFrame:
    rows = []
    teams = ["Brazil", "Argentina", "France", "Germany", "Spain", "Italy"]
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        outcome = i % 3
        row = {col: 0.0 for col in TRAINING_FEATURE_COLUMNS}
        row.update(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "team_a": home,
                "team_b": away,
                "outcome": outcome,
                "neutral_venue": 1.0,
            }
        )
        row["elo_diff"] = float(i % 5)
        rows.append(row)
    return pd.DataFrame(rows)


def _mini_results() -> pd.DataFrame:
    rows = []
    teams = ["Brazil", "Argentina", "France", "Germany", "Spain", "Italy"]
    for i in range(100):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        hs = 1 if i % 3 else 0
        aw = 0 if hs else 1
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aw,
                "tournament": "Friendly",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def test_ensemble_predict_returns_uncertainty():
    training = _training_frame(120)
    results = _mini_results()
    dc = fit_dixon_coles(results)
    x = training[TRAINING_FEATURE_COLUMNS].to_numpy()
    y = training["outcome"].to_numpy().astype(int)
    ol = OrderedLogitModel()
    ol.fit(x, y, feature_columns=TRAINING_FEATURE_COLUMNS.copy())
    ml = FootballOutcomeModel()
    ml.train(training, calibrate=False)

    ensemble = PredictionEnsemble(dc, ol, ml, weights=(0.4, 0.3, 0.3))
    feats = {col: float(training.iloc[-1][col]) for col in TRAINING_FEATURE_COLUMNS}
    out = ensemble.predict(feats, "Brazil", "Argentina", neutral=True)
    total = out.prob_team_a_win + out.prob_draw + out.prob_team_b_win
    assert abs(total - 1.0) < 1e-5
    assert out.score_prediction is not None
    assert len(out.uncertainty) == 3
    assert out.components.dixon_coles[0] >= 0


def test_ensemble_default_params():
    dc = DixonColesParams()
    ol = OrderedLogitModel()
    ml = FootballOutcomeModel()
    ens = PredictionEnsemble(dc, ol, ml)
    assert sum(ens.weights) == pytest.approx(1.0)
