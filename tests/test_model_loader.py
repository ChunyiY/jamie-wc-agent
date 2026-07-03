"""Tests for fast model loader."""

from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.model_loader import load_or_train_football_stack, results_fingerprint, train_football_stack


def _mini_results() -> pd.DataFrame:
    rows = []
    teams = ["Brazil", "Argentina", "France", "Germany", "Spain", "Italy"]
    for i in range(120):
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
                "outcome": 0 if hs > aw else (2 if aw > hs else 1),
            }
        )
    return pd.DataFrame(rows)


def test_results_fingerprint_stable():
    df = _mini_results()
    config = AppConfig(modeling_match_limit=100, elo_match_limit=100)
    assert results_fingerprint(df, config) == results_fingerprint(df, config)


def test_train_football_stack_runs_fast(tmp_path: Path):
    config = AppConfig(
        modeling_match_limit=80,
        elo_match_limit=80,
        fast_train_skip_calibration=True,
        elo_table_path=tmp_path / "elo.csv",
    )
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    results = _mini_results()
    progress = []

    def report(pct: float, msg: str) -> None:
        progress.append((pct, msg))

    elo, model, analyzer, summary = train_football_stack(results, config, progress=report)
    assert len(elo.ratings) > 0
    assert model.bundle is not None
    assert analyzer is not None
    assert summary.source == "trained"
    assert progress[-1][0] == 1.0
