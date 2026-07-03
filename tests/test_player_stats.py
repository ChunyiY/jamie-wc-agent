"""Tests for player statistics module."""

from pathlib import Path

import pandas as pd

from src.player_stats import (
    head_to_head_scorers,
    load_goalscorers,
    players_to_dataframe,
    recent_intl_attack_stats,
    team_player_summary,
)
from src.squad_data import merge_squads_per90


def _sample_goalscorers(tmp_path: Path) -> Path:
    path = tmp_path / "goalscorers.csv"
    rows = [
        ["2024-01-01", "Brazil", "Argentina", "Brazil", "Vinícius Júnior", 30, "FALSE", "FALSE"],
        ["2024-06-01", "Brazil", "France", "Brazil", "Vinícius Júnior", 12, "FALSE", "TRUE"],
        ["2023-03-01", "Argentina", "Brazil", "Argentina", "Lionel Messi", 70, "FALSE", "FALSE"],
        ["2022-11-01", "Argentina", "France", "Argentina", "Lionel Messi", 80, "FALSE", "FALSE"],
        ["1916-07-02", "Brazil", "Uruguay", "Brazil", "Pelé", 44, "FALSE", "FALSE"],
    ]
    pd.DataFrame(
        rows,
        columns=["date", "home_team", "away_team", "team", "scorer", "minute", "own_goal", "penalty"],
    ).to_csv(path, index=False)
    return path


def _sample_squads(tmp_path: Path) -> pd.DataFrame:
    squads = pd.DataFrame(
        [
            [1, "Vinícius Júnior", "vini", "Brazil", "BRA", "FW", "Real Madrid", 24, 100_000_000],
            [2, "Lionel Messi", "messi", "Argentina", "ARG", "FW", "Inter Miami", 37, 50_000_000],
        ],
        columns=["player_id", "player_name", "slug", "country", "country_code", "position", "club", "age", "rt_value_estimate_eur"],
    )
    per90 = pd.DataFrame(
        [
            [1, "Vinícius Júnior", "vini", "2025-26", 2000, 0.5, 0.3, 3.0, 2.0, 1.0, 0.5, 0.5, 40.0, 85.0, 0, 7.5],
            [2, "Lionel Messi", "messi", "2025-26", 1800, 0.6, 0.4, 2.5, 2.5, 0.5, 0.3, 0.2, 50.0, 88.0, 0, 7.8],
        ],
        columns=[
            "player_id", "player_name", "slug", "season", "minutes", "goals_per90",
            "assists_per90", "shots_per90", "key_passes_per90", "tackles_per90",
            "interceptions_per90", "clearances_per90", "passes_per90", "pass_accuracy_pct",
            "saves_per90", "rating",
        ],
    )
    return merge_squads_per90(squads, per90)


def test_team_player_summary_active_only(tmp_path: Path):
    gs = load_goalscorers(_sample_goalscorers(tmp_path))
    merged = _sample_squads(tmp_path)
    summary = team_player_summary(gs, "Brazil", merged_squads=merged)
    assert summary.wc_squad is not None
    names = [p.name for p in summary.top_contributors]
    assert "Vinícius Júnior" in names
    assert "Pelé" not in names
    df = players_to_dataframe(summary.top_contributors)
    assert "俱乐部进球/90" in df.columns


def test_recent_intl_attack_stats(tmp_path: Path):
    gs = load_goalscorers(_sample_goalscorers(tmp_path))
    stats = recent_intl_attack_stats(gs, "Brazil", pd.Timestamp("2025-01-01", tz="UTC"))
    assert stats["recent_goals"] >= 1


def test_head_to_head_scorers(tmp_path: Path):
    gs = load_goalscorers(_sample_goalscorers(tmp_path))
    rows = head_to_head_scorers(gs, "Brazil", "Argentina")
    assert any(r.scorer == "Lionel Messi" for r in rows)
