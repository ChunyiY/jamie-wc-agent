"""Tests for squad data module."""

import pandas as pd

from src.squad_data import merge_squads_per90, squad_strength_features, team_squad_profile


def test_squad_strength_features():
    squads = pd.DataFrame(
        [
            [1, "Player A", "a", "Brazil", "BRA", "FW", "Club A", 25, 50_000_000],
            [2, "Player B", "b", "Brazil", "BRA", "MF", "Club B", 26, 30_000_000],
            [3, "Player C", "c", "Argentina", "ARG", "FW", "Club C", 27, 40_000_000],
        ],
        columns=["player_id", "player_name", "slug", "country", "country_code", "position", "club", "age", "rt_value_estimate_eur"],
    )
    per90 = pd.DataFrame(
        [
            [1, "Player A", "a", "2025-26", 2000, 0.6, 0.2, 3, 1, 1, 1, 1, 40, 80, 0, 7.0],
            [2, "Player B", "b", "2025-26", 2000, 0.2, 0.3, 1, 2, 2, 1, 1, 50, 85, 0, 6.5],
            [3, "Player C", "c", "2025-26", 2000, 0.5, 0.1, 2, 1, 1, 1, 1, 35, 82, 0, 7.2],
        ],
        columns=[
            "player_id", "player_name", "slug", "season", "minutes", "goals_per90",
            "assists_per90", "shots_per90", "key_passes_per90", "tackles_per90",
            "interceptions_per90", "clearances_per90", "passes_per90", "pass_accuracy_pct",
            "saves_per90", "rating",
        ],
    )
    merged = merge_squads_per90(squads, per90)
    profile = team_squad_profile(merged, "Brazil")
    assert profile is not None
    assert profile.squad_size == 2
    feats = squad_strength_features(merged, "Brazil", "Argentina")
    assert "squad_value_diff_log" in feats
    assert feats["squad_attack_per90_diff"] != 0
