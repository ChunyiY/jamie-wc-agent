"""Tests for formation inference and tactical profile."""

from __future__ import annotations

import pandas as pd

from src.football_profile import build_team_football_profile, select_starting_xi


def _squad_row(name: str, pos: str, value: float, **stats: float) -> dict:
    row = {
        "player_name": name,
        "position": pos,
        "rt_value_estimate_eur": value,
        "age": 26,
        "goals_per90": 0.0,
        "assists_per90": 0.0,
        "shots_per90": 0.0,
        "key_passes_per90": 0.0,
        "pass_accuracy_pct": 80.0,
        "tackles_per90": 1.5,
        "interceptions_per90": 1.0,
        "clearances_per90": 2.0,
        "saves_per90": 0.0,
        "rating": 7.0,
    }
    row.update(stats)
    return row


def test_spain_like_squad_gets_433_not_622():
    """High-value defenders + forwards must not produce 6-2-2."""
    rows = [
        _squad_row("GK1", "GK", 17e6),
        _squad_row("GK2", "GK", 16e6),
        _squad_row("CB1", "DF", 18e6),
        _squad_row("CB2", "DF", 17e6),
        _squad_row("LB", "DF", 16e6),
        _squad_row("RB", "DF", 16e6),
        _squad_row("CB3", "DF", 15e6),
        _squad_row("CM1", "MF", 14e6, pass_accuracy_pct=92, key_passes_per90=1.5),
        _squad_row("CM2", "MF", 14e6, pass_accuracy_pct=91, key_passes_per90=1.2),
        _squad_row("CM3", "MF", 13e6, pass_accuracy_pct=88, key_passes_per90=1.8),
        _squad_row("FW1", "FW", 19e6, goals_per90=0.7, shots_per90=3.0),
        _squad_row("FW2", "FW", 19e6, goals_per90=0.5, shots_per90=2.5),
        _squad_row("FW3", "FW", 18e6, goals_per90=0.4, shots_per90=2.8),
        _squad_row("FW4", "FW", 15e6, goals_per90=0.6, shots_per90=2.2),
    ]
    frame = pd.DataFrame(rows)
    frame["country"] = "Spain"
    formation, xi = select_starting_xi(frame)
    assert formation == "4-3-3"
    assert len(xi) == 11
    assert (xi["position"].astype(str).map(lambda p: p) == "GK").sum() == 1
    assert "6-2-2" not in formation

    profile = build_team_football_profile(frame, "Spain")
    assert profile is not None
    assert profile.formation_hint == "4-3-3"
    assert "控球" in profile.style_label or "组织" in profile.style_label or "进攻" in profile.style_label
    assert len(profile.lineup_hint) == 4


def test_select_starting_xi_one_goalkeeper():
    frame = pd.DataFrame(
        [
            _squad_row("G1", "GK", 20e6),
            _squad_row("G2", "GK", 18e6),
            _squad_row("D1", "DF", 15e6),
            _squad_row("D2", "DF", 14e6),
            _squad_row("D3", "DF", 13e6),
            _squad_row("D4", "DF", 12e6),
            _squad_row("M1", "MF", 11e6),
            _squad_row("M2", "MF", 10e6),
            _squad_row("M3", "MF", 9e6),
            _squad_row("F1", "FW", 16e6),
            _squad_row("F2", "FW", 15e6),
            _squad_row("F3", "FW", 14e6),
        ]
    )
    _, xi = select_starting_xi(frame)
    gk_count = (xi["position"] == "GK").sum()
    assert gk_count == 1
