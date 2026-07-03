"""Penalty shootout historical rates for knockout advance modelling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.team_mapping import canonical_team_name


@dataclass
class ShootoutRates:
    team: str
    shootouts_played: int
    shootouts_won: int
    win_rate: float


def load_shootouts(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner", "first_shooter"])
    frame = pd.read_csv(path)
    for col in ("home_team", "away_team", "winner"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str).map(canonical_team_name)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return frame.dropna(subset=["home_team", "away_team", "winner"])


def team_shootout_rates(shootouts: pd.DataFrame, team: str) -> ShootoutRates:
    team = canonical_team_name(team)
    if shootouts.empty:
        return ShootoutRates(team, 0, 0, 0.5)

    mask = (shootouts["home_team"] == team) | (shootouts["away_team"] == team)
    subset = shootouts[mask]
    played = len(subset)
    won = int((subset["winner"] == team).sum())
    rate = won / played if played else 0.5
    return ShootoutRates(team=team, shootouts_played=played, shootouts_won=won, win_rate=rate)


def knockout_advance_probs(
    prob_win_a: float,
    prob_draw: float,
    prob_win_b: float,
    *,
    team_a: str,
    team_b: str,
    shootouts: pd.DataFrame,
    extra_time_draw_share: float = 0.72,
) -> tuple[float, float, dict[str, float]]:
    """P(advance) with ET and historical shootout strength (Beta shrinkage)."""
    ra = team_shootout_rates(shootouts, team_a)
    rb = team_shootout_rates(shootouts, team_b)

    prior = 5.0
    pk_a = (ra.shootouts_won + prior) / (ra.shootouts_played + 2 * prior)
    pk_b = (rb.shootouts_won + prior) / (rb.shootouts_played + 2 * prior)
    pk_share_a = pk_a / (pk_a + pk_b) if (pk_a + pk_b) > 0 else 0.5

    p_draw_to_so = prob_draw * extra_time_draw_share
    advance_a = prob_win_a + p_draw_to_so * 0.5 + p_draw_to_so * 0.5 * pk_share_a
    advance_b = prob_win_b + p_draw_to_so * 0.5 + p_draw_to_so * 0.5 * (1.0 - pk_share_a)
    total = advance_a + advance_b
    if total <= 0:
        return 0.5, 0.5, {}
    meta = {
        "pk_rate_a": pk_share_a,
        "pk_rate_b": 1.0 - pk_share_a,
        "shootouts_a": float(ra.shootouts_played),
        "shootouts_b": float(rb.shootouts_played),
        "draw_to_shootout": p_draw_to_so,
    }
    return advance_a / total, advance_b / total, meta
