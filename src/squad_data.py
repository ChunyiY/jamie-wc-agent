"""World Cup 2026 squad rosters and club per-90 performance data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.team_mapping import canonical_team_name

SQUADS_URL = "https://raw.githubusercontent.com/risingtransfers/world-cup-2026-data/main/data/squads.csv"
PER90_URL = "https://raw.githubusercontent.com/risingtransfers/world-cup-2026-data/main/data/per90_stats.csv"

ATTACK_POSITIONS = {"FW", "MF", "AM", "WG"}

# Raw squad CSV country labels → canonical national team name (results / fixtures).
SQUAD_COUNTRY_ALIASES: dict[str, str] = {
    "Cape Verde Islands": "Cape Verde",
    "Korea Republic": "South Korea",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "United States": "United States",
}


def _top_attackers(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if frame.empty or "position" not in frame.columns:
        return frame
    atk = frame[frame["position"].isin(ATTACK_POSITIONS)].copy()
    if atk.empty:
        return frame
    if "goals_per90" not in atk.columns:
        return atk.head(n)
    atk["_g90"] = pd.to_numeric(atk["goals_per90"], errors="coerce").fillna(0)
    return atk.nlargest(min(n, len(atk)), "_g90")


@dataclass
class SquadPlayer:
    name: str
    team: str
    position: str
    club: str
    age: int
    market_value_eur: float
    goals_per90: float | None = None
    assists_per90: float | None = None
    shots_per90: float | None = None
    key_passes_per90: float | None = None
    tackles_per90: float | None = None
    rating: float | None = None
    minutes: float | None = None
    intl_goals_12m: int = 0
    intl_goals_24m: int = 0
    intl_last_goal: str | None = None
    status: str = "active"


@dataclass
class TeamSquadProfile:
    team: str
    squad_size: int
    avg_age: float
    total_market_value_eur: float
    top11_market_value_eur: float
    top11_goals_per90: float
    top11_assists_per90: float
    top11_shots_per90: float
    attack_depth: int
    players: list[SquadPlayer] = field(default_factory=list)


def load_squads_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["country"] = frame["country"].replace(SQUAD_COUNTRY_ALIASES)
    frame["country"] = frame["country"].astype(str).map(canonical_team_name)
    return frame


def load_per90_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_squads_per90(squads: pd.DataFrame, per90: pd.DataFrame) -> pd.DataFrame:
    if squads.empty:
        return squads
    if per90.empty:
        return squads.copy()
    return squads.merge(per90, on=["player_id", "player_name", "slug"], how="left", suffixes=("", "_p90"))


def _top_n_by_value(frame: pd.DataFrame, n: int = 11) -> pd.DataFrame:
    if frame.empty:
        return frame
    col = "rt_value_estimate_eur" if "rt_value_estimate_eur" in frame.columns else "market_value_eur"
    return frame.nlargest(min(n, len(frame)), col)


def team_squad_profile(merged: pd.DataFrame, team: str) -> TeamSquadProfile | None:
    team = canonical_team_name(team)
    subset = merged[merged["country"] == team].copy()
    if subset.empty:
        return None

    top11 = _top_n_by_value(subset, 11)
    attack_top = _top_attackers(subset, 5)
    attackers = subset[subset["position"].isin(ATTACK_POSITIONS)]

    def _mean(col: str, frame: pd.DataFrame = top11) -> float:
        if col not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[col], errors="coerce").fillna(0).mean())

    val_col = "rt_value_estimate_eur" if "rt_value_estimate_eur" in subset.columns else "market_value_eur"
    values = pd.to_numeric(subset[val_col], errors="coerce").fillna(0)
    top11_values = pd.to_numeric(top11[val_col], errors="coerce").fillna(0)

    players: list[SquadPlayer] = []
    for _, row in subset.iterrows():
        age_val = pd.to_numeric(row.get("age", 0), errors="coerce")
        age_int = 0 if pd.isna(age_val) else int(age_val)
        players.append(
            SquadPlayer(
                name=str(row["player_name"]),
                team=team,
                position=str(row.get("position", "")),
                club=str(row.get("club", "")),
                age=age_int,
                market_value_eur=float(row.get(val_col, 0) or 0),
                goals_per90=_optional_float(row, "goals_per90"),
                assists_per90=_optional_float(row, "assists_per90"),
                shots_per90=_optional_float(row, "shots_per90"),
                key_passes_per90=_optional_float(row, "key_passes_per90"),
                tackles_per90=_optional_float(row, "tackles_per90"),
                rating=_optional_float(row, "rating"),
                minutes=_optional_float(row, "minutes"),
            )
        )
    players.sort(key=lambda p: p.market_value_eur, reverse=True)

    g90 = pd.to_numeric(attackers.get("goals_per90", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return TeamSquadProfile(
        team=team,
        squad_size=len(subset),
        avg_age=float(pd.to_numeric(subset["age"], errors="coerce").mean()),
        total_market_value_eur=float(values.sum()),
        top11_market_value_eur=float(top11_values.sum()),
        top11_goals_per90=_mean("goals_per90", attack_top if not attack_top.empty else top11),
        top11_assists_per90=_mean("assists_per90", attack_top if not attack_top.empty else top11),
        top11_shots_per90=_mean("shots_per90", attack_top if not attack_top.empty else top11),
        attack_depth=int((g90 > 0.15).sum()),
        players=players,
    )


def _optional_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index or pd.isna(row[col]) or row[col] == "":
        return None
    return float(row[col])


def squad_strength_features(
    merged: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    prediction_context: str = "world_cup",
) -> dict[str, float]:
    from src.competition_weights import club_proxy_discount

    discount = club_proxy_discount(prediction_context)
    pa = team_squad_profile(merged, team_a)
    pb = team_squad_profile(merged, team_b)

    def _val(p: TeamSquadProfile | None, attr: str) -> float:
        return float(getattr(p, attr)) if p else 0.0

    raw = {
        "squad_value_diff_log": _log_diff(_val(pa, "top11_market_value_eur"), _val(pb, "top11_market_value_eur")),
        "squad_attack_per90_diff": _val(pa, "top11_goals_per90") - _val(pb, "top11_goals_per90"),
        "squad_creativity_per90_diff": _val(pa, "top11_assists_per90") - _val(pb, "top11_assists_per90"),
        "squad_age_diff": _val(pa, "avg_age") - _val(pb, "avg_age"),
        "squad_depth_diff": float((pa.attack_depth if pa else 0) - (pb.attack_depth if pb else 0)),
    }
    return {k: discount * v for k, v in raw.items()}


def _log_diff(a: float, b: float) -> float:
    return float(np.log1p(max(a, 0)) - np.log1p(max(b, 0)))


SQUAD_FEATURE_COLUMNS = [
    "squad_value_diff_log",
    "squad_attack_per90_diff",
    "squad_creativity_per90_diff",
    "squad_age_diff",
    "squad_depth_diff",
    "recent_attack_diff",
    "recent_scorers_diff",
    "recent_concentration_diff",
]
