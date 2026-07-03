"""Combined international match statistics (StatsBomb + WC 2026) for team quality features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.competition_weights import match_row_weight
from src.team_mapping import canonical_team_name
from src.wc2026_data import build_wc2026_match_stats

MATCH_STATS_FILENAME = "intl_match_stats.csv"
FIFA_RANKINGS_FILENAME = "fifa_rankings_wc2026.csv"


@dataclass
class TeamMatchQualityProfile:
    team: str
    xg_per_match: float
    goals_per_match: float
    shots_per_match: float
    sot_pct: float
    xg_overperformance: float
    sample_matches: int


def load_intl_match_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["team"] = frame["team"].astype(str).map(canonical_team_name)
    frame["opponent"] = frame["opponent"].astype(str).map(canonical_team_name)
    return frame.dropna(subset=["date", "team"]).sort_values("date")


def load_fifa_rankings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["team"] = frame["team"].astype(str).map(canonical_team_name)
    return frame


def build_team_match_quality(
    stats: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    *,
    prediction_context: str = "world_cup",
    window: int = 12,
) -> TeamMatchQualityProfile:
    team = canonical_team_name(team)
    if before_date.tzinfo is None:
        before_date = before_date.tz_localize("UTC")
    if stats.empty:
        return TeamMatchQualityProfile(team, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    subset = stats[(stats["team"] == team) & (stats["date"] < before_date)].tail(window)
    if subset.empty:
        return TeamMatchQualityProfile(team, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    weights = []
    xg_w = gf_w = shots_w = sot_w = 0.0
    total_w = 0.0
    for _, row in subset.iterrows():
        age = (before_date - row["date"]).total_seconds() / 86400.0
        w = match_row_weight(str(row.get("tournament", "FIFA World Cup")), age, prediction_context=prediction_context)
        weights.append(w)
        total_w += w
        xg_w += w * float(row.get("xg", 0) or 0)
        gf_w += w * float(row.get("goals_for", 0) or 0)
        shots_w += w * float(row.get("shots", 0) or 0)
        sot_w += w * float(row.get("sot_pct", 0) or 0)

    tw = max(total_w, 1e-9)
    xg_avg = xg_w / tw
    gf_avg = gf_w / tw
    shots_avg = shots_w / tw
    sot_avg = sot_w / tw
    xg_over = gf_avg - xg_avg

    return TeamMatchQualityProfile(
        team=team,
        xg_per_match=round(xg_avg, 3),
        goals_per_match=round(gf_avg, 3),
        shots_per_match=round(shots_avg, 2),
        sot_pct=round(sot_avg, 3),
        xg_overperformance=round(xg_over, 3),
        sample_matches=len(subset),
    )


MATCH_QUALITY_COLUMNS = [
    "xg_quality_diff",
    "sot_pct_diff",
    "shots_volume_diff",
    "xg_overperf_diff",
    "fifa_rank_diff",
]


def match_quality_feature_diff(
    stats: pd.DataFrame,
    rankings: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: pd.Timestamp,
    *,
    prediction_context: str = "world_cup",
) -> dict[str, float]:
    pa = build_team_match_quality(stats, team_a, before_date, prediction_context=prediction_context)
    pb = build_team_match_quality(stats, team_b, before_date, prediction_context=prediction_context)

    rank_a = rank_b = 50.0
    if not rankings.empty:
        ra = rankings[rankings["team"] == canonical_team_name(team_a)]
        rb = rankings[rankings["team"] == canonical_team_name(team_b)]
        if not ra.empty:
            rank_a = float(ra.iloc[0].get("fifa_rank", 50))
        if not rb.empty:
            rank_b = float(rb.iloc[0].get("fifa_rank", 50))

    return {
        "xg_quality_diff": pa.xg_per_match - pb.xg_per_match,
        "sot_pct_diff": pa.sot_pct - pb.sot_pct,
        "shots_volume_diff": pa.shots_per_match - pb.shots_per_match,
        "xg_overperf_diff": pa.xg_overperformance - pb.xg_overperformance,
        "fifa_rank_diff": rank_b - rank_a,  # lower rank number = better
    }


def merge_all_match_stats(data_dir: Path) -> pd.DataFrame:
    """Combine on-disk StatsBomb + WC2026 team-match rows."""
    parts: list[pd.DataFrame] = []
    sb_path = data_dir / MATCH_STATS_FILENAME
    if sb_path.exists():
        sb = load_intl_match_stats(sb_path)
        if not sb.empty:
            parts.append(sb)
    wc = build_wc2026_match_stats(data_dir)
    if not wc.empty:
        parts.append(wc)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(
        subset=["date", "team", "opponent", "source"], keep="last"
    )
    return combined.reset_index(drop=True)


def quality_profile_to_dataframe(p: TeamMatchQualityProfile) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"指标": "样本场次", "数值": str(p.sample_matches)},
            {"指标": "场均 xG", "数值": f"{p.xg_per_match:.2f}"},
            {"指标": "场均进球", "数值": f"{p.goals_per_match:.2f}"},
            {"指标": "场均射门", "数值": f"{p.shots_per_match:.1f}"},
            {"指标": "射正率", "数值": f"{p.sot_pct:.1%}"},
            {"指标": "xG 超额进球", "数值": f"{p.xg_overperformance:+.2f}"},
        ]
    )
