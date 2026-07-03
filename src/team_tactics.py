"""
Team-level tactical and mentality indices from international results + goal scorers.

Club per-90 aggregates (shots, conversion) are blended separately with a discount
(see competition_weights.club_proxy_discount).

Proxies when event data is limited
----------------------------------
* **Clinical index** — goals per match adjusted for opponent defensive strength (Elo).
* **Late-game mentality** — share of goals scored after minute 75 (goalscorers.csv).
* **Big-game index** — points rate vs opponents stronger by Elo.
* **Knockout mentality** — points rate in knockout / WC knock-out stages.
* **Shot conversion (club)** — Σgoals/Σshots from per-90 roster (on-target proxy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.competition_weights import match_row_weight, tournament_tier, CompetitionTier
from src.elo import EloSystem
from src.squad_data import ATTACK_POSITIONS, team_squad_profile
from src.team_mapping import canonical_team_name

KNOCKOUT_KEYWORDS = ("knockout", "round of", "quarter", "semi", "final", "play-off", "playoff")


def _utc_ts(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _match_dates(results: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(results["date"], utc=True)


@dataclass
class InternationalTacticalProfile:
    team: str
    wc_weighted_form: float
    mentality_index: float
    clinical_index: float
    late_goal_share: float
    knockout_points_rate: float
    clean_sheet_rate: float
    failed_to_score_rate: float
    big_game_points_rate: float


@dataclass
class ClubTacticalProxy:
    team: str
    shot_volume: float
    shot_conversion: float
    pass_accuracy: float
    defensive_actions: float


def _is_knockout_row(row: pd.Series) -> bool:
    stage = str(row.get("stage", "")).lower()
    tournament = str(row.get("tournament", "")).lower()
    if any(k in stage for k in KNOCKOUT_KEYWORDS):
        return True
    if "world cup" in tournament and any(k in stage for k in ("round", "final", "semi", "quarter")):
        return True
    return False


def _team_points(row: pd.Series, team: str) -> float:
    team = canonical_team_name(team)
    if row["home_team"] == team:
        hs, aws = int(row["home_score"]), int(row["away_score"])
    else:
        hs, aws = int(row["away_score"]), int(row["home_score"])
    if hs > aws:
        return 3.0
    if hs == aws:
        return 1.0
    return 0.0


def _team_goals(row: pd.Series, team: str) -> tuple[int, int]:
    if row["home_team"] == team:
        return int(row["home_score"]), int(row["away_score"])
    return int(row["away_score"]), int(row["home_score"])


def _opponent_elo(row: pd.Series, team: str, elo: EloSystem) -> float:
    opp = row["away_team"] if row["home_team"] == team else row["home_team"]
    return elo.get_rating(str(opp))


def build_international_tactical_profile(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    elo: EloSystem,
    *,
    prediction_context: str = "world_cup",
    window: int = 20,
) -> InternationalTacticalProfile:
    team = canonical_team_name(team)
    before_date = _utc_ts(before_date)
    dates = _match_dates(results)

    mask = (
        (dates < before_date)
        & ((results["home_team"] == team) | (results["away_team"] == team))
    )
    history = results.loc[mask].sort_values("date").tail(window)
    if history.empty:
        return InternationalTacticalProfile(
            team=team,
            wc_weighted_form=0.0,
            mentality_index=0.0,
            clinical_index=0.0,
            late_goal_share=0.0,
            knockout_points_rate=0.0,
            clean_sheet_rate=0.0,
            failed_to_score_rate=0.0,
            big_game_points_rate=0.0,
        )

    weighted_pts = 0.0
    total_w = 0.0
    goals_for = 0
    goals_against = 0
    clean_sheets = 0
    failed = 0
    ko_pts: list[float] = []
    ko_n = 0
    big_pts: list[float] = []
    big_n = 0
    team_elo = elo.get_rating(team)

    for _, row in history.iterrows():
        row_date = _utc_ts(row["date"])
        age = (before_date - row_date).total_seconds() / 86400.0
        w = match_row_weight(
            str(row.get("tournament", "Friendly")),
            age,
            prediction_context=prediction_context,
        )
        pts = _team_points(row, team)
        weighted_pts += w * pts
        total_w += w
        gf, ga = _team_goals(row, team)
        goals_for += gf
        goals_against += ga
        if ga == 0:
            clean_sheets += 1
        if gf == 0:
            failed += 1
        if _is_knockout_row(row):
            ko_pts.append(pts)
            ko_n += 1
        opp_elo = _opponent_elo(row, team, elo)
        if opp_elo >= team_elo - 25:
            big_pts.append(pts)
            big_n += 1

    n = max(len(history), 1)
    wc_form = weighted_pts / max(total_w, 1e-9)
    gf_avg = goals_for / n
    # Clinical: scoring vs expectation given opponent quality
    opp_elo_avg = float(np.mean([_opponent_elo(row, team, elo) for _, row in history.iterrows()]))
    expected_gf = max(0.6, 1.35 * (team_elo / max(opp_elo_avg, 1200)))
    clinical = float(np.clip(gf_avg / expected_gf, 0.2, 2.5))

    late_share = _late_goal_share(goalscorers, team, before_date, months=24)
    ko_rate = float(np.mean(ko_pts)) if ko_pts else wc_form / 3.0
    big_rate = float(np.mean(big_pts)) if big_pts else wc_form / 3.0
    mentality = 0.45 * (big_rate / 3.0) + 0.35 * (ko_rate / 3.0) + 0.20 * late_share

    return InternationalTacticalProfile(
        team=team,
        wc_weighted_form=round(wc_form, 3),
        mentality_index=round(mentality, 3),
        clinical_index=round(clinical, 3),
        late_goal_share=round(late_share, 3),
        knockout_points_rate=round(ko_rate, 3),
        clean_sheet_rate=round(clean_sheets / n, 3),
        failed_to_score_rate=round(failed / n, 3),
        big_game_points_rate=round(big_rate, 3),
    )


def _late_goal_share(
    goalscorers: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    *,
    months: int = 24,
) -> float:
    if goalscorers.empty:
        return 0.0
    team = canonical_team_name(team)
    if before_date.tzinfo is None:
        before_date = before_date.tz_localize("UTC")
    cutoff = before_date - pd.Timedelta(days=months * 30)
    subset = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] < before_date)
        & (goalscorers["date"] >= cutoff)
        & (~goalscorers.get("own_goal", False))
    ]
    if subset.empty or "minute" not in subset.columns:
        return 0.0
    mins = pd.to_numeric(subset["minute"], errors="coerce").dropna()
    if mins.empty:
        return 0.0
    return float((mins >= 75).mean())


def build_club_tactical_proxy(merged_squads: pd.DataFrame, team: str) -> ClubTacticalProxy | None:
    profile = team_squad_profile(merged_squads, team)
    if profile is None:
        return None
    subset = merged_squads[merged_squads["country"] == canonical_team_name(team)]
    if subset.empty:
        return None

    atk = subset[subset["position"].isin(ATTACK_POSITIONS)]
    defs = subset[subset["position"].astype(str).str.upper().isin({"DF", "CB", "LB", "RB", "WB"})]

    shots = pd.to_numeric(atk.get("shots_per90", pd.Series(dtype=float)), errors="coerce").fillna(0)
    goals = pd.to_numeric(atk.get("goals_per90", pd.Series(dtype=float)), errors="coerce").fillna(0)
    shot_vol = float(shots.mean()) if len(shots) else 0.0
    conversion = float(goals.sum() / max(shots.sum(), 0.1)) if len(shots) else 0.0

    pass_acc = pd.to_numeric(subset.get("pass_accuracy_pct", pd.Series(dtype=float)), errors="coerce").fillna(0)
    tackles = pd.to_numeric(defs.get("tackles_per90", pd.Series(dtype=float)), errors="coerce").fillna(0)
    inter = pd.to_numeric(defs.get("interceptions_per90", pd.Series(dtype=float)), errors="coerce").fillna(0)

    return ClubTacticalProxy(
        team=canonical_team_name(team),
        shot_volume=round(shot_vol, 3),
        shot_conversion=round(conversion, 3),
        pass_accuracy=round(float(pass_acc.mean()) if len(pass_acc) else 0.0, 1),
        defensive_actions=round(float(tackles.mean() + inter.mean()) if len(tackles) else 0.0, 3),
    )


TACTICAL_INTL_COLUMNS = [
    "wc_form_diff",
    "mentality_diff",
    "clinical_diff",
    "late_goal_diff",
    "clean_sheet_trend_diff",
]

TACTICAL_CLUB_COLUMNS = [
    "club_shot_volume_diff",
    "club_conversion_diff",
    "club_defense_proxy_diff",
]


def international_tactical_diff(
    results: pd.DataFrame,
    goalscorers: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: pd.Timestamp,
    elo: EloSystem,
    *,
    prediction_context: str = "world_cup",
) -> dict[str, float]:
    pa = build_international_tactical_profile(
        results, goalscorers, team_a, before_date, elo, prediction_context=prediction_context
    )
    pb = build_international_tactical_profile(
        results, goalscorers, team_b, before_date, elo, prediction_context=prediction_context
    )
    return {
        "wc_form_diff": pa.wc_weighted_form - pb.wc_weighted_form,
        "mentality_diff": pa.mentality_index - pb.mentality_index,
        "clinical_diff": pa.clinical_index - pb.clinical_index,
        "late_goal_diff": pa.late_goal_share - pb.late_goal_share,
        "clean_sheet_trend_diff": pa.clean_sheet_rate - pb.clean_sheet_rate,
    }


def club_tactical_diff(
    merged_squads: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    prediction_context: str = "world_cup",
) -> dict[str, float]:
    from src.competition_weights import club_proxy_discount

    discount = club_proxy_discount(prediction_context)
    ca = build_club_tactical_proxy(merged_squads, team_a)
    cb = build_club_tactical_proxy(merged_squads, team_b)
    if ca is None or cb is None:
        return {c: 0.0 for c in TACTICAL_CLUB_COLUMNS}
    return {
        "club_shot_volume_diff": discount * (ca.shot_volume - cb.shot_volume),
        "club_conversion_diff": discount * (ca.shot_conversion - cb.shot_conversion),
        "club_defense_proxy_diff": discount * (ca.defensive_actions - cb.defensive_actions),
    }


def tactical_profile_to_dataframe(profile: InternationalTacticalProfile) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"指标": "世界杯周期加权状态", "数值": f"{profile.wc_weighted_form:.2f} 分/场权重"},
            {"指标": "大赛心态指数", "数值": f"{profile.mentality_index:.2f}"},
            {"指标": "进球效率 (临床)", "数值": f"{profile.clinical_index:.2f}"},
            {"指标": "晚场进球占比 (>75')", "数值": f"{profile.late_goal_share:.1%}"},
            {"指标": "淘汰赛场均得分", "数值": f"{profile.knockout_points_rate:.2f}"},
            {"指标": "硬仗得分率 (对强队)", "数值": f"{profile.big_game_points_rate:.2f}"},
            {"指标": "零封率", "数值": f"{profile.clean_sheet_rate:.1%}"},
            {"指标": "哑火率", "数值": f"{profile.failed_to_score_rate:.1%}"},
        ]
    )
