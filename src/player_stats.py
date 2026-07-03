"""International player statistics: active squad, WC roster merge, attack proxies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.squad_data import SquadPlayer, TeamSquadProfile, merge_squads_per90, team_squad_profile
from src.player_matching import best_name_match
from src.team_mapping import canonical_team_name

ACTIVE_MONTHS = 18
RECENT_MONTHS = 12


@dataclass
class PlayerRecord:
    name: str
    team: str
    position: str
    club: str
    age: int | None
    market_value_eur: float | None
    goals_per90: float | None
    assists_per90: float | None
    shots_per90: float | None
    rating: float | None
    intl_goals_career: int
    intl_goals_12m: int
    intl_goals_24m: int
    intl_penalties: int
    avg_goal_minute: float | None
    last_intl_goal: pd.Timestamp | None
    contribution_score: float
    status: str


@dataclass
class TeamPlayerSummary:
    team: str
    active_players: int
    squad_players: int
    top_contributors: list[PlayerRecord] = field(default_factory=list)
    wc_squad: TeamSquadProfile | None = None


@dataclass
class HeadToHeadScorer:
    scorer: str
    team: str
    goals: int
    last_goal_date: pd.Timestamp | None


def load_goalscorers(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "team", "scorer", "minute", "own_goal", "penalty"]
        )
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    for col in ("home_team", "away_team", "team", "scorer"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str).map(canonical_team_name)
    for col in ("own_goal", "penalty"):
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.upper().isin({"TRUE", "1", "YES"})
        else:
            frame[col] = False
    if "minute" in frame.columns:
        frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce")
    return frame.dropna(subset=["date", "team", "scorer"]).sort_values("date").reset_index(drop=True)


def recent_intl_attack_stats(
    goalscorers: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    *,
    months: int = RECENT_MONTHS,
) -> dict[str, float]:
    """Point-in-time international attack proxy from goal scorers."""
    team = canonical_team_name(team)
    if goalscorers.empty:
        return {"recent_goals": 0.0, "recent_scorers": 0.0, "top_scorer_share": 0.0, "attack_depth": 0.0}

    if before_date.tzinfo is None:
        before_date = before_date.tz_localize("UTC")
    cutoff = before_date - pd.Timedelta(days=months * 30)

    subset = goalscorers[
        (goalscorers["team"] == team)
        & (goalscorers["date"] < before_date)
        & (goalscorers["date"] >= cutoff)
        & (~goalscorers["own_goal"])
    ]
    if subset.empty:
        return {"recent_goals": 0.0, "recent_scorers": 0.0, "top_scorer_share": 0.0, "attack_depth": 0.0}

    total = len(subset)
    by_scorer = subset.groupby("scorer").size()
    top_share = float(by_scorer.max() / total) if total else 0.0
    depth = float((by_scorer >= 2).sum())
    return {
        "recent_goals": float(total),
        "recent_scorers": float(by_scorer.shape[0]),
        "top_scorer_share": top_share,
        "attack_depth": depth,
    }


def recent_attack_feature_diff(
    goalscorers: pd.DataFrame,
    team_a: str,
    team_b: str,
    before_date: pd.Timestamp,
) -> dict[str, float]:
    sa = recent_intl_attack_stats(goalscorers, team_a, before_date)
    sb = recent_intl_attack_stats(goalscorers, team_b, before_date)
    return {
        "recent_attack_diff": sa["recent_goals"] - sb["recent_goals"],
        "recent_scorers_diff": sa["recent_scorers"] - sb["recent_scorers"],
        "recent_concentration_diff": sa["top_scorer_share"] - sb["top_scorer_share"],
    }


def _intl_stats_for_player(
    goalscorers: pd.DataFrame,
    team: str,
    player_name: str,
    *,
    reference_date: pd.Timestamp | None = None,
    squad_names: list[str] | None = None,
) -> dict:
    team = canonical_team_name(team)
    ref = reference_date or goalscorers["date"].max()
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")
    lookup_name = player_name
    if squad_names:
        lookup_name = best_name_match(player_name, squad_names) or player_name
    subset = goalscorers[(goalscorers["team"] == team) & (goalscorers["scorer"] == lookup_name)]
    if subset.empty and lookup_name != player_name:
        subset = goalscorers[(goalscorers["team"] == team) & (goalscorers["scorer"] == player_name)]
    if subset.empty:
        return {
            "career": 0,
            "g12": 0,
            "g24": 0,
            "pens": 0,
            "avg_min": None,
            "last": None,
            "active": False,
        }
    cut12 = ref - pd.Timedelta(days=365)
    cut24 = ref - pd.Timedelta(days=730)
    cut_active = ref - pd.Timedelta(days=ACTIVE_MONTHS * 30)
    last = subset["date"].max()
    return {
        "career": len(subset),
        "g12": int((subset["date"] >= cut12).sum()),
        "g24": int((subset["date"] >= cut24).sum()),
        "pens": int(subset["penalty"].sum()),
        "avg_min": float(subset["minute"].dropna().mean()) if subset["minute"].notna().any() else None,
        "last": last,
        "active": bool(last >= cut_active),
    }


def _contribution_score(
    *,
    g12: int,
    g24: int,
    goals_per90: float | None,
    assists_per90: float | None,
    market_value: float | None,
    rating: float | None,
) -> float:
    score = g12 * 3.0 + g24 * 1.0
    if goals_per90 is not None:
        score += goals_per90 * 8.0
    if assists_per90 is not None:
        score += assists_per90 * 4.0
    if rating is not None:
        score += rating * 2.0
    if market_value is not None:
        score += min(market_value / 5_000_000, 5.0)
    return round(score, 2)


def build_team_player_summary(
    team: str,
    goalscorers: pd.DataFrame,
    merged_squads: pd.DataFrame,
    *,
    reference_date: pd.Timestamp | None = None,
    top_n: int = 15,
) -> TeamPlayerSummary:
    team = canonical_team_name(team)
    ref = reference_date or pd.Timestamp.now(tz="UTC")
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    profile = team_squad_profile(merged_squads, team) if not merged_squads.empty else None
    records: list[PlayerRecord] = []

    if profile:
        squad_names = [sp.name for sp in profile.players]
        for sp in profile.players:
            ist = _intl_stats_for_player(
                goalscorers, team, sp.name, reference_date=ref, squad_names=squad_names
            )
            status = "现役" if ist["active"] or ist["g24"] > 0 or sp.age < 36 else "久未入选"
            if not ist["active"] and ist["g24"] == 0 and sp.age >= 36:
                status = "老将/替补"
            records.append(
                PlayerRecord(
                    name=sp.name,
                    team=team,
                    position=sp.position,
                    club=sp.club,
                    age=sp.age,
                    market_value_eur=sp.market_value_eur,
                    goals_per90=sp.goals_per90,
                    assists_per90=sp.assists_per90,
                    shots_per90=sp.shots_per90,
                    rating=sp.rating,
                    intl_goals_career=ist["career"],
                    intl_goals_12m=ist["g12"],
                    intl_goals_24m=ist["g24"],
                    intl_penalties=ist["pens"],
                    avg_goal_minute=ist["avg_min"],
                    last_intl_goal=ist["last"],
                    contribution_score=_contribution_score(
                        g12=ist["g12"],
                        g24=ist["g24"],
                        goals_per90=sp.goals_per90,
                        assists_per90=sp.assists_per90,
                        market_value=sp.market_value_eur,
                        rating=sp.rating,
                    ),
                    status=status,
                )
            )
    else:
        cut = ref - pd.Timedelta(days=ACTIVE_MONTHS * 30)
        subset = goalscorers[(goalscorers["team"] == team) & (goalscorers["date"] >= cut)]
        for scorer, group in subset.groupby("scorer"):
            ist = _intl_stats_for_player(goalscorers, team, str(scorer), reference_date=ref)
            records.append(
                PlayerRecord(
                    name=str(scorer),
                    team=team,
                    position="—",
                    club="—",
                    age=None,
                    market_value_eur=None,
                    goals_per90=None,
                    assists_per90=None,
                    shots_per90=None,
                    rating=None,
                    intl_goals_career=ist["career"],
                    intl_goals_12m=ist["g12"],
                    intl_goals_24m=ist["g24"],
                    intl_penalties=ist["pens"],
                    avg_goal_minute=ist["avg_min"],
                    last_intl_goal=ist["last"],
                    contribution_score=float(ist["g12"] * 3 + ist["g24"]),
                    status="现役",
                )
            )

    records.sort(key=lambda r: (r.contribution_score, r.intl_goals_12m), reverse=True)
    active = [r for r in records if r.status in {"现役", "老将/替补"} and (r.intl_goals_24m > 0 or r.goals_per90)]
    if not active:
        active = [r for r in records if r.intl_goals_24m > 0][:top_n]
    if not active:
        active = records[:top_n]

    return TeamPlayerSummary(
        team=team,
        active_players=len(active),
        squad_players=profile.squad_size if profile else len(records),
        top_contributors=active[:top_n],
        wc_squad=profile,
    )


def team_player_summary(
    goalscorers: pd.DataFrame,
    team: str,
    *,
    merged_squads: pd.DataFrame | None = None,
    reference_date: pd.Timestamp | None = None,
    top_n: int = 12,
) -> TeamPlayerSummary:
    return build_team_player_summary(
        team,
        goalscorers,
        merged_squads if merged_squads is not None else pd.DataFrame(),
        reference_date=reference_date,
        top_n=top_n,
    )


def head_to_head_scorers(
    goalscorers: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    limit: int = 10,
    since_years: int = 15,
) -> list[HeadToHeadScorer]:
    team_a = canonical_team_name(team_a)
    team_b = canonical_team_name(team_b)
    cutoff = goalscorers["date"].max() - pd.Timedelta(days=since_years * 365)
    mask = (
        (
            ((goalscorers["home_team"] == team_a) & (goalscorers["away_team"] == team_b))
            | ((goalscorers["home_team"] == team_b) & (goalscorers["away_team"] == team_a))
        )
        & (goalscorers["date"] >= cutoff)
        & goalscorers["team"].isin([team_a, team_b])
        & (~goalscorers["own_goal"])
    )
    subset = goalscorers[mask]
    rows: list[HeadToHeadScorer] = []
    for (scorer, team), group in subset.groupby(["scorer", "team"]):
        rows.append(
            HeadToHeadScorer(
                scorer=str(scorer),
                team=str(team),
                goals=len(group),
                last_goal_date=group["date"].max(),
            )
        )
    rows.sort(key=lambda r: (r.goals, r.last_goal_date or pd.Timestamp.min), reverse=True)
    return rows[:limit]


def players_to_dataframe(records: list[PlayerRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    def _fmt_money(v: float | None) -> str:
        if v is None:
            return "—"
        if v >= 1_000_000:
            return f"€{v / 1_000_000:.1f}M"
        return f"€{v / 1000:.0f}K"

    return pd.DataFrame(
        [
            {
                "球员": r.name,
                "位置": r.position,
                "俱乐部": r.club,
                "年龄": r.age if r.age else "—",
                "RT估值": _fmt_money(r.market_value_eur),
                "俱乐部进球/90": f"{r.goals_per90:.2f}" if r.goals_per90 is not None else "—",
                "俱乐部助攻/90": f"{r.assists_per90:.2f}" if r.assists_per90 is not None else "—",
                "射门/90": f"{r.shots_per90:.2f}" if r.shots_per90 is not None else "—",
                "评分": f"{r.rating:.1f}" if r.rating is not None else "—",
                "国家队12月": r.intl_goals_12m,
                "国家队24月": r.intl_goals_24m,
                "点球": r.intl_penalties,
                "最近进球": r.last_intl_goal.strftime("%Y-%m-%d") if r.last_intl_goal is not None else "—",
                "贡献指数": r.contribution_score,
                "状态": r.status,
            }
            for r in records
        ]
    )


def h2h_scorers_to_dataframe(rows: list[HeadToHeadScorer]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "球员": r.scorer,
                "球队": r.team,
                "近15年交锋进球": r.goals,
                "最近进球": r.last_goal_date.strftime("%Y-%m-%d") if r.last_goal_date is not None else "—",
            }
            for r in rows
        ]
    )
