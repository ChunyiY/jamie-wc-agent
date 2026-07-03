"""Feature engineering for international football match prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.elo import EloSystem
from src.player_stats import recent_attack_feature_diff
from src.squad_data import SQUAD_FEATURE_COLUMNS, squad_strength_features
from src.team_mapping import canonical_team_name
from src.team_tactics import (
    TACTICAL_CLUB_COLUMNS,
    TACTICAL_INTL_COLUMNS,
    club_tactical_diff,
    international_tactical_diff,
)
from src.match_stats import MATCH_QUALITY_COLUMNS, match_quality_feature_diff

FEATURE_COLUMNS = [
    "elo_diff",
    "home_advantage_flag",
    "neutral_venue",
    "tournament_weight",
    "knockout_stage",
    "team_a_form5_wins",
    "team_a_form5_draws",
    "team_a_form5_losses",
    "team_b_form5_wins",
    "team_b_form5_draws",
    "team_b_form5_losses",
    "team_a_form10_wins",
    "team_a_form10_draws",
    "team_a_form10_losses",
    "team_b_form10_wins",
    "team_b_form10_draws",
    "team_b_form10_losses",
    "team_a_gf_avg",
    "team_a_ga_avg",
    "team_b_gf_avg",
    "team_b_ga_avg",
    "team_a_clean_sheet_rate",
    "team_b_clean_sheet_rate",
    "team_a_failed_to_score_rate",
    "team_b_failed_to_score_rate",
    "team_a_avg_goal_diff",
    "team_b_avg_goal_diff",
    "team_a_opponent_elo_avg",
    "team_b_opponent_elo_avg",
    "rest_diff_days",
] + TACTICAL_INTL_COLUMNS + SQUAD_FEATURE_COLUMNS + TACTICAL_CLUB_COLUMNS + MATCH_QUALITY_COLUMNS

# Training uses international tactics only; club + static squad are prediction-time overlays.
TRAINING_FEATURE_COLUMNS = [
    c
    for c in FEATURE_COLUMNS
    if c not in SQUAD_FEATURE_COLUMNS and c not in TACTICAL_CLUB_COLUMNS and c not in MATCH_QUALITY_COLUMNS
]

EloLookupKey = tuple[Any, str, str]


@dataclass
class TeamFormSnapshot:
    wins5: int
    draws5: int
    losses5: int
    wins10: int
    draws10: int
    losses10: int
    goals_for_avg: float
    goals_against_avg: float
    clean_sheet_rate: float
    failed_to_score_rate: float
    avg_goal_diff: float
    opponent_elo_avg: float


def _team_results_before_date(
    results: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
) -> pd.DataFrame:
    team = canonical_team_name(team)
    mask = (
        (results["date"] < before_date)
        & ((results["home_team"] == team) | (results["away_team"] == team))
    )
    return results.loc[mask].sort_values("date")


def _result_points_for_team(row: pd.Series, team: str) -> float:
    if row["home_team"] == team:
        if row["home_score"] > row["away_score"]:
            return 3.0
        if row["home_score"] == row["away_score"]:
            return 1.0
        return 0.0
    if row["away_score"] > row["home_score"]:
        return 3.0
    if row["away_score"] == row["home_score"]:
        return 1.0
    return 0.0


def _team_goals(row: pd.Series, team: str) -> tuple[int, int]:
    if row["home_team"] == team:
        return int(row["home_score"]), int(row["away_score"])
    return int(row["away_score"]), int(row["home_score"])


def _opponent_elo_at_row(
    row: pd.Series,
    team: str,
    elo_lookup: dict[EloLookupKey, tuple[float, float]] | None,
    elo: EloSystem,
) -> float:
    if elo_lookup is not None:
        key = (row["date"], row["home_team"], row["away_team"])
        if key in elo_lookup:
            home_before, away_before = elo_lookup[key]
            return away_before if row["home_team"] == team else home_before
    opponent = row["away_team"] if row["home_team"] == team else row["home_team"]
    return elo.get_rating(opponent)


def elo_lookup_from_history(history: pd.DataFrame) -> dict[EloLookupKey, tuple[float, float]]:
    """Build a lookup table from an Elo fit_history dataframe."""
    lookup: dict[EloLookupKey, tuple[float, float]] = {}
    for _, row in history.iterrows():
        key = (row["date"], row["home_team"], row["away_team"])
        lookup[key] = (float(row["home_elo_before"]), float(row["away_elo_before"]))
    return lookup


def _form_from_history_window(
    history: pd.DataFrame,
    team: str,
    elo: EloSystem,
    *,
    elo_lookup: dict[EloLookupKey, tuple[float, float]] | None = None,
) -> TeamFormSnapshot:
    if history.empty:
        return TeamFormSnapshot(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, elo.get_rating(team))

    last10 = history.tail(10)
    last5 = history.tail(5)

    def summarize(window: pd.DataFrame) -> tuple[int, int, int]:
        wins = draws = losses = 0
        for _, row in window.iterrows():
            points = _result_points_for_team(row, team)
            if points == 3:
                wins += 1
            elif points == 1:
                draws += 1
            else:
                losses += 1
        return wins, draws, losses

    wins5, draws5, losses5 = summarize(last5)
    wins10, draws10, losses10 = summarize(last10)

    goals_for: list[int] = []
    goals_against: list[int] = []
    clean_sheets = 0
    failed_to_score = 0
    opponent_elos: list[float] = []

    for _, row in last10.iterrows():
        gf, ga = _team_goals(row, team)
        goals_for.append(gf)
        goals_against.append(ga)
        if ga == 0:
            clean_sheets += 1
        if gf == 0:
            failed_to_score += 1
        opponent_elos.append(_opponent_elo_at_row(row, team, elo_lookup, elo))

    n = max(len(last10), 1)
    return TeamFormSnapshot(
        wins5=wins5,
        draws5=draws5,
        losses5=losses5,
        wins10=wins10,
        draws10=draws10,
        losses10=losses10,
        goals_for_avg=float(np.mean(goals_for)) if goals_for else 0.0,
        goals_against_avg=float(np.mean(goals_against)) if goals_against else 0.0,
        clean_sheet_rate=clean_sheets / n,
        failed_to_score_rate=failed_to_score / n,
        avg_goal_diff=float(np.mean(np.array(goals_for) - np.array(goals_against))) if goals_for else 0.0,
        opponent_elo_avg=float(np.mean(opponent_elos)) if opponent_elos else elo.get_rating(team),
    )


def compute_team_form(
    results: pd.DataFrame,
    team: str,
    before_date: pd.Timestamp,
    elo: EloSystem,
    *,
    elo_lookup: dict[EloLookupKey, tuple[float, float]] | None = None,
) -> TeamFormSnapshot:
    history = _team_results_before_date(results, team, before_date)
    return _form_from_history_window(history, team, elo, elo_lookup=elo_lookup)


def _rest_days(results: pd.DataFrame, team: str, before_date: pd.Timestamp) -> float | None:
    history = _team_results_before_date(results, team, before_date)
    if history.empty:
        return None
    last_match_date = history.iloc[-1]["date"]
    return float((before_date - last_match_date).days)


def build_elo_lookup(results: pd.DataFrame, elo: EloSystem | None = None) -> dict[EloLookupKey, tuple[float, float]]:
    """Precompute per-match Elo ratings before each result update."""
    system = elo or EloSystem()
    history = system.fit_history(results.sort_values("date"))
    lookup: dict[EloLookupKey, tuple[float, float]] = {}
    for _, row in history.iterrows():
        key = (row["date"], row["home_team"], row["away_team"])
        lookup[key] = (float(row["home_elo_before"]), float(row["away_elo_before"]))
    return lookup


def build_match_features(
    results: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    match_date: pd.Timestamp,
    elo: EloSystem,
    neutral: bool = False,
    tournament: str = "Friendly",
    stage: str = "group",
    days_since_last_match_home: float | None = None,
    days_since_last_match_away: float | None = None,
    elo_lookup: dict[EloLookupKey, tuple[float, float]] | None = None,
    goalscorers: pd.DataFrame | None = None,
    merged_squads: pd.DataFrame | None = None,
    prediction_context: str = "world_cup",
    intl_match_stats: pd.DataFrame | None = None,
    fifa_rankings: pd.DataFrame | None = None,
) -> dict[str, float]:
    team_a = canonical_team_name(team_a)
    team_b = canonical_team_name(team_b)

    form_a = compute_team_form(results, team_a, match_date, elo, elo_lookup=elo_lookup)
    form_b = compute_team_form(results, team_b, match_date, elo, elo_lookup=elo_lookup)

    rest_a = days_since_last_match_home
    rest_b = days_since_last_match_away
    if rest_a is None:
        rest_a = _rest_days(results, team_a, match_date)
    if rest_b is None:
        rest_b = _rest_days(results, team_b, match_date)

    rest_diff = 0.0
    if rest_a is not None and rest_b is not None:
        rest_diff = float(rest_a - rest_b)

    from src.elo import tournament_importance

    features = {
        "elo_diff": elo.get_rating(team_a) - elo.get_rating(team_b),
        "home_advantage_flag": 0.0 if neutral else 1.0,
        "neutral_venue": 1.0 if neutral else 0.0,
        "tournament_weight": tournament_importance(tournament),
        "knockout_stage": 1.0 if str(stage).lower() in {"knockout", "round of 16", "quarterfinal", "semifinal", "final"} else 0.0,
        "team_a_form5_wins": float(form_a.wins5),
        "team_a_form5_draws": float(form_a.draws5),
        "team_a_form5_losses": float(form_a.losses5),
        "team_b_form5_wins": float(form_b.wins5),
        "team_b_form5_draws": float(form_b.draws5),
        "team_b_form5_losses": float(form_b.losses5),
        "team_a_form10_wins": float(form_a.wins10),
        "team_a_form10_draws": float(form_a.draws10),
        "team_a_form10_losses": float(form_a.losses10),
        "team_b_form10_wins": float(form_b.wins10),
        "team_b_form10_draws": float(form_b.draws10),
        "team_b_form10_losses": float(form_b.losses10),
        "team_a_gf_avg": form_a.goals_for_avg,
        "team_a_ga_avg": form_a.goals_against_avg,
        "team_b_gf_avg": form_b.goals_for_avg,
        "team_b_ga_avg": form_b.goals_against_avg,
        "team_a_clean_sheet_rate": form_a.clean_sheet_rate,
        "team_b_clean_sheet_rate": form_b.clean_sheet_rate,
        "team_a_failed_to_score_rate": form_a.failed_to_score_rate,
        "team_b_failed_to_score_rate": form_b.failed_to_score_rate,
        "team_a_avg_goal_diff": form_a.avg_goal_diff,
        "team_b_avg_goal_diff": form_b.avg_goal_diff,
        "team_a_opponent_elo_avg": form_a.opponent_elo_avg,
        "team_b_opponent_elo_avg": form_b.opponent_elo_avg,
        "rest_diff_days": rest_diff,
    }
    gs = goalscorers if goalscorers is not None else pd.DataFrame()
    if not gs.empty:
        features.update(recent_attack_feature_diff(gs, team_a, team_b, match_date))
    else:
        features.update({"recent_attack_diff": 0.0, "recent_scorers_diff": 0.0, "recent_concentration_diff": 0.0})

    if not results.empty:
        features.update(
            international_tactical_diff(
                results, gs, team_a, team_b, match_date, elo, prediction_context=prediction_context
            )
        )
    else:
        features.update({c: 0.0 for c in TACTICAL_INTL_COLUMNS})

    if merged_squads is not None and not merged_squads.empty:
        features.update(
            squad_strength_features(merged_squads, team_a, team_b, prediction_context=prediction_context)
        )
        features.update(
            club_tactical_diff(merged_squads, team_a, team_b, prediction_context=prediction_context)
        )
    else:
        for col in SQUAD_FEATURE_COLUMNS + TACTICAL_CLUB_COLUMNS:
            features[col] = 0.0

    stats = intl_match_stats if intl_match_stats is not None else pd.DataFrame()
    ranks = fifa_rankings if fifa_rankings is not None else pd.DataFrame()
    if not stats.empty:
        features.update(
            match_quality_feature_diff(
                stats, ranks, team_a, team_b, match_date, prediction_context=prediction_context
            )
        )
    else:
        for col in MATCH_QUALITY_COLUMNS:
            features[col] = 0.0
    return features


def build_training_frame_point_in_time(
    results: pd.DataFrame,
    *,
    elo_lookup: dict[EloLookupKey, tuple[float, float]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    goalscorers: pd.DataFrame | None = None,
    merged_squads: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build training features with point-in-time Elo and form.

    Uses incremental per-team history (O(n)) instead of scanning the full table each row.
    """
    from collections import defaultdict

    from src.elo import tournament_importance

    frame = results.sort_values("date").reset_index(drop=True)
    lookup = elo_lookup or build_elo_lookup(frame)
    elo = EloSystem()
    rows: list[dict[str, float | int | pd.Timestamp | str]] = []
    team_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_match_date: dict[str, pd.Timestamp] = {}
    total = len(frame)

    for idx, row in frame.iterrows():
        home = canonical_team_name(row["home_team"])
        away = canonical_team_name(row["away_team"])
        match_date = row["date"]

        home_hist = pd.DataFrame(team_history[home]) if team_history[home] else pd.DataFrame()
        away_hist = pd.DataFrame(team_history[away]) if team_history[away] else pd.DataFrame()
        form_a = _form_from_history_window(home_hist, home, elo, elo_lookup=lookup)
        form_b = _form_from_history_window(away_hist, away, elo, elo_lookup=lookup)

        rest_a = float((match_date - last_match_date[home]).days) if home in last_match_date else row.get("days_since_last_match_home")
        rest_b = float((match_date - last_match_date[away]).days) if away in last_match_date else row.get("days_since_last_match_away")
        rest_diff = 0.0
        if rest_a is not None and rest_b is not None:
            rest_diff = float(rest_a) - float(rest_b)

        neutral = bool(row.get("neutral", False))
        tournament = str(row.get("tournament", "Friendly"))
        stage = str(row.get("stage", "group"))

        features: dict[str, float | int | pd.Timestamp | str] = {
            "elo_diff": elo.get_rating(home) - elo.get_rating(away),
            "home_advantage_flag": 0.0 if neutral else 1.0,
            "neutral_venue": 1.0 if neutral else 0.0,
            "tournament_weight": tournament_importance(tournament),
            "knockout_stage": 1.0 if stage.lower() in {"knockout", "round of 16", "quarterfinal", "semifinal", "final"} else 0.0,
            "team_a_form5_wins": float(form_a.wins5),
            "team_a_form5_draws": float(form_a.draws5),
            "team_a_form5_losses": float(form_a.losses5),
            "team_b_form5_wins": float(form_b.wins5),
            "team_b_form5_draws": float(form_b.draws5),
            "team_b_form5_losses": float(form_b.losses5),
            "team_a_form10_wins": float(form_a.wins10),
            "team_a_form10_draws": float(form_a.draws10),
            "team_a_form10_losses": float(form_a.losses10),
            "team_b_form10_wins": float(form_b.wins10),
            "team_b_form10_draws": float(form_b.draws10),
            "team_b_form10_losses": float(form_b.losses10),
            "team_a_gf_avg": form_a.goals_for_avg,
            "team_a_ga_avg": form_a.goals_against_avg,
            "team_b_gf_avg": form_b.goals_for_avg,
            "team_b_ga_avg": form_b.goals_against_avg,
            "team_a_clean_sheet_rate": form_a.clean_sheet_rate,
            "team_b_clean_sheet_rate": form_b.clean_sheet_rate,
            "team_a_failed_to_score_rate": form_a.failed_to_score_rate,
            "team_b_failed_to_score_rate": form_b.failed_to_score_rate,
            "team_a_avg_goal_diff": form_a.avg_goal_diff,
            "team_b_avg_goal_diff": form_b.avg_goal_diff,
            "team_a_opponent_elo_avg": form_a.opponent_elo_avg,
            "team_b_opponent_elo_avg": form_b.opponent_elo_avg,
            "rest_diff_days": rest_diff,
            "date": match_date,
            "team_a": row["home_team"],
            "team_b": row["away_team"],
            "outcome": int(row["outcome"]),
        }
        gs = goalscorers if goalscorers is not None else pd.DataFrame()
        if not gs.empty:
            features.update(recent_attack_feature_diff(gs, home, away, match_date))
        else:
            features.update({"recent_attack_diff": 0.0, "recent_scorers_diff": 0.0, "recent_concentration_diff": 0.0})
        features.update(
            international_tactical_diff(frame, gs, home, away, match_date, elo, prediction_context="default")
        )
        # No squad / club / match-quality proxies in training (point-in-time international only).
        for col in SQUAD_FEATURE_COLUMNS + TACTICAL_CLUB_COLUMNS + MATCH_QUALITY_COLUMNS:
            features[col] = 0.0
        rows.append(features)

        row_record = row.to_dict()
        for team in (home, away):
            team_history[team].append(row_record)
            if len(team_history[team]) > 10:
                team_history[team] = team_history[team][-10:]
            last_match_date[team] = match_date

        elo.update_match(
            row["home_team"],
            row["away_team"],
            int(row["home_score"]),
            int(row["away_score"]),
            neutral=neutral,
            tournament=tournament,
        )

        if progress_callback and (idx + 1) % 200 == 0:
            progress_callback(idx + 1, total)

    if progress_callback and total:
        progress_callback(total, total)

    return pd.DataFrame(rows)


def build_training_frame(results: pd.DataFrame, elo: EloSystem | None = None) -> pd.DataFrame:
    """Backward-compatible alias using point-in-time feature generation."""
    _ = elo
    return build_training_frame_point_in_time(results)
