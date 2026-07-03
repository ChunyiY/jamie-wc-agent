"""Dixon–Coles (1997) Poisson score model with low-score correlation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.score_prediction import ScoreLine, ScorePrediction
from src.team_mapping import canonical_team_name
from src.time_decay import match_sample_weights


@dataclass
class DixonColesParams:
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.24
    rho: float = -0.08
    league_avg: float = 1.35


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 1e-9:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_a * rho
    if x == 1 and y == 0:
        return 1.0 + lam_h * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def fit_dixon_coles(
    results: pd.DataFrame,
    *,
    sample_weights: np.ndarray | None = None,
    rho_grid: tuple[float, ...] = (-0.15, -0.10, -0.05, 0.0, 0.05),
) -> DixonColesParams:
    """Estimate attack/defence strengths and ρ via weighted historical goals."""
    frame = results.sort_values("date").copy()
    if frame.empty:
        return DixonColesParams()

    weights = sample_weights if sample_weights is not None else match_sample_weights(frame)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(frame):
        weights = match_sample_weights(frame)

    total_w = weights.sum()
    league_home = float(np.average(frame["home_score"], weights=weights))
    league_away = float(np.average(frame["away_score"], weights=weights))
    league_avg = max(0.8, (league_home + league_away) / 2.0)

    teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    attack: dict[str, float] = {}
    defense: dict[str, float] = {}

    for team in teams:
        home_mask = frame["home_team"] == team
        away_mask = frame["away_team"] == team
        gf = 0.0
        ga = 0.0
        w_gf = 0.0
        w_ga = 0.0
        if home_mask.any():
            w_h = weights[home_mask.to_numpy()]
            gf += float(np.sum(frame.loc[home_mask, "home_score"].to_numpy() * w_h))
            ga += float(np.sum(frame.loc[home_mask, "away_score"].to_numpy() * w_h))
            w_gf += float(w_h.sum())
            w_ga += float(w_h.sum())
        if away_mask.any():
            w_a = weights[away_mask.to_numpy()]
            gf += float(np.sum(frame.loc[away_mask, "away_score"].to_numpy() * w_a))
            ga += float(np.sum(frame.loc[away_mask, "home_score"].to_numpy() * w_a))
            w_gf += float(w_a.sum())
            w_ga += float(w_a.sum())
        attack[team] = max(0.35, (gf / max(w_gf, 1e-9)) / league_avg)
        defense[team] = max(0.35, (ga / max(w_ga, 1e-9)) / league_avg)

    home_adv = math.log(max(league_home, 0.5) / max(league_away, 0.5))
    home_adv = float(np.clip(home_adv, -0.1, 0.45))

    frame = frame.reset_index(drop=True)
    weights = np.asarray(weights[: len(frame)], dtype=float)

    best_rho = -0.08
    best_ll = float("inf")
    for rho in rho_grid:
        ll = 0.0
        for i in range(len(frame)):
            row = frame.iloc[i]
            w = weights[i]
            lam_h, lam_a = expected_lambdas(
                DixonColesParams(attack, defense, home_adv, rho, league_avg),
                row["home_team"],
                row["away_team"],
                neutral=bool(row.get("neutral", False)),
            )
            x, y = int(row["home_score"]), int(row["away_score"])
            p = score_probability(lam_h, lam_a, x, y, rho, max_goals=max(x, y, 5))
            ll -= w * math.log(max(p, 1e-12))
        if ll < best_ll:
            best_ll = ll
            best_rho = float(rho)

    return DixonColesParams(
        attack=attack,
        defense=defense,
        home_advantage=home_adv,
        rho=best_rho,
        league_avg=league_avg,
    )


def expected_lambdas(
    params: DixonColesParams,
    team_a: str,
    team_b: str,
    *,
    neutral: bool,
) -> tuple[float, float]:
    team_a = canonical_team_name(team_a)
    team_b = canonical_team_name(team_b)
    att_a = params.attack.get(team_a, 1.0)
    def_b = params.defense.get(team_b, 1.0)
    att_b = params.attack.get(team_b, 1.0)
    def_a = params.defense.get(team_a, 1.0)
    home_boost = 0.0 if neutral else params.home_advantage
    lam_a = params.league_avg * att_a * def_b * math.exp(home_boost)
    lam_b = params.league_avg * att_b * def_a
    return max(0.2, lam_a), max(0.2, lam_b)


def score_probability(
    lam_h: float,
    lam_a: float,
    goals_h: int,
    goals_a: int,
    rho: float,
    *,
    max_goals: int = 5,
) -> float:
    if goals_h > max_goals or goals_a > max_goals:
        return 0.0
    base = _poisson_pmf(goals_h, lam_h) * _poisson_pmf(goals_a, lam_a)
    return max(0.0, base * _tau(goals_h, goals_a, lam_h, lam_a, rho))


def build_score_matrix(
    lam_h: float,
    lam_a: float,
    rho: float,
    *,
    max_goals: int = 5,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            row.append(score_probability(lam_h, lam_a, i, j, rho, max_goals=max_goals))
        matrix.append(row)
    total = sum(sum(r) for r in matrix)
    if total > 0:
        matrix = [[c / total for c in row] for row in matrix]
    return matrix


def outcome_probs_from_matrix(matrix: list[list[float]]) -> tuple[float, float, float]:
    p_a = p_d = p_b = 0.0
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            if i > j:
                p_a += prob
            elif i == j:
                p_d += prob
            else:
                p_b += prob
    return p_a, p_d, p_b


def predict_dixon_coles(
    params: DixonColesParams,
    team_a: str,
    team_b: str,
    *,
    neutral: bool = True,
    max_goals: int = 5,
    top_n: int = 8,
) -> tuple[tuple[float, float, float], ScorePrediction]:
    lam_a, lam_b = expected_lambdas(params, team_a, team_b, neutral=neutral)
    matrix = build_score_matrix(lam_a, lam_b, params.rho, max_goals=max_goals)
    p_a, p_d, p_b = outcome_probs_from_matrix(matrix)

    scorelines: list[ScoreLine] = []
    over_25 = btts = 0.0
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            scorelines.append(ScoreLine(goals_a=i, goals_b=j, probability=prob))
            if i + j >= 3:
                over_25 += prob
            if i >= 1 and j >= 1:
                btts += prob
    scorelines.sort(key=lambda s: s.probability, reverse=True)

    sp = ScorePrediction(
        expected_goals_a=lam_a,
        expected_goals_b=lam_b,
        most_likely=scorelines[0],
        top_scorelines=scorelines[:top_n],
        score_matrix=matrix,
        max_goals=max_goals,
        over_25_prob=over_25,
        under_25_prob=1.0 - over_25,
        btts_prob=btts,
        note="Dixon–Coles (1997) Poisson model with low-score ρ correction.",
    )
    return (p_a, p_d, p_b), sp
