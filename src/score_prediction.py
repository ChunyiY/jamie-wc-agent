"""Poisson scoreline prediction aligned to W/D/L model probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.features import TeamFormSnapshot


@dataclass
class ScoreLine:
    goals_a: int
    goals_b: int
    probability: float

    @property
    def label(self) -> str:
        return f"{self.goals_a}–{self.goals_b}"


@dataclass
class TotalGoalsSummary:
    """Distribution of total goals (home + away) from the score matrix."""

    most_likely_total: int
    most_likely_total_prob: float
    expected_total: float
    under_25_prob: float
    over_25_prob: float
    top_totals: list[tuple[int, float]]  # (total_goals, probability)


def total_goals_summary(
    score_matrix: list[list[float]],
    *,
    expected_goals_a: float,
    expected_goals_b: float,
) -> TotalGoalsSummary:
    """Aggregate exact-score matrix into total-goals distribution."""
    totals: dict[int, float] = {}
    for i, row in enumerate(score_matrix):
        for j, prob in enumerate(row):
            t = i + j
            totals[t] = totals.get(t, 0.0) + prob
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    mode_total, mode_prob = ranked[0] if ranked else (0, 0.0)
    over_25 = sum(p for t, p in totals.items() if t >= 3)
    return TotalGoalsSummary(
        most_likely_total=mode_total,
        most_likely_total_prob=mode_prob,
        expected_total=expected_goals_a + expected_goals_b,
        under_25_prob=1.0 - over_25,
        over_25_prob=over_25,
        top_totals=ranked[:5],
    )


@dataclass
class ScorePrediction:
    expected_goals_a: float
    expected_goals_b: float
    most_likely: ScoreLine
    top_scorelines: list[ScoreLine]
    score_matrix: list[list[float]]
    max_goals: int
    over_25_prob: float
    under_25_prob: float
    btts_prob: float
    note: str


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 1e-9:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _estimate_base_lambdas(form_a: TeamFormSnapshot, form_b: TeamFormSnapshot) -> tuple[float, float]:
    """Expected goals from recent attack/defence form (last 10 matches)."""
    lam_a = (form_a.goals_for_avg + form_b.goals_against_avg) / 2.0
    lam_b = (form_b.goals_for_avg + form_a.goals_against_avg) / 2.0
    return max(0.35, lam_a), max(0.35, lam_b)


def _outcome_probs_from_matrix(matrix: list[list[float]]) -> tuple[float, float, float]:
    p_a = p_d = p_b = 0.0
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            if i > j:
                p_a += prob
            elif i == j:
                p_d += prob
            else:
                p_b += prob
    total = p_a + p_d + p_b
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return p_a / total, p_d / total, p_b / total


def _build_score_matrix(lam_a: float, lam_b: float, *, max_goals: int = 5) -> list[list[float]]:
    rows: list[list[float]] = []
    for i in range(max_goals + 1):
        row = [_poisson_pmf(i, lam_a) * _poisson_pmf(j, lam_b) for j in range(max_goals + 1)]
        rows.append(row)
    total = sum(sum(r) for r in rows)
    if total > 0:
        rows = [[cell / total for cell in row] for row in rows]
    return rows


def _align_lambdas_to_outcomes(
    lam_a: float,
    lam_b: float,
    target_a: float,
    target_d: float,
    target_b: float,
    *,
    max_goals: int = 5,
    iterations: int = 24,
) -> tuple[float, float]:
    """Scale lambdas so the score matrix implies similar W/D/L as the outcome model."""
    for _ in range(iterations):
        matrix = _build_score_matrix(lam_a, lam_b, max_goals=max_goals)
        imp_a, imp_d, imp_b = _outcome_probs_from_matrix(matrix)
        if imp_a < 1e-9 or imp_b < 1e-9:
            break
        lam_a *= (target_a / imp_a) ** 0.35
        lam_b *= (target_b / imp_b) ** 0.35
        # Nudge draw via slight total-goals adjustment
        draw_ratio = target_d / max(imp_d, 1e-9)
        scale = draw_ratio**0.15
        lam_a *= scale
        lam_b *= scale
        lam_a = max(0.25, min(lam_a, 3.5))
        lam_b = max(0.25, min(lam_b, 3.5))
    return lam_a, lam_b


def predict_scoreline(
    *,
    form_a: TeamFormSnapshot,
    form_b: TeamFormSnapshot,
    prob_team_a_win: float,
    prob_draw: float,
    prob_team_b_win: float,
    max_goals: int = 5,
    top_n: int = 8,
) -> ScorePrediction:
    """
    Predict exact scorelines using independent Poisson goals, aligned to W/D/L probs.
    """
    lam_a, lam_b = _estimate_base_lambdas(form_a, form_b)
    lam_a, lam_b = _align_lambdas_to_outcomes(
        lam_a,
        lam_b,
        prob_team_a_win,
        prob_draw,
        prob_team_b_win,
        max_goals=max_goals,
    )
    matrix = _build_score_matrix(lam_a, lam_b, max_goals=max_goals)

    scorelines: list[ScoreLine] = []
    over_25 = 0.0
    btts = 0.0
    for i, row in enumerate(matrix):
        for j, prob in enumerate(row):
            scorelines.append(ScoreLine(goals_a=i, goals_b=j, probability=prob))
            if i + j >= 3:
                over_25 += prob
            if i >= 1 and j >= 1:
                btts += prob

    scorelines.sort(key=lambda s: s.probability, reverse=True)
    most_likely = scorelines[0]

    return ScorePrediction(
        expected_goals_a=lam_a,
        expected_goals_b=lam_b,
        most_likely=most_likely,
        top_scorelines=scorelines[:top_n],
        score_matrix=matrix,
        max_goals=max_goals,
        over_25_prob=over_25,
        under_25_prob=1.0 - over_25,
        btts_prob=btts,
        note=(
            "Poisson score model using recent goals for/against, "
            "aligned to the win/draw/loss probabilities above. Not a guarantee."
        ),
    )


def _mismatch_lambda_adjustments(
    lam_a: float,
    lam_b: float,
    *,
    elo_diff: float,
    team_a: str,
    team_b: str,
    merged_squads: pd.DataFrame | None,
) -> tuple[float, float]:
    """Boost favorite xG when Elo / squad gap suggests a mismatch (e.g. Argentina vs Cape Verde)."""
    if abs(elo_diff) < 200:
        return lam_a, lam_b

    boost = min(0.38, abs(elo_diff) / 1100.0)
    if merged_squads is not None and not merged_squads.empty:
        from src.squad_data import team_squad_profile

        pa = team_squad_profile(merged_squads, team_a)
        pb = team_squad_profile(merged_squads, team_b)
        if pa and pb and pa.top11_market_value_eur > 0 and pb.top11_market_value_eur > 0:
            ratio = pa.top11_market_value_eur / pb.top11_market_value_eur
            if elo_diff > 0 and ratio >= 3:
                boost += min(0.14, math.log(ratio) * 0.045)
            elif elo_diff < 0 and ratio <= 1 / 3:
                boost += min(0.14, math.log(1 / ratio) * 0.045)

    if elo_diff > 0:
        return lam_a * (1.0 + boost), lam_b * max(0.65, 1.0 - boost * 0.4)
    return lam_a * max(0.65, 1.0 - boost * 0.4), lam_b * (1.0 + boost)


def refine_score_for_context(
    sp: ScorePrediction,
    *,
    team_a: str,
    team_b: str,
    prob_team_a_win: float,
    prob_draw: float,
    prob_team_b_win: float,
    elo_diff: float = 0.0,
    merged_squads: pd.DataFrame | None = None,
    form_a: TeamFormSnapshot | None = None,
    form_b: TeamFormSnapshot | None = None,
    rho: float = -0.08,
    max_goals: int = 5,
    top_n: int = 8,
) -> ScorePrediction:
    """Re-estimate score matrix using mismatch-aware lambdas, aligned to 1X2 probs."""
    from src.dixon_coles import build_score_matrix

    lam_a, lam_b = sp.expected_goals_a, sp.expected_goals_b

    if form_a is not None and form_b is not None:
        base_a, base_b = _estimate_base_lambdas(form_a, form_b)
        lam_a = 0.55 * lam_a + 0.45 * base_a
        lam_b = 0.55 * lam_b + 0.45 * base_b

    lam_a, lam_b = _mismatch_lambda_adjustments(
        lam_a, lam_b, elo_diff=elo_diff, team_a=team_a, team_b=team_b, merged_squads=merged_squads
    )
    matrix = build_score_matrix(lam_a, lam_b, rho, max_goals=max_goals)

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

    note = sp.note
    if abs(elo_diff) >= 200:
        note += " Mismatch adjustment applied (Elo/squad gap)."

    return ScorePrediction(
        expected_goals_a=lam_a,
        expected_goals_b=lam_b,
        most_likely=scorelines[0],
        top_scorelines=scorelines[:top_n],
        score_matrix=matrix,
        max_goals=max_goals,
        over_25_prob=over_25,
        under_25_prob=1.0 - over_25,
        btts_prob=btts,
        note=note,
    )
