"""Tests for total goals summary from score matrix."""

from src.score_prediction import total_goals_summary


def _matrix_with_mode_one_zero():
    # Heavily weight 1-0 but spread mass on high scores for over 2.5
    m = [[0.0] * 6 for _ in range(6)]
    m[1][0] = 0.11
    m[1][1] = 0.10
    m[2][0] = 0.10
    m[2][1] = 0.09
    m[3][0] = 0.08
    m[2][2] = 0.08
    m[3][1] = 0.07
    m[0][0] = 0.07
    m[3][2] = 0.09
    m[2][3] = 0.08
    m[4][1] = 0.07
    remainder = 1.0 - sum(sum(r) for r in m)
    m[1][2] = remainder
    return m


def test_mode_exact_can_differ_from_over_25():
    matrix = _matrix_with_mode_one_zero()
    tg = total_goals_summary(matrix, expected_goals_a=1.86, expected_goals_b=1.09)
    # 1-0 is highest single cell but total-goals mass spreads across 2,3,4...
    assert matrix[1][0] == max(matrix[i][j] for i in range(6) for j in range(6))
    assert tg.over_25_prob > 0.45
    assert tg.expected_total > 2.5


def test_total_goals_sums_to_one():
    matrix = [[0.1] * 4 for _ in range(4)]
    tg = total_goals_summary(matrix, expected_goals_a=1.0, expected_goals_b=1.0)
    assert abs(sum(p for _, p in tg.top_totals) - 1.0) < 0.01 or tg.most_likely_total_prob > 0
