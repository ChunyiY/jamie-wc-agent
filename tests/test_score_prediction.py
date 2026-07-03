"""Tests for Poisson scoreline prediction."""

from src.features import TeamFormSnapshot
from src.score_prediction import predict_scoreline


def _form(gf: float, ga: float) -> TeamFormSnapshot:
    return TeamFormSnapshot(
        wins5=2,
        draws5=1,
        losses5=2,
        wins10=4,
        draws10=2,
        losses10=4,
        goals_for_avg=gf,
        goals_against_avg=ga,
        clean_sheet_rate=0.2,
        failed_to_score_rate=0.2,
        avg_goal_diff=gf - ga,
        opponent_elo_avg=1500.0,
    )


def test_predict_scoreline_returns_most_likely():
    result = predict_scoreline(
        form_a=_form(1.8, 0.9),
        form_b=_form(1.1, 1.2),
        prob_team_a_win=0.50,
        prob_draw=0.28,
        prob_team_b_win=0.22,
    )
    assert result.most_likely.probability > 0
    assert sum(line.probability for line in result.top_scorelines) <= 1.0
    assert result.expected_goals_a > 0
    assert result.expected_goals_b > 0
    assert 0 <= result.over_25_prob <= 1
    assert len(result.score_matrix) == result.max_goals + 1


def test_strong_favorite_has_higher_expected_goals():
    favorite = predict_scoreline(
        form_a=_form(2.4, 0.6),
        form_b=_form(0.7, 1.8),
        prob_team_a_win=0.62,
        prob_draw=0.22,
        prob_team_b_win=0.16,
    )
    assert favorite.expected_goals_a > favorite.expected_goals_b
