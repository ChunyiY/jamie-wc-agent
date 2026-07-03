"""Tests for fan betting guide."""

from src.betting_guide import build_betting_guide
from src.features import TeamFormSnapshot
from src.match_analyzer import MatchPrediction
from src.score_prediction import ScoreLine, ScorePrediction


def _prediction() -> MatchPrediction:
    sp = ScorePrediction(
        expected_goals_a=1.6,
        expected_goals_b=0.9,
        most_likely=ScoreLine(2, 1, 0.11),
        top_scorelines=[ScoreLine(2, 1, 0.11), ScoreLine(1, 1, 0.10)],
        score_matrix=[[0.1] * 4 for _ in range(4)],
        max_goals=3,
        over_25_prob=0.58,
        under_25_prob=0.42,
        btts_prob=0.52,
        note="test",
    )
    form = TeamFormSnapshot(2, 1, 2, 4, 2, 4, 1.5, 1.0, 0.2, 0.2, 0.5, 1500.0)
    return MatchPrediction(
        team_a="Brazil",
        team_b="Argentina",
        prob_team_a_win=0.48,
        prob_draw=0.27,
        prob_team_b_win=0.25,
        confidence=0.56,
        advance_prob_team_a=None,
        advance_prob_team_b=None,
        features={},
        form_a=form,
        form_b=form,
        explanation=[],
        score_prediction=sp,
    )


def test_betting_guide_includes_score_and_stake():
    guide = build_betting_guide(_prediction(), bankroll=70.0)
    assert guide.recommended_scoreline == "2–1"
    assert guide.recommended_total_goals > 0
    assert "Brazil" in guide.primary_outcome or guide.primary_outcome == "No clear lean"
    assert len(guide.cautions) >= 2
