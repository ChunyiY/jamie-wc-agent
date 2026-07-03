"""High-level match analysis combining Elo, features, and ensemble model output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.elo import EloSystem
from src.ensemble import ComponentProbabilities, PredictionEnsemble
from src.features import FEATURE_COLUMNS, TeamFormSnapshot, build_match_features, compute_team_form
from src.model import FootballOutcomeModel
from src.score_prediction import ScorePrediction, predict_scoreline
from src.shootouts import knockout_advance_probs
from src.team_mapping import canonical_team_name


@dataclass
class MatchPrediction:
    team_a: str
    team_b: str
    prob_team_a_win: float
    prob_draw: float
    prob_team_b_win: float
    confidence: float
    advance_prob_team_a: float | None
    advance_prob_team_b: float | None
    features: dict[str, float]
    form_a: TeamFormSnapshot
    form_b: TeamFormSnapshot
    explanation: list[str]
    score_prediction: ScorePrediction | None = None
    component_probs: ComponentProbabilities | None = None
    uncertainty: tuple[float, float, float] | None = None


class MatchAnalyzer:
    def __init__(
        self,
        results: pd.DataFrame,
        elo: EloSystem,
        model: FootballOutcomeModel,
        ensemble: PredictionEnsemble | None = None,
        goalscorers: pd.DataFrame | None = None,
        merged_squads: pd.DataFrame | None = None,
        shootouts: pd.DataFrame | None = None,
        intl_match_stats: pd.DataFrame | None = None,
        fifa_rankings: pd.DataFrame | None = None,
    ):
        self.results = results
        self.elo = elo
        self.model = model
        self.ensemble = ensemble
        self.goalscorers = goalscorers if goalscorers is not None else pd.DataFrame()
        self.merged_squads = merged_squads if merged_squads is not None else pd.DataFrame()
        self.shootouts = shootouts if shootouts is not None else pd.DataFrame()
        self.intl_match_stats = intl_match_stats if intl_match_stats is not None else pd.DataFrame()
        self.fifa_rankings = fifa_rankings if fifa_rankings is not None else pd.DataFrame()

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        *,
        match_date: pd.Timestamp | None = None,
        neutral: bool = False,
        tournament: str = "FIFA World Cup",
        stage: str = "group",
        knockout: bool = False,
    ) -> MatchPrediction:
        team_a = canonical_team_name(team_a)
        team_b = canonical_team_name(team_b)
        match_date = match_date or pd.Timestamp.utcnow()

        features = build_match_features(
            self.results,
            team_a,
            team_b,
            match_date=match_date,
            elo=self.elo,
            neutral=neutral,
            tournament=tournament,
            stage=stage,
            goalscorers=self.goalscorers,
            merged_squads=self.merged_squads,
            prediction_context="world_cup" if "world cup" in tournament.lower() else "default",
            intl_match_stats=self.intl_match_stats if not self.intl_match_stats.empty else None,
            fifa_rankings=self.fifa_rankings if not self.fifa_rankings.empty else None,
        )
        form_a = compute_team_form(self.results, team_a, match_date, self.elo)
        form_b = compute_team_form(self.results, team_b, match_date, self.elo)

        component_probs = None
        uncertainty = None
        score_pred = None

        if self.ensemble is not None:
            output = self.ensemble.predict(
                features,
                team_a,
                team_b,
                neutral=neutral,
                merged_squads=self.merged_squads if not self.merged_squads.empty else None,
                form_a=form_a,
                form_b=form_b,
            )
            probs = {
                "team_a": output.prob_team_a_win,
                "draw": output.prob_draw,
                "team_b": output.prob_team_b_win,
            }
            confidence = output.confidence
            component_probs = output.components
            uncertainty = output.uncertainty
            score_pred = output.score_prediction
            explanation = self._build_ensemble_explanation(
                team_a, team_b, features, form_a, form_b, probs, confidence, output
            )
        elif self.model.bundle is None:
            probs = self._elo_baseline_probs(team_a, team_b, neutral)
            confidence = max(probs.values()) * 0.75
            explanation = [
                "Model not trained yet; showing Elo-based baseline probabilities.",
                f"Elo difference ({team_a} - {team_b}): {features['elo_diff']:.1f}",
            ]
            score_pred = predict_scoreline(
                form_a=form_a,
                form_b=form_b,
                prob_team_a_win=probs["team_a"],
                prob_draw=probs["draw"],
                prob_team_b_win=probs["team_b"],
            )
        else:
            probabilities = self.model.predict_proba(features)
            probs = {
                "team_a": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "team_b": float(probabilities[2]),
            }
            confidence = self.model.confidence_from_probs(probabilities)
            explanation = self._build_explanation(team_a, team_b, features, form_a, form_b, probs, confidence)
            score_pred = predict_scoreline(
                form_a=form_a,
                form_b=form_b,
                prob_team_a_win=probs["team_a"],
                prob_draw=probs["draw"],
                prob_team_b_win=probs["team_b"],
            )

        advance_a = advance_b = None
        if knockout or str(stage).lower() != "group":
            if not self.shootouts.empty:
                advance_a, advance_b, pk_meta = knockout_advance_probs(
                    probs["team_a"],
                    probs["draw"],
                    probs["team_b"],
                    team_a=team_a,
                    team_b=team_b,
                    shootouts=self.shootouts,
                )
                explanation.append(
                    f"Knockout advance uses ET + penalty history "
                    f"(PK edge {team_a}: {pk_meta.get('pk_rate_a', 0.5):.1%})."
                )
            else:
                advance_a, advance_b = self._approximate_advance_probs(
                    probs["team_a"], probs["draw"], probs["team_b"]
                )
                explanation.append(
                    "Advance probability is approximate (download shootouts.csv for PK model)."
                )

        return MatchPrediction(
            team_a=team_a,
            team_b=team_b,
            prob_team_a_win=probs["team_a"],
            prob_draw=probs["draw"],
            prob_team_b_win=probs["team_b"],
            confidence=confidence,
            advance_prob_team_a=advance_a,
            advance_prob_team_b=advance_b,
            features=features,
            form_a=form_a,
            form_b=form_b,
            explanation=explanation,
            score_prediction=score_pred,
            component_probs=component_probs,
            uncertainty=uncertainty,
        )

    def target_probability(
        self,
        prediction: MatchPrediction,
        *,
        market_type: str,
        target_team: str | None = None,
    ) -> float:
        market_type = market_type.lower()
        target_team = canonical_team_name(target_team) if target_team else None

        if market_type == "match_winner":
            if target_team == prediction.team_a:
                return prediction.prob_team_a_win
            if target_team == prediction.team_b:
                return prediction.prob_team_b_win
            raise ValueError("Match winner market requires target_team to be team_a or team_b.")

        if market_type == "team_to_advance":
            if target_team == prediction.team_a:
                return prediction.advance_prob_team_a or prediction.prob_team_a_win
            if target_team == prediction.team_b:
                return prediction.advance_prob_team_b or prediction.prob_team_b_win
            raise ValueError("Advance market requires target_team.")

        if market_type == "team_to_win_tournament":
            elo_a = self.elo.get_rating(prediction.team_a)
            elo_b = self.elo.get_rating(prediction.team_b)
            if target_team == prediction.team_a:
                base = prediction.prob_team_a_win
                return float(min(0.35, base * 0.5 + (elo_a / 3000)))
            if target_team == prediction.team_b:
                base = prediction.prob_team_b_win
                return float(min(0.35, base * 0.5 + (elo_b / 3000)))
            raise ValueError("Tournament winner market requires target_team.")

        raise ValueError(f"Unsupported market type: {market_type}")

    def _elo_baseline_probs(self, team_a: str, team_b: str, neutral: bool) -> dict[str, float]:
        from src.elo import expected_score

        rating_a = self.elo.get_rating(team_a)
        rating_b = self.elo.get_rating(team_b)
        if not neutral:
            rating_a += self.elo.config.elo_home_advantage
        win_a = expected_score(rating_a, rating_b)
        win_b = expected_score(rating_b, rating_a)
        draw = max(0.05, 1.0 - abs(win_a - win_b))
        total = win_a + win_b + draw
        return {
            "team_a": win_a / total,
            "draw": draw / total,
            "team_b": win_b / total,
        }

    def _approximate_advance_probs(self, p_a: float, p_draw: float, p_b: float) -> tuple[float, float]:
        advance_a = p_a + 0.5 * p_draw
        advance_b = p_b + 0.5 * p_draw
        total = advance_a + advance_b
        return advance_a / total, advance_b / total

    def _build_explanation(
        self,
        team_a: str,
        team_b: str,
        features: dict[str, float],
        form_a: TeamFormSnapshot,
        form_b: TeamFormSnapshot,
        probs: dict[str, float],
        confidence: float,
    ) -> list[str]:
        lines = [
            f"Elo edge favors {team_a if features['elo_diff'] >= 0 else team_b} "
            f"(difference {features['elo_diff']:.1f}).",
            f"Recent form (last 5): {team_a} {form_a.wins5}W-{form_a.draws5}D-{form_a.losses5}L vs "
            f"{team_b} {form_b.wins5}W-{form_b.draws5}D-{form_b.losses5}L.",
            f"Attack trend: {team_a} GF avg {form_a.goals_for_avg:.2f}, {team_b} GF avg {form_b.goals_for_avg:.2f}.",
            f"Defense trend: {team_a} GA avg {form_a.goals_against_avg:.2f}, {team_b} GA avg {form_b.goals_against_avg:.2f}.",
            f"Model probabilities — {team_a}: {probs['team_a']:.1%}, draw: {probs['draw']:.1%}, {team_b}: {probs['team_b']:.1%}.",
            f"Confidence level: {confidence:.1%} (not a guarantee).",
        ]
        return lines

    def _build_ensemble_explanation(
        self,
        team_a: str,
        team_b: str,
        features: dict[str, float],
        form_a: TeamFormSnapshot,
        form_b: TeamFormSnapshot,
        probs: dict[str, float],
        confidence: float,
        output,
    ) -> list[str]:
        lines = self._build_explanation(team_a, team_b, features, form_a, form_b, probs, confidence)
        comp = output.components
        w = comp.weights
        lines.extend(
            [
                f"Ensemble blend — Dixon–Coles {w[0]:.0%}, ordered logit {w[1]:.0%}, multinomial {w[2]:.0%}.",
                f"Dixon–Coles: {comp.dixon_coles[0]:.1%} / {comp.dixon_coles[1]:.1%} / {comp.dixon_coles[2]:.1%}.",
                f"Ordered logit: {comp.ordered_logit[0]:.1%} / {comp.ordered_logit[1]:.1%} / {comp.ordered_logit[2]:.1%}.",
                f"Multinomial: {comp.multinomial[0]:.1%} / {comp.multinomial[1]:.1%} / {comp.multinomial[2]:.1%}.",
            ]
        )
        if output.uncertainty:
            u = output.uncertainty
            lines.append(
                f"Approx. 90% uncertainty half-widths — {team_a}: ±{u[0]:.1%}, draw: ±{u[1]:.1%}, {team_b}: ±{u[2]:.1%}."
            )
        for note in output.notes:
            lines.append(note)
        return lines
