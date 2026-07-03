"""Ensemble of Dixon–Coles, ordered logit, and multinomial logistic models."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src.dixon_coles import DixonColesParams, predict_dixon_coles
from src.model import FootballOutcomeModel
from src.ordered_logit import OrderedLogitModel
from src.score_prediction import ScorePrediction, refine_score_for_context


@dataclass
class ComponentProbabilities:
    dixon_coles: tuple[float, float, float]
    ordered_logit: tuple[float, float, float]
    multinomial: tuple[float, float, float]
    weights: tuple[float, float, float]


@dataclass
class EnsembleOutput:
    prob_team_a_win: float
    prob_draw: float
    prob_team_b_win: float
    confidence: float
    uncertainty: tuple[float, float, float]
    components: ComponentProbabilities
    score_prediction: ScorePrediction
    notes: list[str] = field(default_factory=list)


def _blend(
    p_dc: tuple[float, float, float],
    p_ol: tuple[float, float, float],
    p_ml: tuple[float, float, float],
    weights: tuple[float, float, float],
) -> np.ndarray:
    w_dc, w_ol, w_ml = weights
    vec = w_dc * np.array(p_dc) + w_ol * np.array(p_ol) + w_ml * np.array(p_ml)
    vec = np.clip(vec, 1e-9, 1.0)
    return vec / vec.sum()


def _uncertainty_from_components(
    blended: np.ndarray,
    p_dc: np.ndarray,
    p_ol: np.ndarray,
    p_ml: np.ndarray,
) -> tuple[float, float, float]:
    """90% interval half-width: blend of component spread and binomial SE."""
    spread = np.std(np.vstack([p_dc, p_ol, p_ml]), axis=0)
    binomial = np.sqrt(np.clip(blended * (1.0 - blended), 0.0, 0.25)) * 1.645
    half_width = 0.6 * spread + 0.4 * binomial
    return float(half_width[0]), float(half_width[1]), float(half_width[2])


@dataclass
class PredictionEnsemble:
    dixon_coles: DixonColesParams
    ordered_logit: OrderedLogitModel
    multinomial: FootballOutcomeModel
    weights: tuple[float, float, float] = (0.35, 0.30, 0.35)

    def predict(
        self,
        features: dict[str, float],
        team_a: str,
        team_b: str,
        *,
        neutral: bool = True,
        feature_vector: list[float] | None = None,
        feature_columns: list[str] | None = None,
        merged_squads: pd.DataFrame | None = None,
        form_a=None,
        form_b=None,
    ) -> EnsembleOutput:
        p_dc_tuple, score_pred = predict_dixon_coles(
            self.dixon_coles, team_a, team_b, neutral=neutral
        )
        p_dc = np.array(p_dc_tuple)

        cols = feature_columns or (self.multinomial.bundle.feature_columns if self.multinomial.bundle else [])
        if feature_vector is None and cols:
            x = np.array([[features[c] for c in cols]])
        else:
            x = np.array([feature_vector or []])
        p_ol = self.ordered_logit.predict_proba(x)[0]
        p_ml = self.multinomial.predict_proba(features) if self.multinomial.bundle else p_dc

        blended = _blend(tuple(p_dc), tuple(p_ol), tuple(p_ml), self.weights)
        unc = _uncertainty_from_components(blended, p_dc, p_ol, np.array(p_ml))

        score_pred = refine_score_for_context(
            score_pred,
            team_a=team_a,
            team_b=team_b,
            prob_team_a_win=float(blended[0]),
            prob_draw=float(blended[1]),
            prob_team_b_win=float(blended[2]),
            elo_diff=float(features.get("elo_diff", 0.0)),
            merged_squads=merged_squads,
            form_a=form_a,
            form_b=form_b,
            rho=self.dixon_coles.rho,
        )

        notes = [
            f"Ensemble weights — Dixon–Coles {self.weights[0]:.0%}, "
            f"Ordered logit {self.weights[1]:.0%}, Multinomial {self.weights[2]:.0%}.",
            "1X2 and score markets share the Dixon–Coles score matrix (Maher 1982; Dixon & Coles 1997).",
        ]

        return EnsembleOutput(
            prob_team_a_win=float(blended[0]),
            prob_draw=float(blended[1]),
            prob_team_b_win=float(blended[2]),
            confidence=float(np.max(blended)),
            uncertainty=unc,
            components=ComponentProbabilities(
                dixon_coles=tuple(p_dc),
                ordered_logit=tuple(p_ol),
                multinomial=tuple(p_ml),
                weights=self.weights,
            ),
            score_prediction=score_pred,
            notes=notes,
        )

    def save(self, path) -> None:
        joblib.dump(
            {
                "dixon_coles": self.dixon_coles,
                "ordered_logit": self.ordered_logit,
                "multinomial": self.multinomial.bundle,
                "weights": self.weights,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "PredictionEnsemble":
        data = joblib.load(path)
        ml = FootballOutcomeModel()
        ml.bundle = data["multinomial"]
        return cls(
            dixon_coles=data["dixon_coles"],
            ordered_logit=data["ordered_logit"],
            multinomial=ml,
            weights=tuple(data["weights"]),
        )
