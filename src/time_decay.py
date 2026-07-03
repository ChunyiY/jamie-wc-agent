"""Time decay and tournament importance sample weights for training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.competition_weights import training_sample_weights


def match_sample_weights(
    results: pd.DataFrame,
    *,
    reference_date: pd.Timestamp | None = None,
    half_life_days: float = 900.0,
    min_weight: float = 0.15,
    prediction_context: str = "default",
) -> np.ndarray:
    """
    Exponential time-decay × competition tier (Dixon–Coles 1997; Rue & Salvesen 2000).

    Pass prediction_context='world_cup' when fitting models aimed at World Cup forecasting
  (shorter half-life, higher tier weights on competitive internationals).
    """
    _ = half_life_days  # legacy kwarg; context selects half-life in competition_weights
    return training_sample_weights(
        results,
        reference_date=reference_date,
        prediction_context=prediction_context,
        min_weight=min_weight,
    )
