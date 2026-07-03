"""
Competition-context weighting for international football models.

Literature basis
----------------
* Dixon & Coles (1997), J. Royal Stat. Soc. C — time-decayed Poisson goals; not all
  matches carry equal information.
* Rue & Salvesen (2000) — prediction and retrospective analysis with exponential
  decay; recent competitive fixtures matter more than distant friendlies.
* Baio & Blangiardo (2010), J. Applied Stat. — hierarchical models separating
  competition levels; we approximate this via tier multipliers.
* Groll et al. (2014), AStA Adv. Stat. Anal. — FIFA World Cup forecasting using
  FIFA ranking + player/club covariates; club strength is informative but must be
  down-weighted vs international results when predicting national-team outcomes.
* Leitner et al. (2010) — ensemble / ranking approaches for World Cups; major
  tournament cycles dominate short-run friendly form.

Design
------
1. **Training sample weights** — time decay × competition tier (for Dixon–Coles,
   ordered logit, multinomial).
2. **Form weights** — when building team form for a *target* competition (e.g.
   World Cup), up-weight WC, qualifiers, continental finals; down-weight friendlies.
3. **Club proxy discount** — squad per-90 features are scaled down at prediction
   time because club league competitiveness ≠ international match intensity.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

# Half-lives (days) — Rue & Salvesen style; shorter for WC-focused context.
DEFAULT_HALF_LIFE_DAYS = 900.0
WORLD_CUP_HALF_LIFE_DAYS = 540.0


class CompetitionTier(IntEnum):
  """Higher tier = more informative for national-team prediction."""

  FRIENDLY = 1
  MINOR = 2
  QUALIFIER = 3
  CONTINENTAL = 4
  WORLD_CUP = 5


TIER_MULTIPLIERS: dict[CompetitionTier, float] = {
    CompetitionTier.FRIENDLY: 0.45,
    CompetitionTier.MINOR: 0.65,
    CompetitionTier.QUALIFIER: 0.90,
    CompetitionTier.CONTINENTAL: 1.05,
    CompetitionTier.WORLD_CUP: 1.35,
}

# When *predicting* a World Cup match, form should emphasise these tiers.
FORM_TIER_WORLD_CUP: dict[CompetitionTier, float] = {
    CompetitionTier.FRIENDLY: 0.35,
    CompetitionTier.MINOR: 0.55,
    CompetitionTier.QUALIFIER: 1.00,
    CompetitionTier.CONTINENTAL: 1.10,
    CompetitionTier.WORLD_CUP: 1.40,
}

# Club per-90 covariates are discounted vs international signals (Groll et al. 2014).
CLUB_PROXY_WEIGHT: dict[str, float] = {
    "default": 0.55,
    "world_cup": 0.28,
    "continental": 0.40,
}


def tournament_tier(tournament: str) -> CompetitionTier:
    key = str(tournament).lower().strip()
    if "world cup" in key or "fifa world cup" in key:
        return CompetitionTier.WORLD_CUP
    if any(x in key for x in ("euro", "copa america", "afcon", "asian cup", "gold cup")):
        return CompetitionTier.CONTINENTAL
    if "nations league" in key:
        return CompetitionTier.MINOR
    if any(x in key for x in ("qualif", "qualification", "prelim")):
        return CompetitionTier.QUALIFIER
    if "friendly" in key or key in {"", "nan"}:
        return CompetitionTier.FRIENDLY
    return CompetitionTier.MINOR


def tier_training_multiplier(tier: CompetitionTier) -> float:
    return TIER_MULTIPLIERS[tier]


def tier_form_multiplier(tier: CompetitionTier, *, prediction_context: str = "default") -> float:
    if prediction_context == "world_cup":
        return FORM_TIER_WORLD_CUP[tier]
    return TIER_MULTIPLIERS[tier]


def club_proxy_discount(prediction_context: str = "default") -> float:
    return CLUB_PROXY_WEIGHT.get(prediction_context, CLUB_PROXY_WEIGHT["default"])


def _time_decay(age_days: np.ndarray, half_life_days: float) -> np.ndarray:
    age_days = np.clip(age_days, 0.0, None)
    return np.power(0.5, age_days / half_life_days)


def match_row_weight(
    tournament: str,
    age_days: float,
    *,
    prediction_context: str = "default",
    half_life_days: float | None = None,
) -> float:
    """Single-match weight for form aggregation."""
    tier = tournament_tier(tournament)
    hl = half_life_days or (
        WORLD_CUP_HALF_LIFE_DAYS if prediction_context == "world_cup" else DEFAULT_HALF_LIFE_DAYS
    )
    time_w = float(_time_decay(np.array([age_days]), hl)[0])
    tier_w = tier_form_multiplier(tier, prediction_context=prediction_context)
    return time_w * tier_w


def training_sample_weights(
    results: pd.DataFrame,
    *,
    reference_date: pd.Timestamp | None = None,
    prediction_context: str = "default",
    half_life_days: float | None = None,
    min_weight: float = 0.12,
) -> np.ndarray:
    """
    Combined time × competition tier weights for model fitting.

    For World Cup–focused retraining, pass prediction_context='world_cup' to
    emphasise recent competitive internationals (Rue & Salvesen; Baio–Blangiardo).
    """
    if results.empty:
        return np.array([])

    frame = results.copy()
    dates = pd.to_datetime(frame["date"], utc=True)
    ref = reference_date or dates.max()
    if ref.tzinfo is None:
        ref = ref.tz_localize("UTC")

    hl = half_life_days or (
        WORLD_CUP_HALF_LIFE_DAYS if prediction_context == "world_cup" else DEFAULT_HALF_LIFE_DAYS
    )
    age_days = ((ref - dates).dt.total_seconds() / 86400.0).clip(lower=0.0).to_numpy()
    time_w = _time_decay(age_days, hl)

    tournaments = frame.get("tournament", pd.Series(["Friendly"] * len(frame))).astype(str)
    tier_w = tournaments.map(
        lambda t: tier_training_multiplier(tournament_tier(t))
    ).to_numpy(dtype=float)

    weights = time_w * tier_w
    weights = np.clip(weights, min_weight, None)
    weights *= len(weights) / max(weights.sum(), 1e-9)
    return weights
