"""Tests for competition-context weighting."""

import pandas as pd
import pytest

from src.competition_weights import (
    CompetitionTier,
    club_proxy_discount,
    tier_form_multiplier,
    tournament_tier,
    training_sample_weights,
)


def test_world_cup_tier_highest():
    assert tournament_tier("FIFA World Cup") == CompetitionTier.WORLD_CUP
    assert tournament_tier("Friendly") == CompetitionTier.FRIENDLY


def test_form_weights_favour_world_cup_context():
    wc = tier_form_multiplier(CompetitionTier.WORLD_CUP, prediction_context="world_cup")
    fr = tier_form_multiplier(CompetitionTier.FRIENDLY, prediction_context="world_cup")
    assert wc > fr


def test_club_proxy_discounted_for_world_cup():
    assert club_proxy_discount("world_cup") < club_proxy_discount("default")


def test_training_weights_nonzero():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="90D", tz="UTC"),
            "tournament": [
                "FIFA World Cup",
                "Friendly",
                "UEFA Euro qualification",
                "Friendly",
                "Copa America",
            ],
        }
    )
    w = training_sample_weights(frame, prediction_context="world_cup")
    assert len(w) == 5
    assert w.sum() == pytest.approx(len(w), rel=0.01)
