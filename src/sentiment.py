"""Optional sentiment module — neutral fallback when keys are unavailable."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SentimentResult:
    score: float
    label: str
    source: str
    available: bool
    note: str


def get_match_sentiment(team_a: str, team_b: str, tournament: str = "") -> SentimentResult:
    """
    Return neutral sentiment unless optional integrations are configured.

    Sentiment is informational only and should never dominate the football model.
    """
    twitter_key = os.getenv("TWITTER_BEARER_TOKEN")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")

    if not twitter_key and not hf_key:
        return SentimentResult(
            score=0.0,
            label="neutral",
            source="none",
            available=False,
            note="Sentiment integrations not configured. Returning neutral sentiment.",
        )

    # Phase 3 placeholder: keep neutral until explicit sentiment pipeline is implemented.
    return SentimentResult(
        score=0.0,
        label="neutral",
        source="stub",
        available=False,
        note="Sentiment keys detected, but sentiment weighting is disabled in Phase 1.",
    )
