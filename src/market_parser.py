"""Parse Polymarket market titles into football entities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.team_mapping import canonical_team_name, fuzzy_match_team


class MarketType(str, Enum):
    MATCH_WINNER = "match_winner"
    TEAM_TO_ADVANCE = "team_to_advance"
    TEAM_TO_WIN_TOURNAMENT = "team_to_win_tournament"
    GROUP_WINNER = "group_winner"
    OVER_UNDER_GOALS = "over_under_goals"
    UNKNOWN = "unknown"


@dataclass
class ParsedMarket:
    market_type: MarketType
    team_a: str | None
    team_b: str | None
    target_team: str | None
    target_outcome: str
    confidence: float
    needs_manual_mapping: bool
    parse_notes: str


PATTERNS: list[tuple[MarketType, re.Pattern[str], float]] = [
    (
        MarketType.MATCH_WINNER,
        re.compile(r"will\s+(?P<team_a>.+?)\s+beat\s+(?P<team_b>.+?)\??$", re.I),
        0.92,
    ),
    (
        MarketType.MATCH_WINNER,
        re.compile(r"(?P<team_a>.+?)\s+vs\.?\s+(?P<team_b>.+?)(?:\s+winner|\s+win|\?)?$", re.I),
        0.9,
    ),
    (
        MarketType.TEAM_TO_ADVANCE,
        re.compile(
            r"will\s+(?P<target>.+?)\s+advance(?:\s+to\s+(?:the\s+)?(?P<round>[\w\s]+))?\??$",
            re.I,
        ),
        0.9,
    ),
    (
        MarketType.TEAM_TO_WIN_TOURNAMENT,
        re.compile(r"will\s+(?P<target>.+?)\s+win\s+(?:the\s+)?(?P<tournament>.+?)\??$", re.I),
        0.88,
    ),
    (
        MarketType.MATCH_WINNER,
        re.compile(r"will\s+(?P<team_a>.+?)\s+win\s+(?:against|vs\.?)\s+(?P<team_b>.+?)\??$", re.I),
        0.9,
    ),
]


def parse_market_title(title: str, known_teams: list[str]) -> ParsedMarket:
    cleaned = " ".join(title.strip().split())

    for market_type, pattern, base_confidence in PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue

        groups = match.groupdict()
        if market_type == MarketType.MATCH_WINNER:
            team_a_raw = groups.get("team_a", "")
            team_b_raw = groups.get("team_b", "")
            team_a, conf_a = fuzzy_match_team(team_a_raw, known_teams)
            team_b, conf_b = fuzzy_match_team(team_b_raw, known_teams)
            confidence = min(base_confidence, conf_a, conf_b)
            return ParsedMarket(
                market_type=market_type,
                team_a=team_a,
                team_b=team_b,
                target_team=team_a,
                target_outcome="yes",
                confidence=confidence,
                needs_manual_mapping=confidence < 0.85 or not team_a or not team_b,
                parse_notes=f"Parsed as match winner: {team_a_raw} vs {team_b_raw}",
            )

        if market_type in {MarketType.TEAM_TO_ADVANCE, MarketType.TEAM_TO_WIN_TOURNAMENT}:
            target_raw = groups.get("target", "")
            target_team, conf = fuzzy_match_team(target_raw, known_teams)
            confidence = min(base_confidence, conf)
            return ParsedMarket(
                market_type=market_type,
                team_a=None,
                team_b=None,
                target_team=target_team,
                target_outcome="yes",
                confidence=confidence,
                needs_manual_mapping=confidence < 0.85 or not target_team,
                parse_notes=f"Parsed as {market_type.value}: {target_raw}",
            )

    # Low-confidence fallback: look for two known teams in title
    found_teams: list[tuple[str, float]] = []
    for team in known_teams:
        if re.search(rf"\b{re.escape(team)}\b", cleaned, re.I):
            found_teams.append((canonical_team_name(team), 0.8))

    if len(found_teams) >= 2:
        return ParsedMarket(
            market_type=MarketType.MATCH_WINNER,
            team_a=found_teams[0][0],
            team_b=found_teams[1][0],
            target_team=found_teams[0][0],
            target_outcome="yes",
            confidence=0.8,
            needs_manual_mapping=True,
            parse_notes="Inferred two teams from title with low confidence.",
        )

    if len(found_teams) == 1:
        return ParsedMarket(
            market_type=MarketType.TEAM_TO_WIN_TOURNAMENT,
            team_a=None,
            team_b=None,
            target_team=found_teams[0][0],
            target_outcome="yes",
            confidence=0.75,
            needs_manual_mapping=True,
            parse_notes="Inferred single team from title with low confidence.",
        )

    return ParsedMarket(
        market_type=MarketType.UNKNOWN,
        team_a=None,
        team_b=None,
        target_team=None,
        target_outcome="yes",
        confidence=0.0,
        needs_manual_mapping=True,
        parse_notes="Could not parse market title confidently.",
    )
