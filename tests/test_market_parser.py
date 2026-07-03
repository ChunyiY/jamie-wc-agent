"""Tests for market title parser."""

from src.market_parser import MarketType, parse_market_title


KNOWN = [
    "Brazil",
    "Argentina",
    "United States",
    "Belgium",
    "France",
    "Germany",
]


def test_parse_match_winner_beat():
    parsed = parse_market_title("Will USA beat Belgium?", KNOWN)
    assert parsed.market_type == MarketType.MATCH_WINNER
    assert parsed.team_a == "United States"
    assert parsed.team_b == "Belgium"
    assert parsed.confidence >= 0.85


def test_parse_advance_market():
    parsed = parse_market_title("Will United States advance to the quarterfinal?", KNOWN)
    assert parsed.market_type == MarketType.TEAM_TO_ADVANCE
    assert parsed.target_team == "United States"


def test_parse_vs_title_needs_review_when_low_confidence():
    parsed = parse_market_title("USA vs Belgium", KNOWN)
    assert parsed.market_type == MarketType.MATCH_WINNER
    assert parsed.team_a == "United States"
    assert parsed.team_b == "Belgium"


def test_unknown_title_manual_mapping():
    parsed = parse_market_title("Random market about weather", KNOWN)
    assert parsed.needs_manual_mapping is True
    assert parsed.confidence < 0.85
