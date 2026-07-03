"""Tests for strategy engine."""

from src.compliance import ComplianceStatus, TradingMode
from src.market_parser import MarketType, ParsedMarket
from src.polymarket_public_client import PolymarketMarket
from src.risk import RiskSettings
from src.strategy import Recommendation, StrategyInput, evaluate_strategy


def _compliance_ok() -> ComplianceStatus:
    from src.compliance import MarketPlatform

    return ComplianceStatus(
        mode=TradingMode.PAPER_TRADING,
        market_platform=MarketPlatform.INTERNATIONAL,
        jurisdiction_allowed=True,
        jurisdiction_reason="ok",
        geoblock_blocked=False,
        geoblock_country="CH",
        geoblock_region=None,
        geoblock_error=None,
        live_trading_enabled=False,
    )


def test_wide_spread_no_trade():
    market = PolymarketMarket(
        market_id="1",
        question="Will Brazil beat Argentina?",
        best_bid=0.40,
        best_ask=0.55,
        midpoint=0.475,
        spread=0.15,
        liquidity=5000,
    )
    parsed = ParsedMarket(
        market_type=MarketType.MATCH_WINNER,
        team_a="Brazil",
        team_b="Argentina",
        target_team="Brazil",
        target_outcome="yes",
        confidence=0.95,
        needs_manual_mapping=False,
        parse_notes="",
    )
    result = evaluate_strategy(
        StrategyInput(
            model_prob=0.65,
            model_confidence=0.7,
            market=market,
            parsed_market=parsed,
            compliance=_compliance_ok(),
            bankroll=10_000,
            risk_settings=RiskSettings(bankroll=10_000, max_spread=0.06),
        )
    )
    assert result.recommendation == Recommendation.NO_TRADE_WIDE_SPREAD


def test_value_long_yes_when_edge_high():
    market = PolymarketMarket(
        market_id="2",
        question="Will Brazil beat Argentina?",
        best_bid=0.48,
        best_ask=0.50,
        midpoint=0.49,
        spread=0.02,
        liquidity=5000,
    )
    parsed = ParsedMarket(
        market_type=MarketType.MATCH_WINNER,
        team_a="Brazil",
        team_b="Argentina",
        target_team="Brazil",
        target_outcome="yes",
        confidence=0.95,
        needs_manual_mapping=False,
        parse_notes="",
    )
    result = evaluate_strategy(
        StrategyInput(
            model_prob=0.72,
            model_confidence=0.7,
            market=market,
            parsed_market=parsed,
            compliance=_compliance_ok(),
            bankroll=10_000,
            risk_settings=RiskSettings(bankroll=10_000, min_edge_threshold=0.05, max_spread=0.06),
        )
    )
    assert result.recommendation == Recommendation.VALUE_LONG_YES
    assert result.suggested_paper_size > 0
