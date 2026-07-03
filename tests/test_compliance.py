"""Tests for compliance module."""

from src.compliance import (
    COMPLIANCE_BANNER,
    MarketPlatform,
    TradingMode,
    check_jurisdiction_allowed,
    evaluate_compliance,
    place_live_limit_order_stub,
    resolve_market_platform,
)
from src.config import AppConfig


def test_compliance_banner_present():
    assert "paper-trading research" in COMPLIANCE_BANNER.lower()
    assert "does not guarantee profit" in COMPLIANCE_BANNER.lower()


def test_blocked_jurisdiction_us_on_international_platform():
    allowed, reason = check_jurisdiction_allowed("US", market_platform=MarketPlatform.INTERNATIONAL)
    assert allowed is False
    assert "restricted" in reason.lower()


def test_us_platform_allows_us_users():
    allowed, reason = check_jurisdiction_allowed("US", market_platform=MarketPlatform.US)
    assert allowed is True
    assert "polymarket us" in reason.lower()


def test_resolve_market_platform_defaults_to_us_for_us_country():
    config = AppConfig(user_country_code="US")
    assert resolve_market_platform(config) == MarketPlatform.US


def test_unknown_jurisdiction_defaults_conservative_on_international():
    config = AppConfig(user_country_code=None)
    allowed, reason = check_jurisdiction_allowed("", config=config, market_platform=MarketPlatform.INTERNATIONAL)
    assert allowed is False
    assert "unknown" in reason.lower()


def test_evaluate_compliance_us_platform_allows_paper_trading():
    config = AppConfig(enable_live_trading=False, polymarket_platform="us", user_country_code="US")
    status = evaluate_compliance(user_confirmed_jurisdiction="US", config=config)
    assert status.market_platform == MarketPlatform.US
    assert status.jurisdiction_allowed is True
    assert status.is_restricted is False
    assert status.can_trade_live is False
    assert status.mode in {TradingMode.PAPER_TRADING, TradingMode.LIVE_TRADING_DISABLED}


def test_evaluate_compliance_read_only_when_blocked_international():
    config = AppConfig(enable_live_trading=False, polymarket_platform="international", user_country_code="US")
    status = evaluate_compliance(user_confirmed_jurisdiction="US", config=config)
    assert status.is_restricted
    assert status.can_trade_live is False
    assert status.mode == TradingMode.READ_ONLY_ANALYTICS


def test_live_trading_stub_raises():
    config = AppConfig(enable_live_trading=False)
    try:
        place_live_limit_order_stub(
            market_id="1",
            side="yes",
            price=0.5,
            size=10,
            user_confirmed=True,
            config=config,
        )
        raised = False
    except NotImplementedError:
        raised = True
    assert raised
