"""Tests for market client factory."""

from src.compliance import MarketPlatform
from src.config import AppConfig
from src.market_client import get_market_client
from src.polymarket_public_client import PolymarketPublicClient
from src.polymarket_us_client import PolymarketUSClient


def test_get_market_client_returns_us_client_for_enum():
    config = AppConfig(polymarket_platform="us")
    client = get_market_client(config, platform=MarketPlatform.US)
    assert isinstance(client, PolymarketUSClient)
    assert client.platform == "us"


def test_get_market_client_returns_international_by_default():
    config = AppConfig(polymarket_platform="international", user_country_code="CH")
    client = get_market_client(config)
    assert isinstance(client, PolymarketPublicClient)
    assert client.platform == "international"
