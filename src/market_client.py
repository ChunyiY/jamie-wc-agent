"""Factory for selecting the correct Polymarket market data client."""

from __future__ import annotations

from typing import Protocol

from src.compliance import MarketPlatform, resolve_market_platform
from src.config import AppConfig, load_config
from src.polymarket_public_client import PolymarketMarket, PolymarketPublicClient


class MarketDataClient(Protocol):
    platform: str

    def search_football_markets(
        self,
        keywords: list[str],
        *,
        min_liquidity: float = 0.0,
        active_only: bool = True,
        limit_per_keyword: int = 20,
    ) -> list[PolymarketMarket]: ...


def _platform_value(platform: MarketPlatform | str) -> str:
    if isinstance(platform, MarketPlatform):
        return platform.value
    return str(platform).lower().strip()


def get_market_client(
    config: AppConfig | None = None,
    *,
    platform: MarketPlatform | str | None = None,
    user_country: str | None = None,
    geoblock_country: str | None = None,
) -> MarketDataClient:
    """Return the Polymarket client for the resolved platform."""
    config = config or load_config()
    resolved = platform or resolve_market_platform(
        config,
        user_country=user_country,
        geoblock_country=geoblock_country,
    )
    if _platform_value(resolved) == MarketPlatform.US.value:
        from src.polymarket_us_client import PolymarketUSClient

        return PolymarketUSClient(config)
    return PolymarketPublicClient(config)
