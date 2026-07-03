"""Read-only Polymarket US client (regulated US platform)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from src.config import AppConfig, load_config
from src.polymarket_public_client import PolymarketMarket
from src.utils import safe_float, setup_logger

try:
    from polymarket_us import PolymarketUS
except ImportError:  # pragma: no cover - optional dependency at import time
    PolymarketUS = None  # type: ignore[misc, assignment]


class PolymarketUSClient:
    """Fetch public market data from the regulated Polymarket US API."""

    FOOTBALL_KEYWORDS = (
        "football",
        "soccer",
        "fifa",
        "world cup",
        "uefa",
        "copa",
        "premier league",
        "la liga",
        "champions league",
        "qualifier",
        "round of 16",
        "quarter",
        "semifinal",
    )

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.logger = setup_logger("polymarket_us_client")
        if PolymarketUS is None:
            raise ImportError(
                "polymarket-us is not installed. Run: pip install polymarket-us"
            )
        self._client = PolymarketUS(
            key_id=self.config.polymarket_us_key_id,
            secret_key=self.config.polymarket_us_secret_key,
        )

    @property
    def platform(self) -> str:
        return "us"

    def _px_value(self, payload: dict[str, Any] | None) -> float | None:
        if not payload:
            return None
        return safe_float(payload.get("value"))

    def _book_levels(self, book: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        market_data = book.get("marketData") or book
        bids = market_data.get("bids") or []
        offers = market_data.get("offers") or []

        best_bid = self._px_value(bids[0].get("px")) if bids else None
        best_ask = self._px_value(offers[0].get("px")) if offers else None
        spread = None
        midpoint = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            midpoint = (best_bid + best_ask) / 2.0
        elif market_data.get("stats"):
            midpoint = self._px_value((market_data["stats"] or {}).get("currentPx"))
        return best_bid, best_ask, midpoint if spread is not None else midpoint

    def get_orderbook(self, market_slug: str) -> dict[str, Any]:
        return self._client.markets.book(market_slug)

    def get_bbo(self, market_slug: str) -> dict[str, Any]:
        return self._client.markets.bbo(market_slug)

    def search_markets(self, keyword: str, *, limit: int = 25) -> list[dict[str, Any]]:
        data = self._client.search.query({"q": keyword, "limit": limit})
        markets: list[dict[str, Any]] = []
        for event in data.get("events", []) or []:
            for market in event.get("markets", []) or []:
                enriched = dict(market)
                enriched["_event_title"] = event.get("title")
                enriched["_event_slug"] = event.get("slug")
                enriched["_platform"] = "us"
                markets.append(enriched)
        return markets

    def enrich_market(self, raw_market: dict[str, Any]) -> PolymarketMarket:
        event_title = raw_market.get("_event_title") or raw_market.get("question") or ""
        team_title = raw_market.get("title") or ""
        if team_title and event_title and team_title.lower() not in event_title.lower():
            question = f"Will {team_title} advance? ({event_title})"
        else:
            question = event_title or team_title or "Unknown market"

        slug = str(raw_market.get("slug") or raw_market.get("id") or question)
        market_id = str(raw_market.get("id") or slug)

        best_bid = best_ask = midpoint = spread = None
        liquidity = safe_float(raw_market.get("volume"), default=0.0)
        try:
            bbo = self.get_bbo(slug)
            market_data = (bbo or {}).get("marketData") or {}
            best_bid = self._px_value(market_data.get("bestBid"))
            best_ask = self._px_value(market_data.get("bestAsk"))
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid
                midpoint = (best_bid + best_ask) / 2.0
            else:
                midpoint = self._px_value((market_data.get("stats") or {}).get("currentPx"))
            open_interest = safe_float(market_data.get("openInterest"))
            if open_interest:
                liquidity = max(liquidity or 0.0, open_interest)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("BBO fetch failed for %s: %s", slug, exc)

        if midpoint is None:
            outcome_prices = raw_market.get("outcomePrices") or []
            if outcome_prices:
                midpoint = safe_float(outcome_prices[0])

        event_slug = raw_market.get("_event_slug")
        url = f"https://polymarket.us/event/{event_slug}" if event_slug else f"https://polymarket.us/markets/{slug}"

        close_date = None
        end_date = raw_market.get("endDate") or raw_market.get("gameStartTime")
        if end_date:
            try:
                close_date = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            except ValueError:
                close_date = None

        side_ids: list[str] = []
        for side in raw_market.get("marketSides") or []:
            identifier = side.get("identifier") or side.get("id")
            if identifier is not None:
                side_ids.append(str(identifier))

        return PolymarketMarket(
            market_id=market_id,
            question=question,
            event=event_title or None,
            outcome_names=["Yes", "No"],
            token_ids=side_ids or [slug],
            best_bid=best_bid,
            best_ask=best_ask,
            midpoint=midpoint,
            spread=spread,
            volume=safe_float(raw_market.get("volume"), default=0.0),
            liquidity=liquidity,
            close_date=close_date,
            url=url,
            active=bool(raw_market.get("active", True)) and not bool(raw_market.get("closed", False)),
            platform="us",
            raw=raw_market,
        )

    def is_football_market(self, market: PolymarketMarket) -> bool:
        haystack = " ".join(
            part.lower()
            for part in [market.question, market.event or "", json.dumps(market.raw)]
            if part
        )
        return any(keyword in haystack for keyword in self.FOOTBALL_KEYWORDS)

    def is_binary_market(self, market: PolymarketMarket) -> bool:
        return True

    def search_football_markets(
        self,
        keywords: list[str],
        *,
        min_liquidity: float = 0.0,
        active_only: bool = True,
        limit_per_keyword: int = 20,
    ) -> list[PolymarketMarket]:
        seen: set[str] = set()
        results: list[PolymarketMarket] = []

        for keyword in keywords:
            try:
                raw_markets = self.search_markets(keyword, limit=limit_per_keyword)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("US search failed for keyword '%s': %s", keyword, exc)
                continue

            for raw in raw_markets:
                if active_only and not raw.get("active", True):
                    continue
                if raw.get("closed", False):
                    continue
                enriched = self.enrich_market(raw)
                if enriched.market_id in seen:
                    continue
                if not self.is_football_market(enriched):
                    continue
                if not self.is_binary_market(enriched):
                    continue
                if (enriched.liquidity or 0.0) < min_liquidity:
                    continue
                seen.add(enriched.market_id)
                results.append(enriched)
        return results

    def refresh_bbo(self, markets: list[PolymarketMarket], *, limit: int = 3) -> None:
        """Refresh best bid/ask for the first N markets using the US BBO endpoint."""
        for market in markets[:limit]:
            slug = market.token_ids[0] if market.token_ids else market.market_id
            try:
                bbo = self.get_bbo(str(slug))
                market_data = (bbo or {}).get("marketData") or {}
                market.best_bid = self._px_value(market_data.get("bestBid"))
                market.best_ask = self._px_value(market_data.get("bestAsk"))
                if market.best_bid is not None and market.best_ask is not None:
                    market.spread = market.best_ask - market.best_bid
                    market.midpoint = (market.best_bid + market.best_ask) / 2.0
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("BBO refresh failed for %s: %s", slug, exc)

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()
