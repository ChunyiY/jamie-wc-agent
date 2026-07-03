"""Read-only Polymarket public API client."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import requests
from pydantic import BaseModel, Field

from src.config import AppConfig, load_config
from src.utils import parse_json_list, safe_float, setup_logger


class PolymarketMarket(BaseModel):
    market_id: str
    question: str
    event: str | None = None
    outcome_names: list[str] = Field(default_factory=list)
    token_ids: list[str] = Field(default_factory=list)
    best_bid: float | None = None
    best_ask: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    volume: float | None = None
    liquidity: float | None = None
    close_date: datetime | None = None
    url: str | None = None
    active: bool = True
    platform: str = "international"
    raw: dict[str, Any] = Field(default_factory=dict)


class PolymarketPublicClient:
    platform = "international"

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
    )

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.logger = setup_logger("polymarket_public_client")

    def _get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        response = requests.get(url, params=params, timeout=self.config.request_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def search_markets(self, keyword: str, *, limit: int = 25) -> list[dict[str, Any]]:
        data = self._get(
            self.config.polymarket_gamma_url,
            "/public-search",
            params={"q": keyword, "limit_per_type": limit, "keep_closed_markets": 0},
        )
        markets: list[dict[str, Any]] = []
        for event in data.get("events", []) or []:
            for market in event.get("markets", []) or []:
                market = dict(market)
                market["_event_title"] = event.get("title")
                market["_event_slug"] = event.get("slug")
                markets.append(market)
        for market in data.get("markets", []) or []:
            markets.append(market)
        return markets

    def list_active_markets(self, *, limit: int = 50, keyword: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "active": True,
            "closed": False,
            "limit": limit,
        }
        if keyword:
            return self.search_markets(keyword, limit=limit)
        return self._get(self.config.polymarket_gamma_url, "/markets", params=params)

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        return self._get(self.config.polymarket_clob_url, "/book", params={"token_id": token_id})

    def get_midpoint(self, token_id: str) -> float | None:
        try:
            data = self._get(self.config.polymarket_clob_url, "/midpoint", params={"token_id": token_id})
            return safe_float(data.get("mid"))
        except requests.RequestException:
            return None

    def get_spread(self, token_id: str) -> float | None:
        try:
            data = self._get(self.config.polymarket_clob_url, "/spread", params={"token_id": token_id})
            return safe_float(data.get("spread"))
        except requests.RequestException:
            return None

    def enrich_market(self, raw_market: dict[str, Any]) -> PolymarketMarket:
        question = raw_market.get("question") or raw_market.get("title") or "Unknown market"
        market_id = str(raw_market.get("id") or raw_market.get("condition_id") or raw_market.get("slug") or question)
        event = raw_market.get("_event_title") or raw_market.get("event_title")

        outcome_names = parse_json_list(raw_market.get("outcomes"))
        token_ids = parse_json_list(raw_market.get("clobTokenIds") or raw_market.get("clob_token_ids"))

        best_bid = best_ask = midpoint = spread = None
        yes_token = token_ids[0] if token_ids else None
        if yes_token:
            try:
                book = self.get_orderbook(str(yes_token))
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                if bids:
                    best_bid = safe_float(bids[0].get("price"))
                if asks:
                    best_ask = safe_float(asks[0].get("price"))
                if best_bid is not None and best_ask is not None:
                    midpoint = (best_bid + best_ask) / 2.0
                    spread = best_ask - best_bid
                else:
                    midpoint = self.get_midpoint(str(yes_token))
                    spread = self.get_spread(str(yes_token))
            except requests.RequestException as exc:
                self.logger.warning("Orderbook fetch failed for %s: %s", market_id, exc)

        if midpoint is None:
            outcome_prices = parse_json_list(raw_market.get("outcomePrices"))
            if outcome_prices:
                midpoint = safe_float(outcome_prices[0])

        slug = raw_market.get("slug") or raw_market.get("_event_slug")
        url = f"https://polymarket.com/event/{slug}" if slug else None

        close_date = None
        end_date = raw_market.get("endDate") or raw_market.get("end_date_iso")
        if end_date:
            try:
                close_date = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            except ValueError:
                close_date = None

        return PolymarketMarket(
            market_id=market_id,
            question=question,
            event=event,
            outcome_names=[str(name) for name in outcome_names] or ["Yes", "No"],
            token_ids=[str(token) for token in token_ids],
            best_bid=best_bid,
            best_ask=best_ask,
            midpoint=midpoint,
            spread=spread,
            volume=safe_float(raw_market.get("volumeNum") or raw_market.get("volume"), default=0.0),
            liquidity=safe_float(raw_market.get("liquidityNum") or raw_market.get("liquidity"), default=0.0),
            close_date=close_date,
            url=url,
            active=bool(raw_market.get("active", True)),
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
        return len(market.outcome_names) == 2 or len(market.token_ids) >= 1

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
            except requests.RequestException as exc:
                self.logger.error("Search failed for keyword '%s': %s", keyword, exc)
                continue

            for raw in raw_markets:
                if active_only and not raw.get("active", True):
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
