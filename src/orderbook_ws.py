"""Optional live Polymarket order book updates via public WebSocket."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.utils import safe_float

try:
    import websocket
except ImportError:  # pragma: no cover - optional dependency
    websocket = None

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class OrderbookSnapshot:
    asset_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    spread: float | None = None
    bids: list[dict[str, Any]] = field(default_factory=list)
    asks: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = 0.0


def _best_from_book_sides(bids: list[dict[str, Any]], asks: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    best_bid = safe_float(bids[0]["price"]) if bids else None
    best_ask = safe_float(asks[0]["price"]) if asks else None
    return best_bid, best_ask


def fetch_live_orderbook_snapshot(
    token_ids: list[str],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, OrderbookSnapshot]:
    """
    Connect to Polymarket public market channel and return a short-lived snapshot.

    Read-only. No trading actions are performed.
    """
    if websocket is None:
        raise RuntimeError("websocket-client is not installed. Run: pip install websocket-client")
    if not token_ids:
        return {}

    snapshots: dict[str, OrderbookSnapshot] = {
        token_id: OrderbookSnapshot(asset_id=token_id) for token_id in token_ids
    }
    done = threading.Event()

    def on_message(_ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        event_type = payload.get("event_type")
        if event_type == "book":
            asset_id = str(payload.get("asset_id", ""))
            if asset_id not in snapshots:
                return
            bids = payload.get("bids") or []
            asks = payload.get("asks") or []
            best_bid, best_ask = _best_from_book_sides(bids, asks)
            spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
            snapshots[asset_id] = OrderbookSnapshot(
                asset_id=asset_id,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                bids=bids,
                asks=asks,
                updated_at=time.time(),
            )
            if all(snapshot.updated_at > 0 for snapshot in snapshots.values()):
                done.set()
        elif event_type == "best_bid_ask":
            asset_id = str(payload.get("asset_id", ""))
            if asset_id not in snapshots:
                return
            best_bid = safe_float(payload.get("best_bid"), default=float("nan"))
            best_ask = safe_float(payload.get("best_ask"), default=float("nan"))
            spread = safe_float(payload.get("spread"), default=float("nan"))
            snapshots[asset_id] = OrderbookSnapshot(
                asset_id=asset_id,
                best_bid=None if best_bid != best_bid else best_bid,
                best_ask=None if best_ask != best_ask else best_ask,
                spread=None if spread != spread else spread,
                updated_at=time.time(),
            )
            if all(snapshot.updated_at > 0 for snapshot in snapshots.values()):
                done.set()

    def on_open(ws: Any) -> None:
        ws.send(
            json.dumps(
                {
                    "assets_ids": token_ids,
                    "type": "market",
                    "custom_feature_enabled": True,
                }
            )
        )

    ws_app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
    )
    thread = threading.Thread(target=ws_app.run_forever, kwargs={"ping_interval": 20, "ping_timeout": 10}, daemon=True)
    thread.start()
    done.wait(timeout=timeout_seconds)
    ws_app.close()
    thread.join(timeout=1.0)
    return snapshots
