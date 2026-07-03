"""Paper trading ledger for research and manual decision support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import ensure_csv_columns, utc_now_iso

PAPER_TRADE_COLUMNS = [
    "trade_id",
    "created_at",
    "market_id",
    "market_title",
    "team_a",
    "team_b",
    "side",
    "size",
    "entry_price",
    "model_prob",
    "recommendation",
    "status",
    "exit_price",
    "pnl",
    "closed_at",
    "notes",
]


class PaperTradingLedger:
    def __init__(self, path: Path):
        self.path = path
        self.frame = ensure_csv_columns(path, PAPER_TRADE_COLUMNS)

    def save(self) -> None:
        self.frame.to_csv(self.path, index=False)

    def reload(self) -> None:
        self.frame = ensure_csv_columns(self.path, PAPER_TRADE_COLUMNS)

    def record_trade(
        self,
        *,
        market_id: str,
        market_title: str,
        team_a: str | None,
        team_b: str | None,
        side: str,
        size: float,
        entry_price: float,
        model_prob: float,
        recommendation: str,
        notes: str = "",
    ) -> dict:
        trade_id = f"PT-{len(self.frame) + 1:05d}"
        row = {
            "trade_id": trade_id,
            "created_at": utc_now_iso(),
            "market_id": market_id,
            "market_title": market_title,
            "team_a": team_a or "",
            "team_b": team_b or "",
            "side": side.upper(),
            "size": size,
            "entry_price": entry_price,
            "model_prob": model_prob,
            "recommendation": recommendation,
            "status": "open",
            "exit_price": "",
            "pnl": 0.0,
            "closed_at": "",
            "notes": notes,
        }
        self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)
        self.save()
        return row

    def close_trade(self, trade_id: str, exit_price: float, won: bool | None = None) -> None:
        mask = self.frame["trade_id"] == trade_id
        if not mask.any():
            raise ValueError(f"Trade {trade_id} not found")

        idx = self.frame.index[mask][0]
        side = str(self.frame.at[idx, "side"]).upper()
        size = float(self.frame.at[idx, "size"])
        entry = float(self.frame.at[idx, "entry_price"])

        if side == "YES":
            pnl = size * (exit_price - entry)
            status = "won" if (won is True or exit_price > entry) else "lost"
        else:
            pnl = size * (entry - exit_price)
            status = "won" if (won is True or exit_price < entry) else "lost"

        self.frame.at[idx, "exit_price"] = exit_price
        self.frame.at[idx, "pnl"] = pnl
        self.frame.at[idx, "status"] = status
        self.frame.at[idx, "closed_at"] = utc_now_iso()
        self.save()

    def open_exposure(self) -> float:
        open_trades = self.frame[self.frame["status"] == "open"]
        return float(open_trades["size"].sum()) if not open_trades.empty else 0.0

    def total_pnl(self) -> float:
        return float(self.frame["pnl"].fillna(0).sum())

    def performance_summary(self) -> dict[str, float | int]:
        closed = self.frame[self.frame["status"].isin(["won", "lost", "closed"])]
        wins = int((closed["pnl"] > 0).sum()) if not closed.empty else 0
        losses = int((closed["pnl"] < 0).sum()) if not closed.empty else 0
        return {
            "open_trades": int((self.frame["status"] == "open").sum()),
            "closed_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "total_pnl": self.total_pnl(),
            "open_exposure": self.open_exposure(),
        }
