"""Conservative bankroll and exposure risk management."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import AppConfig, load_config


@dataclass
class RiskSettings:
    bankroll: float
    min_edge_threshold: float = 0.06
    max_spread: float = 0.06
    min_liquidity: float = 100.0
    min_model_confidence: float = 0.55
    kelly_multiplier: float = 0.08
    max_risk_per_trade: float = 0.005
    max_exposure_per_match: float = 0.015
    max_total_exposure: float = 0.06
    min_ticket_size: float = 1.0
    liquidity_participation_cap: float = 0.05


@dataclass
class RiskState:
    daily_loss_pct: float
    consecutive_losses: int
    watch_only: bool
    stop_new_recommendations: bool


def kelly_fraction(probability: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (probability - price) / (1 - price))


def liquidity_cap(market_liquidity: float | None, participation_cap: float) -> float:
    if market_liquidity is None or market_liquidity <= 0:
        return float("inf")
    return market_liquidity * participation_cap


def compute_paper_trade_size(
    *,
    bankroll: float,
    model_prob: float,
    execution_price: float,
    risk_settings: RiskSettings,
    calibration_poor: bool = False,
    current_match_exposure: float = 0.0,
    total_open_exposure: float = 0.0,
    market_liquidity: float | None = None,
) -> float:
    fraction = kelly_fraction(model_prob, execution_price)
    raw_size = bankroll * fraction * risk_settings.kelly_multiplier

    if calibration_poor:
        raw_size *= 0.5

    max_trade = bankroll * risk_settings.max_risk_per_trade
    remaining_match = bankroll * risk_settings.max_exposure_per_match - current_match_exposure
    remaining_total = bankroll * risk_settings.max_total_exposure - total_open_exposure
    liq_cap = liquidity_cap(market_liquidity, risk_settings.liquidity_participation_cap)

    size = max(0.0, min(raw_size, max_trade, remaining_match, remaining_total, liq_cap))

    if 0 < size < risk_settings.min_ticket_size:
        if bankroll >= risk_settings.min_ticket_size and max_trade >= risk_settings.min_ticket_size:
            size = risk_settings.min_ticket_size
        else:
            size = 0.0

    return round(size, 2)


def evaluate_risk_state(paper_trades: pd.DataFrame, bankroll: float) -> RiskState:
    if paper_trades.empty or bankroll <= 0:
        return RiskState(daily_loss_pct=0.0, consecutive_losses=0, watch_only=False, stop_new_recommendations=False)

    frame = paper_trades.copy()
    if "pnl" not in frame.columns:
        frame["pnl"] = 0.0
    if "status" in frame.columns:
        closed = frame[frame["status"].isin(["won", "lost", "closed"])]
    else:
        closed = frame

    daily_loss_pct = 0.0
    if "closed_at" in closed.columns and not closed.empty:
        closed = closed.copy()
        closed["closed_at"] = pd.to_datetime(closed["closed_at"], errors="coerce", utc=True)
        today = pd.Timestamp.utcnow().normalize()
        today_pnl = closed.loc[closed["closed_at"].dt.normalize() == today, "pnl"].sum()
        daily_loss_pct = float(-today_pnl / bankroll) if today_pnl < 0 else 0.0

    consecutive_losses = 0
    if not closed.empty and "pnl" in closed.columns:
        sort_col = "closed_at" if "closed_at" in closed.columns else closed.columns[0]
        for pnl in closed.sort_values(sort_col, ascending=False)["pnl"].tolist():
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

    watch_only = consecutive_losses >= 3
    stop_new = daily_loss_pct > 0.05
    return RiskState(
        daily_loss_pct=daily_loss_pct,
        consecutive_losses=consecutive_losses,
        watch_only=watch_only,
        stop_new_recommendations=stop_new,
    )


def default_risk_settings(config: AppConfig | None = None, bankroll: float | None = None) -> RiskSettings:
    config = config or load_config()
    return RiskSettings(
        bankroll=bankroll or config.default_bankroll,
        min_edge_threshold=config.default_min_edge,
        max_spread=config.default_max_spread,
        min_liquidity=config.default_min_liquidity,
        min_model_confidence=config.default_min_model_confidence,
        kelly_multiplier=config.default_kelly_multiplier,
        max_risk_per_trade=config.default_max_risk_per_trade,
        max_exposure_per_match=config.default_max_exposure_per_match,
        max_total_exposure=config.default_max_total_exposure,
        min_ticket_size=config.min_ticket_size,
        liquidity_participation_cap=config.liquidity_participation_cap,
    )


def small_bankroll_guidance(bankroll: float) -> list[str]:
    max_trade = bankroll * 0.005
    return [
        f"With ${bankroll:.2f}, a single trade is typically capped around ${max_trade:.2f} (0.5% of bankroll).",
        "Most markets should still be NO_TRADE. Patience protects a small bankroll.",
        "Paper-trade first until you have at least 10–20 logged decisions.",
        "Never use market orders; limit orders only, and skip if the ask moves.",
    ]
