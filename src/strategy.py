"""Edge calculation and conservative recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.compliance import ComplianceStatus
from src.config import AppConfig, load_config
from src.market_parser import MarketType, ParsedMarket
from src.polymarket_public_client import PolymarketMarket
from src.risk import RiskSettings, compute_paper_trade_size


class Recommendation(str, Enum):
    BLOCKED_BY_COMPLIANCE = "BLOCKED_BY_COMPLIANCE"
    READ_ONLY_ONLY = "READ_ONLY_ONLY"
    NEEDS_MANUAL_MAPPING = "NEEDS_MANUAL_MAPPING"
    NO_TRADE_WIDE_SPREAD = "NO_TRADE_WIDE_SPREAD"
    NO_TRADE_LOW_LIQUIDITY = "NO_TRADE_LOW_LIQUIDITY"
    WATCH_LOW_CONFIDENCE = "WATCH_LOW_CONFIDENCE"
    WATCH_ONLY = "WATCH_ONLY"
    WATCH = "WATCH"
    NO_TRADE = "NO_TRADE"
    VALUE_CANDIDATE = "VALUE_CANDIDATE"
    VALUE_LONG_YES = "VALUE_LONG_YES"
    VALUE_LONG_NO = "VALUE_LONG_NO"


@dataclass
class StrategyInput:
    model_prob: float
    model_confidence: float
    market: PolymarketMarket
    parsed_market: ParsedMarket
    compliance: ComplianceStatus
    bankroll: float
    risk_settings: RiskSettings
    calibration_poor: bool = False
    minutes_to_close: float | None = None
    kickoff_started: bool = False
    price_moved_bps: float | None = None


@dataclass
class StrategyResult:
    recommendation: Recommendation
    model_prob: float
    market_implied_prob: float | None
    buy_yes_execution_price: float | None
    buy_no_execution_price: float | None
    spread: float | None
    raw_edge_yes: float | None
    raw_edge_no: float | None
    adjusted_edge_yes: float | None
    adjusted_edge_no: float | None
    suggested_paper_size: float
    confidence_level: str
    edge_strength: str
    selected_side: str | None
    risk_warning: str
    reason: str
    reasons_for: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    invalidation_triggers: list[str] = field(default_factory=list)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "moderate"
    if confidence >= 0.55:
        return "low-moderate"
    return "low"


def _edge_strength(adjusted_edge: float | None, threshold: float) -> str:
    if adjusted_edge is None:
        return "none"
    if adjusted_edge >= threshold + 0.04:
        return "moderate"
    if adjusted_edge >= threshold:
        return "weak"
    return "none"


def evaluate_strategy(data: StrategyInput, config: AppConfig | None = None) -> StrategyResult:
    config = config or load_config()
    market = data.market
    midpoint = market.midpoint
    best_bid = market.best_bid
    best_ask = market.best_ask
    spread = market.spread

    buy_yes = best_ask if best_ask is not None else None
    buy_no = (1.0 - best_bid) if best_bid is not None else None

    raw_edge_yes = raw_edge_no = adjusted_edge_yes = adjusted_edge_no = None
    if buy_yes is not None:
        raw_edge_yes = data.model_prob - buy_yes
        adjusted_edge_yes = raw_edge_yes - config.slippage_buffer - config.model_uncertainty_buffer
    if buy_no is not None:
        raw_edge_no = (1.0 - data.model_prob) - buy_no
        adjusted_edge_no = raw_edge_no - config.slippage_buffer - config.model_uncertainty_buffer

    base = StrategyResult(
        recommendation=Recommendation.NO_TRADE,
        model_prob=data.model_prob,
        market_implied_prob=midpoint,
        buy_yes_execution_price=buy_yes,
        buy_no_execution_price=buy_no,
        spread=spread,
        raw_edge_yes=raw_edge_yes,
        raw_edge_no=raw_edge_no,
        adjusted_edge_yes=adjusted_edge_yes,
        adjusted_edge_no=adjusted_edge_no,
        suggested_paper_size=0.0,
        confidence_level=_confidence_label(data.model_confidence),
        edge_strength="none",
        selected_side=None,
        risk_warning="Markets are uncertain. Most outcomes should be NO_TRADE.",
        reason="No actionable edge after risk filters.",
        reasons_against=["Adjusted edge below threshold after execution costs."],
        invalidation_triggers=[
            "Spread widens beyond your max spread setting.",
            "Liquidity drops before you place a manual limit order.",
            "Model confidence falls on refreshed data.",
        ],
    )

    if data.compliance.is_restricted or not data.compliance.can_paper_trade:
        base.recommendation = Recommendation.BLOCKED_BY_COMPLIANCE
        base.reason = "Compliance restrictions active. Read-only analytics only."
        base.risk_warning = "Do not attempt to bypass geographic or platform restrictions."
        base.reasons_against.append(base.reason)
        return base

    if data.parsed_market.needs_manual_mapping or data.parsed_market.confidence < config.parser_confidence_threshold:
        base.recommendation = Recommendation.NEEDS_MANUAL_MAPPING
        base.reason = "Market mapping uncertain. Confirm teams and market type manually."
        base.reasons_against.append(base.reason)
        return base

    if data.kickoff_started:
        base.recommendation = Recommendation.WATCH_ONLY
        base.reason = "Match may have started. Default to watch-only unless you explicitly accept live-market risk."
        base.reasons_against.append(base.reason)
        return base

    if buy_yes is None and buy_no is None:
        base.recommendation = Recommendation.NO_TRADE
        base.reason = "No executable bid/ask available. Midpoint alone is not tradable."
        base.reasons_against.append(base.reason)
        return base

    if spread is not None and spread > data.risk_settings.max_spread:
        base.recommendation = Recommendation.NO_TRADE_WIDE_SPREAD
        base.reason = f"Spread {spread:.3f} exceeds max {data.risk_settings.max_spread:.3f}."
        base.reasons_against.append(base.reason)
        return base

    if (market.liquidity or 0.0) < data.risk_settings.min_liquidity:
        base.recommendation = Recommendation.NO_TRADE_LOW_LIQUIDITY
        base.reason = f"Liquidity {(market.liquidity or 0):.0f} below minimum {data.risk_settings.min_liquidity:.0f}."
        base.reasons_against.append(base.reason)
        return base

    if data.calibration_poor:
        base.reasons_against.append("Model calibration is weak on recent holdout data. Size is reduced by 50%.")

    if data.model_confidence < data.risk_settings.min_model_confidence:
        base.recommendation = Recommendation.WATCH_LOW_CONFIDENCE
        base.reason = "Model confidence too low for a value candidate. Watch only."
        base.reasons_against.append(base.reason)
        return base

    if data.price_moved_bps is not None and data.price_moved_bps > config.max_price_move_bps:
        base.recommendation = Recommendation.NO_TRADE
        base.reason = f"Price moved {data.price_moved_bps:.0f} bps; avoid chasing."
        base.reasons_against.append(base.reason)
        return base

    if (
        data.minutes_to_close is not None
        and data.minutes_to_close <= 30
        and spread is not None
        and spread > data.risk_settings.max_spread * 0.75
    ):
        base.recommendation = Recommendation.NO_TRADE
        base.reason = "Market closes soon with a wide spread. Avoid new positions."
        base.reasons_against.append(base.reason)
        return base

    yes_edge = adjusted_edge_yes or -1.0
    no_edge = adjusted_edge_no or -1.0
    threshold = data.risk_settings.min_edge_threshold

    chosen_side = None
    chosen_edge = None
    chosen_price = None

    if yes_edge >= threshold and yes_edge >= no_edge:
        chosen_side = "yes"
        chosen_edge = yes_edge
        chosen_price = buy_yes
        base.recommendation = Recommendation.VALUE_LONG_YES
    elif no_edge >= threshold:
        chosen_side = "no"
        chosen_edge = no_edge
        chosen_price = buy_no
        base.recommendation = Recommendation.VALUE_LONG_NO
    else:
        base.recommendation = Recommendation.NO_TRADE
        base.reason = "Adjusted edge below threshold. Default to no trade."
        return base

    base.selected_side = chosen_side
    base.edge_strength = _edge_strength(chosen_edge, threshold)
    if chosen_side == "yes":
        base.recommendation = Recommendation.VALUE_LONG_YES
    else:
        base.recommendation = Recommendation.VALUE_LONG_NO

    size = compute_paper_trade_size(
        bankroll=data.bankroll,
        model_prob=data.model_prob if chosen_side == "yes" else 1.0 - data.model_prob,
        execution_price=chosen_price or 0.5,
        risk_settings=data.risk_settings,
        calibration_poor=data.calibration_poor,
        current_match_exposure=0.0,
        total_open_exposure=0.0,
        market_liquidity=market.liquidity,
    )

    if size <= 0:
        base.recommendation = Recommendation.NO_TRADE
        base.reason = "Risk caps or minimum ticket size block this trade."
        base.reasons_against.append(base.reason)
        base.suggested_paper_size = 0.0
        return base

    base.suggested_paper_size = size
    base.reason = "Model shows possible value versus executable price. Manual confirmation required."
    base.reasons_for = [
        f"Model probability {data.model_prob:.1%} vs executable price {chosen_price:.1%}.",
        f"Adjusted edge {chosen_edge:.3f} after slippage and uncertainty buffers.",
        f"Edge strength: {base.edge_strength} (not a guarantee).",
    ]
    base.reasons_against.append("Prediction markets can stay mispriced longer than expected.")
    base.reasons_against.append("Small bankrolls are vulnerable to a few losses in a row.")
    base.invalidation_triggers.extend(
        [
            f"Ask moves more than {config.max_price_move_bps:.0f} bps before you submit a limit order.",
            "Team news or lineup changes materially shift match outlook.",
            "Spread widens or liquidity thins at execution time.",
        ]
    )
    base.risk_warning = (
        f"Suggested size ${size:,.2f} on a ${data.bankroll:,.2f} bankroll. "
        f"This is research support only, not financial advice."
    )
    return base


def minutes_until_close(market: PolymarketMarket) -> float | None:
    if market.close_date is None:
        return None
    now = datetime.now(timezone.utc)
    close = market.close_date
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    return (close - now).total_seconds() / 60.0
