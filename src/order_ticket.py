"""Order ticket generator for manual human execution only."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.compliance import ComplianceStatus
from src.market_parser import ParsedMarket
from src.polymarket_public_client import PolymarketMarket
from src.strategy import Recommendation, StrategyResult


@dataclass
class OrderTicket:
    market_title: str
    polymarket_url: str | None
    selected_outcome: str
    side: str
    token_id: str | None
    current_bid: float | None
    current_ask: float | None
    limit_price: float | None
    max_acceptable_price: float | None
    recommended_size: float
    max_loss: float
    model_probability: float
    market_implied_probability: float | None
    executable_implied_probability: float | None
    raw_edge: float | None
    adjusted_edge: float | None
    spread: float | None
    liquidity: float | None
    model_confidence: str
    compliance_status: str
    recommendation: str
    reasons_for: list[str] = field(default_factory=list)
    reasons_against: list[str] = field(default_factory=list)
    invalidation_triggers: list[str] = field(default_factory=list)
    human_checklist: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)


def build_order_ticket(
    *,
    market: PolymarketMarket,
    parsed: ParsedMarket,
    strategy: StrategyResult,
    compliance: ComplianceStatus,
    model_confidence: float,
    bankroll: float,
) -> OrderTicket:
    side = "BUY YES"
    token_id = market.token_ids[0] if market.token_ids else None
    limit_price = strategy.buy_yes_execution_price
    raw_edge = strategy.raw_edge_yes
    adjusted_edge = strategy.adjusted_edge_yes
    executable_implied = limit_price
    model_prob = strategy.model_prob

    if strategy.recommendation == Recommendation.VALUE_LONG_NO:
        side = "BUY NO"
        token_id = market.token_ids[1] if len(market.token_ids) > 1 else token_id
        limit_price = strategy.buy_no_execution_price
        raw_edge = strategy.raw_edge_no
        adjusted_edge = strategy.adjusted_edge_no
        executable_implied = limit_price
        model_prob = 1.0 - strategy.model_prob

    max_slippage = 0.02
    max_acceptable = (limit_price + max_slippage) if limit_price is not None else None
    max_loss = strategy.suggested_paper_size

    reasons_for = list(strategy.reasons_for)
    reasons_against = list(strategy.reasons_against)
    invalidation = list(strategy.invalidation_triggers)

    if strategy.recommendation in {Recommendation.VALUE_LONG_YES, Recommendation.VALUE_LONG_NO}:
        reasons_for.append(f"Adjusted edge {adjusted_edge:.3f} meets threshold after buffers.")
        reasons_for.append(f"Suggested size ${strategy.suggested_paper_size:.2f} is capped for your ${bankroll:.2f} bankroll.")
    else:
        reasons_against.append(strategy.reason)

    human_checklist = [
        "I confirmed this market maps to the correct match and outcome.",
        (
            "I verified Polymarket US legality and platform rules for my state."
            if market.platform == "us"
            else "I verified Polymarket legality and platform rules for my jurisdiction."
        ),
        "I will use a LIMIT order only (no market order).",
        "I accept that this is not a guarantee and I can lose the full stake.",
        "I will not chase price if the ask moves beyond max acceptable price.",
    ]

    disclaimers = [
        "This is not a guarantee.",
        "This ticket is not submitted automatically.",
        "User must manually confirm legality and execution.",
        "Past model edge does not guarantee future profit.",
    ]

    return OrderTicket(
        market_title=market.question,
        polymarket_url=market.url,
        selected_outcome=parsed.target_outcome.upper(),
        side=side,
        token_id=token_id,
        current_bid=market.best_bid,
        current_ask=market.best_ask,
        limit_price=limit_price,
        max_acceptable_price=max_acceptable,
        recommended_size=strategy.suggested_paper_size,
        max_loss=max_loss,
        model_probability=model_prob,
        market_implied_probability=strategy.market_implied_prob,
        executable_implied_probability=executable_implied,
        raw_edge=raw_edge,
        adjusted_edge=adjusted_edge,
        spread=strategy.spread,
        liquidity=market.liquidity,
        model_confidence=strategy.confidence_level,
        compliance_status=compliance.mode.value,
        recommendation=strategy.recommendation.value,
        reasons_for=reasons_for,
        reasons_against=reasons_against,
        invalidation_triggers=invalidation,
        human_checklist=human_checklist,
        disclaimers=disclaimers,
    )


def ticket_to_markdown(ticket: OrderTicket) -> str:
    lines = [
        "# Order Ticket (Manual Execution Only)",
        "",
        f"**Market:** {ticket.market_title}",
        f"**URL:** {ticket.polymarket_url or 'N/A'}",
        f"**Recommendation:** {ticket.recommendation}",
        f"**Side:** {ticket.side}",
        f"**Token ID:** {ticket.token_id or 'N/A'}",
        "",
        "## Prices",
        f"- Bid: {ticket.current_bid}",
        f"- Ask: {ticket.current_ask}",
        f"- Limit price: {ticket.limit_price}",
        f"- Max acceptable price: {ticket.max_acceptable_price}",
        f"- Spread: {ticket.spread}",
        f"- Liquidity: {ticket.liquidity}",
        "",
        "## Model vs Market",
        f"- Model probability: {ticket.model_probability:.1%}",
        f"- Midpoint (info only): {ticket.market_implied_probability}",
        f"- Executable implied: {ticket.executable_implied_probability}",
        f"- Raw edge: {ticket.raw_edge}",
        f"- Adjusted edge: {ticket.adjusted_edge}",
        f"- Confidence: {ticket.model_confidence}",
        "",
        "## Size",
        f"- Recommended size: ${ticket.recommended_size:.2f}",
        f"- Max loss if wrong: ${ticket.max_loss:.2f}",
        "",
        "## Reasons FOR",
    ]
    lines.extend(f"- {item}" for item in ticket.reasons_for)
    lines.append("")
    lines.append("## Reasons AGAINST")
    lines.extend(f"- {item}" for item in ticket.reasons_against)
    lines.append("")
    lines.append("## Would invalidate this idea")
    lines.extend(f"- {item}" for item in ticket.invalidation_triggers)
    lines.append("")
    lines.append("## Human checklist")
    lines.extend(f"- [ ] {item}" for item in ticket.human_checklist)
    lines.append("")
    lines.append("## Disclaimers")
    lines.extend(f"- {item}" for item in ticket.disclaimers)
    return "\n".join(lines)
