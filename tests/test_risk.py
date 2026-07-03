"""Tests for risk management."""

from src.risk import RiskSettings, compute_paper_trade_size, evaluate_risk_state, kelly_fraction
import pandas as pd


def test_kelly_fraction_non_negative():
    assert kelly_fraction(0.6, 0.5) > 0
    assert kelly_fraction(0.4, 0.5) == 0


def test_paper_trade_size_capped():
    settings = RiskSettings(bankroll=70, kelly_multiplier=0.08, max_risk_per_trade=0.005)
    size = compute_paper_trade_size(
        bankroll=70,
        model_prob=0.7,
        execution_price=0.5,
        risk_settings=settings,
        market_liquidity=1000.0,
    )
    assert size <= 0.35


def test_risk_state_consecutive_losses_watch_only():
    trades = pd.DataFrame(
        {
            "status": ["lost", "lost", "lost"],
            "pnl": [-10, -10, -10],
            "closed_at": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
        }
    )
    state = evaluate_risk_state(trades, bankroll=10_000)
    assert state.watch_only is True
