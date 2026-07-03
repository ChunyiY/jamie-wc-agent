"""Historical model and recommendation backtesting with time ordering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from src.config import load_config
from src.elo import expected_score
from src.features import FEATURE_COLUMNS, build_training_frame_point_in_time
from src.model import FootballOutcomeModel
from src.risk import RiskSettings, compute_paper_trade_size


@dataclass
class BacktestSummary:
    matches: int
    log_loss: float
    brier_score: float
    accuracy: float
    calibration_poor: bool
    flat_bet_roi: float
    value_bet_roi: float
    max_drawdown: float
    recommendation_count: int = 0
    recommendation_hit_rate: float = 0.0
    avg_edge: float = 0.0


def _elo_baseline_probabilities(frame: pd.DataFrame) -> np.ndarray:
    probs = []
    for _, row in frame.iterrows():
        diff = float(row["elo_diff"])
        home_adv = float(row.get("home_advantage_flag", 0.0))
        rating_a = 1500 + diff / 2 + (50 if home_adv else 0)
        rating_b = 1500 - diff / 2
        win_a = expected_score(rating_a, rating_b)
        win_b = expected_score(rating_b, rating_a)
        draw = max(0.05, 1.0 - abs(win_a - win_b))
        total = win_a + win_b + draw
        probs.append([win_a / total, draw / total, win_b / total])
    return np.array(probs)


def _compute_multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    labels = [0, 1, 2]
    ll = log_loss(y_true, probabilities, labels=labels)
    brier = float(np.mean([brier_score_loss((y_true == label).astype(int), probabilities[:, idx]) for idx, label in enumerate(labels)]))
    acc = accuracy_score(y_true, probabilities.argmax(axis=1))
    return float(ll), float(brier), float(acc)


def _simulate_value_bets(
    test_frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    risk_settings: RiskSettings,
    synthetic_market_noise: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """Proxy value-bet simulation using synthetic noisy market asks (not historical Polymarket)."""
    rng = np.random.default_rng(seed)
    bankroll = risk_settings.bankroll
    rows = []

    for idx, (_, row) in enumerate(test_frame.iterrows()):
        probs = probabilities[idx]
        outcome = int(row["outcome"])
        pick = int(np.argmax(probs))
        confidence = float(np.max(probs))
        model_yes = float(probs[0])
        market_ask = min(0.95, max(0.05, model_yes + rng.normal(0, synthetic_market_noise)))
        edge = model_yes - market_ask
        take = edge >= risk_settings.min_edge_threshold and confidence >= risk_settings.min_model_confidence

        stake = 0.0
        pnl = 0.0
        if take:
            stake = compute_paper_trade_size(
                bankroll=bankroll,
                model_prob=model_yes,
                execution_price=market_ask,
                risk_settings=risk_settings,
                market_liquidity=1000.0,
            )
            if stake > 0:
                won = outcome == 0
                pnl = stake * (1.0 / market_ask - 1.0) if won else -stake
        bankroll += pnl
        rows.append(
            {
                "index": idx,
                "date": row["date"],
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "confidence": confidence,
                "market_ask": market_ask,
                "model_prob": model_yes,
                "edge": edge,
                "signal": take and stake > 0,
                "stake": stake,
                "pnl": pnl,
                "bankroll": bankroll,
                "actual": outcome,
                "pick": pick,
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward_backtest(
    results: pd.DataFrame,
    *,
    model_name: str = "logistic_regression",
    test_fraction: float = 0.2,
    max_training_matches: int | None = 12_000,
    initial_bankroll: float | None = None,
    n_splits: int = 5,
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame, FootballOutcomeModel | None]:
    _ = n_splits
    config = load_config()
    bankroll = initial_bankroll or config.default_bankroll
    modeling_results = results.tail(max_training_matches).copy() if max_training_matches else results.copy()
    training = build_training_frame_point_in_time(modeling_results)
    training = training.dropna(subset=FEATURE_COLUMNS + ["outcome"]).sort_values("date").reset_index(drop=True)

    if len(training) < 200:
        raise ValueError("Need at least 200 matches for walk-forward backtest.")

    split_index = int(len(training) * (1 - test_fraction))
    train = training.iloc[:split_index]
    test = training.iloc[split_index:].copy()

    if model_name == "elo_baseline":
        prob_array = _elo_baseline_probabilities(test)
        model = None
        ll, brier, acc = _compute_multiclass_metrics(test["outcome"].values, prob_array)
        calibration_poor = brier > 0.22 or ll > 1.05
    else:
        model = FootballOutcomeModel()
        metrics = model.train(train, model_name=model_name, test_fraction=0.15, calibrate=True)
        prob_list = []
        for _, row in test.iterrows():
            features = {col: float(row[col]) for col in FEATURE_COLUMNS}
            prob_list.append(model.predict_proba(features))
        prob_array = np.array(prob_list)
        ll, brier, acc = metrics.log_loss, metrics.brier_score, metrics.accuracy
        calibration_poor = metrics.brier_score > 0.22 or metrics.log_loss > 1.05

    outcomes = test["outcome"].values
    predictions = prob_array.argmax(axis=1)
    risk_settings = RiskSettings(bankroll=bankroll)
    value_curve = _simulate_value_bets(test, prob_array, risk_settings=risk_settings)
    signals = value_curve[value_curve["signal"]]
    hit_rate = float((signals["pick"] == signals["actual"]).mean()) if not signals.empty else 0.0

    total_pnl = value_curve["pnl"].sum()
    total_staked = value_curve["stake"].sum()
    running_max = value_curve["bankroll"].cummax()
    drawdown = (value_curve["bankroll"] - running_max) / running_max.replace(0, np.nan)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    summary = BacktestSummary(
        matches=len(test),
        log_loss=ll,
        brier_score=brier,
        accuracy=float(acc),
        calibration_poor=calibration_poor,
        flat_bet_roi=float(total_pnl / bankroll) if bankroll else 0.0,
        value_bet_roi=float(total_pnl / total_staked) if total_staked > 0 else 0.0,
        max_drawdown=max_drawdown,
        recommendation_count=int(signals.shape[0]),
        recommendation_hit_rate=hit_rate,
        avg_edge=float(signals["edge"].mean()) if not signals.empty else 0.0,
    )

    prediction_frame = test[["date", "team_a", "team_b", "outcome"]].copy()
    prediction_frame["pred"] = predictions
    prediction_frame["confidence"] = prob_array.max(axis=1)
    return summary, prediction_frame, value_curve, model


def run_model_backtest(
    results: pd.DataFrame,
    *,
    model_name: str = "logistic_regression",
    test_fraction: float = 0.2,
    initial_bankroll: float | None = None,
    max_training_matches: int | None = 12_000,
) -> tuple[BacktestSummary, pd.DataFrame, pd.DataFrame, FootballOutcomeModel | None]:
    return run_walk_forward_backtest(
        results,
        model_name=model_name,
        test_fraction=test_fraction,
        max_training_matches=max_training_matches,
        initial_bankroll=initial_bankroll,
    )


def compare_model_backtests(
    results: pd.DataFrame,
    test_fraction: float = 0.2,
    max_training_matches: int | None = 12_000,
    initial_bankroll: float | None = None,
) -> pd.DataFrame:
    rows = []
    for model_name in ("elo_baseline", "logistic_regression", "random_forest", "hist_gradient_boosting"):
        try:
            summary, _, _, _ = run_walk_forward_backtest(
                results,
                model_name=model_name,
                test_fraction=test_fraction,
                max_training_matches=max_training_matches,
                initial_bankroll=initial_bankroll,
            )
            rows.append(
                {
                    "model": model_name,
                    "log_loss": summary.log_loss,
                    "brier_score": summary.brier_score,
                    "accuracy": summary.accuracy,
                    "proxy_value_signals": summary.recommendation_count,
                    "signal_hit_rate": summary.recommendation_hit_rate,
                    "proxy_roi": summary.flat_bet_roi,
                    "max_drawdown": summary.max_drawdown,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model_name, "error": str(exc)})
    return pd.DataFrame(rows)


def bankroll_curve_chart(bankroll_curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bankroll_curve["index"], y=bankroll_curve["bankroll"], mode="lines", name="Bankroll"))
    fig.update_layout(
        title="Simulated Bankroll Curve (proxy prices — research only)",
        xaxis_title="Trade index",
        yaxis_title="Bankroll ($)",
    )
    return fig
