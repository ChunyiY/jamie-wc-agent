"""Holdout and walk-forward validation with Elo baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.elo import expected_score
from src.features import FEATURE_COLUMNS
from src.model import FootballOutcomeModel
from src.walk_forward import ranked_probability_score


@dataclass
class BaselineComparison:
    model_log_loss: float
    model_accuracy: float
    model_brier: float
    elo_log_loss: float
    elo_accuracy: float
    majority_log_loss: float
    majority_accuracy: float
    log_loss_vs_elo: float
    accuracy_vs_elo: float


def elo_baseline_probabilities(frame: pd.DataFrame) -> np.ndarray:
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


def majority_class_baseline(y_true: np.ndarray) -> tuple[float, float]:
    labels, counts = np.unique(y_true, return_counts=True)
    majority = labels[np.argmax(counts)]
    preds = np.full_like(y_true, majority)
    probs = np.zeros((len(y_true), 3))
    for idx, label in enumerate(labels):
        probs[:, int(label)] = counts[idx] / len(y_true)
    ll = float(log_loss(y_true, probs, labels=[0, 1, 2]))
    acc = float(accuracy_score(y_true, preds))
    return ll, acc


def compare_to_baselines(
    training_frame: pd.DataFrame,
    model: FootballOutcomeModel,
    *,
    test_fraction: float = 0.2,
) -> BaselineComparison:
    frame = training_frame.dropna(subset=FEATURE_COLUMNS + ["outcome"]).sort_values("date")
    split = int(len(frame) * (1 - test_fraction))
    test = frame.iloc[split:]
    y_test = test["outcome"].to_numpy().astype(int)

    if model.bundle is None:
        raise RuntimeError("Model not trained.")

    model_probs = np.array([model.predict_proba({c: float(row[c]) for c in FEATURE_COLUMNS}) for _, row in test.iterrows()])
    elo_probs = elo_baseline_probabilities(test)
    maj_ll, maj_acc = majority_class_baseline(y_test)

    model_ll = float(log_loss(y_test, model_probs, labels=[0, 1, 2]))
    model_acc = float(accuracy_score(y_test, model_probs.argmax(axis=1)))
    model_brier = float(np.mean([((y_test == k).mean() - model_probs[:, k].mean()) ** 2 for k in range(3)]))
    elo_ll = float(log_loss(y_test, elo_probs, labels=[0, 1, 2]))
    elo_acc = float(accuracy_score(y_test, elo_probs.argmax(axis=1)))

    return BaselineComparison(
        model_log_loss=model_ll,
        model_accuracy=model_acc,
        model_brier=model_brier,
        elo_log_loss=elo_ll,
        elo_accuracy=elo_acc,
        majority_log_loss=maj_ll,
        majority_accuracy=maj_acc,
        log_loss_vs_elo=elo_ll - model_ll,
        accuracy_vs_elo=model_acc - elo_acc,
    )


def summarize_walk_forward(folds_df: pd.DataFrame) -> dict[str, float]:
    if folds_df.empty:
        return {}
    return {
        "avg_ensemble_log_loss": float(folds_df["log_loss_ensemble"].mean()),
        "avg_ensemble_rps": float(folds_df["rps_ensemble"].mean()),
        "avg_ensemble_accuracy": float(folds_df["accuracy_ensemble"].mean()),
        "avg_multinomial_log_loss": float(folds_df["log_loss_multinomial"].mean()),
        "avg_dixon_coles_log_loss": float(folds_df["log_loss_dixon_coles"].mean()),
    }
