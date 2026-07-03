"""Walk-forward (rolling-origin) validation for football outcome models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.dixon_coles import fit_dixon_coles, predict_dixon_coles
from src.features import TRAINING_FEATURE_COLUMNS
from src.model import FootballOutcomeModel
from src.ordered_logit import OrderedLogitModel
from src.time_decay import match_sample_weights


@dataclass
class WalkForwardFold:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    log_loss_ensemble: float
    log_loss_multinomial: float
    log_loss_dixon_coles: float
    log_loss_ordered: float
    accuracy_ensemble: float
    rps_ensemble: float


def ranked_probability_score(y_true: np.ndarray, prob_matrix: np.ndarray) -> float:
    """RPS for ordered outcomes 0,1,2 (lower is better)."""
    classes = np.array([0, 1, 2])
    score = 0.0
    for idx, y in enumerate(y_true):
        cum_pred = np.cumsum(prob_matrix[idx])
        cum_obs = np.array([(classes <= y).mean() for _ in classes])
        score += np.mean((cum_pred - cum_obs) ** 2)
    return float(score / max(len(y_true), 1))


def _ensemble_probs(
    p_dc: tuple[float, float, float],
    p_ol: np.ndarray,
    p_ml: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    w_dc, w_ol, w_ml = weights
    vec = (
        w_dc * np.array(p_dc)
        + w_ol * p_ol
        + w_ml * p_ml
    )
    vec = np.clip(vec, 1e-9, 1.0)
    return vec / vec.sum()


def optimize_ensemble_weights(
    preds_dc: list[tuple[float, float, float]],
    preds_ol: np.ndarray,
    preds_ml: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float, float]:
    """Grid search ensemble weights minimizing log-loss on validation fold."""
    best_w = (1 / 3, 1 / 3, 1 / 3)
    best_ll = float("inf")
    grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65]
    for w_dc in grid:
        for w_ol in grid:
            w_ml = 1.0 - w_dc - w_ol
            if w_ml < -0.01:
                continue
            w_ml = max(0.0, w_ml)
            total = w_dc + w_ol + w_ml
            if total <= 0:
                continue
            w = (w_dc / total, w_ol / total, w_ml / total)
            probs = np.array(
                [_ensemble_probs(p_dc, preds_ol[i], preds_ml[i], w) for i, p_dc in enumerate(preds_dc)]
            )
            ll = log_loss(y_true, probs, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll = ll
                best_w = w
    return best_w


def run_walk_forward_validation(
    results: pd.DataFrame,
    training_frame: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train: int = 2000,
) -> tuple[list[WalkForwardFold], tuple[float, float, float]]:
    """
  Rolling-origin validation on historical results only.

    Returns per-fold metrics and optimal ensemble weights from the last fold.
    """
    frame = results.sort_values("date").reset_index(drop=True)
    train_df = training_frame.sort_values("date").reset_index(drop=True)
    if len(frame) < min_train + 500:
        n_folds = max(2, min(n_folds, 3))

    dates = frame["date"].sort_values().unique()
    fold_size = max(400, len(dates) // (n_folds + 1))
    folds: list[WalkForwardFold] = []
    best_weights = (0.34, 0.33, 0.33)

    for fold_i in range(n_folds):
        cut_idx = min_train + fold_i * fold_size
        if cut_idx >= len(frame) - 200:
            break
        train_end = frame.iloc[cut_idx - 1]["date"]
        test_start = frame.iloc[cut_idx]["date"]
        test_end_idx = min(cut_idx + fold_size, len(frame) - 1)
        test_end = frame.iloc[test_end_idx]["date"]

        train_results = frame[frame["date"] <= train_end]
        test_results = frame[(frame["date"] >= test_start) & (frame["date"] <= test_end)]
        train_features = train_df[train_df["date"] <= train_end]
        test_features = train_df[(train_df["date"] >= test_start) & (train_df["date"] <= test_end)]

        if len(test_results) < 50 or len(train_features) < 100:
            continue

        sw = match_sample_weights(train_results, prediction_context="world_cup")
        dc_params = fit_dixon_coles(train_results, sample_weights=sw)

        x_train = train_features[TRAINING_FEATURE_COLUMNS].to_numpy()
        y_train = train_features["outcome"].to_numpy().astype(int)
        sw_f = match_sample_weights(train_features, prediction_context="world_cup")

        ol = OrderedLogitModel()
        ol.fit(x_train, y_train, sample_weight=sw_f, feature_columns=TRAINING_FEATURE_COLUMNS.copy())

        ml = FootballOutcomeModel()
        sw_f = match_sample_weights(train_features, prediction_context="world_cup")
        ml.train(train_features, model_name="logistic_regression", calibrate=True, sample_weight=sw_f)

        preds_dc: list[tuple[float, float, float]] = []
        preds_ol = []
        preds_ml = []
        y_test = []

        for _, row in test_features.iterrows():
            p_dc, _ = predict_dixon_coles(
                dc_params,
                row["team_a"],
                row["team_b"],
                neutral=bool(row.get("neutral_venue", 1.0) >= 0.5),
            )
            x_row = row[TRAINING_FEATURE_COLUMNS].to_numpy().reshape(1, -1)
            p_ol = ol.predict_proba(x_row)[0]
            feats = {col: float(row[col]) for col in TRAINING_FEATURE_COLUMNS}
            p_ml = ml.predict_proba(feats)
            preds_dc.append(p_dc)
            preds_ol.append(p_ol)
            preds_ml.append(p_ml)
            y_test.append(int(row["outcome"]))

        y_arr = np.array(y_test)
        ol_mat = np.array(preds_ol)
        ml_mat = np.array(preds_ml)
        weights = optimize_ensemble_weights(preds_dc, ol_mat, ml_mat, y_arr)
        best_weights = weights

        ens = np.array([_ensemble_probs(p_dc, ol_mat[i], ml_mat[i], weights) for i, p_dc in enumerate(preds_dc)])
        dc_mat = np.array([list(p) for p in preds_dc])

        folds.append(
            WalkForwardFold(
                fold=fold_i + 1,
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                n_train=len(train_results),
                n_test=len(test_features),
                log_loss_ensemble=float(log_loss(y_arr, ens, labels=[0, 1, 2])),
                log_loss_multinomial=float(log_loss(y_arr, ml_mat, labels=[0, 1, 2])),
                log_loss_dixon_coles=float(log_loss(y_arr, dc_mat, labels=[0, 1, 2])),
                log_loss_ordered=float(log_loss(y_arr, ol_mat, labels=[0, 1, 2])),
                accuracy_ensemble=float(accuracy_score(y_arr, ens.argmax(axis=1))),
                rps_ensemble=ranked_probability_score(y_arr, ens),
            )
        )

    return folds, best_weights


def folds_to_dataframe(folds: list[WalkForwardFold]) -> pd.DataFrame:
    if not folds:
        return pd.DataFrame()
    return pd.DataFrame([f.__dict__ for f in folds])
