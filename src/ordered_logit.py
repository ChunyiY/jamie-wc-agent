"""Ordered logistic regression for W/D/L (draw-aware)."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class OrderedLogitBundle:
    thresholds: LogisticRegression
    scaler: StandardScaler
    feature_columns: list[str]


class OrderedLogitModel:
    """
    Proportional-odds style ordered logit via cumulative binary logistic models.

    Outcome order: 0 = team A win, 1 = draw, 2 = team B win.
    """

    def __init__(self) -> None:
        self.bundle: OrderedLogitBundle | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        feature_columns: list[str] | None = None,
    ) -> None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        thresholds = LogisticRegression(max_iter=2000, solver="lbfgs")
        y_cumulative = (y_train >= 1).astype(int)
        thresholds.fit(x_scaled, y_cumulative, sample_weight=sample_weight)

        self.bundle = OrderedLogitBundle(
            thresholds=thresholds,
            scaler=scaler,
            feature_columns=feature_columns or [],
        )
        self._y_train = y_train
        self._x_scaled = x_scaled
        self._sample_weight = sample_weight

        # Second threshold P(Y >= 2)
        mask = y_train >= 1
        if mask.sum() > 30:
            self._upper = LogisticRegression(max_iter=2000, solver="lbfgs")
            sw = sample_weight[mask] if sample_weight is not None else None
            self._upper.fit(x_scaled[mask], (y_train[mask] >= 2).astype(int), sample_weight=sw)
        else:
            self._upper = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.bundle is None:
            raise RuntimeError("Ordered logit model is not fitted.")
        x_scaled = self.bundle.scaler.transform(x)
        p_ge_1 = self.bundle.thresholds.predict_proba(x_scaled)[:, 1]
        if self._upper is not None:
            p_ge_2 = self._upper.predict_proba(x_scaled)[:, 1]
        else:
            p_ge_2 = p_ge_1 * 0.45

        p_draw = np.clip(p_ge_1 - p_ge_2, 0.0, 1.0)
        p_b = np.clip(p_ge_2, 0.0, 1.0)
        p_a = np.clip(1.0 - p_ge_1, 0.0, 1.0)
        probs = np.column_stack([p_a, p_draw, p_b])
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def save(self, path) -> None:
        if self.bundle is None:
            raise RuntimeError("Nothing to save.")
        joblib.dump({"bundle": self.bundle, "upper": getattr(self, "_upper", None)}, path)

    def load(self, path) -> None:
        data = joblib.load(path)
        self.bundle = data["bundle"]
        self._upper = data.get("upper")
