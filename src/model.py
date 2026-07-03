"""Match outcome models with time-based validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, TRAINING_FEATURE_COLUMNS


@dataclass
class ModelMetrics:
    model_name: str
    log_loss: float
    brier_score: float
    accuracy: float
    calibration_fractions: list[float]
    calibration_mean_predicted: list[float]
    calibration_mean_observed: list[float]


@dataclass
class TrainedModelBundle:
    model_name: str
    model: Any
    scaler: StandardScaler
    feature_columns: list[str]
    metrics: ModelMetrics


class FootballOutcomeModel:
    def __init__(self) -> None:
        self.bundle: TrainedModelBundle | None = None

    def _build_model(self, model_name: str):
        if model_name == "logistic_regression":
            return LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=42,
            )
        if model_name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        if model_name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(random_state=42)
        raise ValueError(f"Unsupported model: {model_name}")

    def train(
        self,
        training_frame: pd.DataFrame,
        *,
        model_name: str = "logistic_regression",
        test_fraction: float = 0.2,
        calibrate: bool = False,
        sample_weight: np.ndarray | None = None,
    ) -> ModelMetrics:
        frame = training_frame.dropna(subset=TRAINING_FEATURE_COLUMNS + ["outcome"]).copy()
        if len(frame) < 50:
            raise ValueError("Need at least 50 historical matches to train a stable model.")

        frame = frame.sort_values("date")
        split_index = int(len(frame) * (1 - test_fraction))
        train = frame.iloc[:split_index]
        test = frame.iloc[split_index:]

        x_train = train[TRAINING_FEATURE_COLUMNS].values
        y_train = train["outcome"].values
        x_test = test[TRAINING_FEATURE_COLUMNS].values
        y_test = test["outcome"].values
        sw_train = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            if len(sw) == len(frame):
                sw_train = sw[:split_index]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = self._build_model(model_name)
        model.fit(x_train_scaled, y_train, sample_weight=sw_train)
        if calibrate and len(train) >= 200:
            model = CalibratedClassifierCV(model, method="isotonic", cv=3)
            model.fit(x_train_scaled, y_train, sample_weight=sw_train)

        probabilities = model.predict_proba(x_test_scaled)
        predictions = model.predict(x_test_scaled)

        metrics = self._compute_metrics(model_name, y_test, probabilities, predictions)
        self.bundle = TrainedModelBundle(
            model_name=model_name,
            model=model,
            scaler=scaler,
            feature_columns=TRAINING_FEATURE_COLUMNS.copy(),
            metrics=metrics,
        )
        return metrics

    def compare_models(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for model_name in (
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
        ):
            try:
                metrics = self.train(training_frame, model_name=model_name)
                rows.append(
                    {
                        "model": model_name,
                        "log_loss": metrics.log_loss,
                        "brier_score": metrics.brier_score,
                        "accuracy": metrics.accuracy,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface model failures in UI table
                rows.append({"model": model_name, "error": str(exc)})
        return pd.DataFrame(rows)

    def predict_proba(self, features: dict[str, float]) -> np.ndarray:
        if self.bundle is None:
            raise RuntimeError("Model is not trained yet.")
        vector = np.array([[features[col] for col in self.bundle.feature_columns]])
        scaled = self.bundle.scaler.transform(vector)
        return self.bundle.model.predict_proba(scaled)[0]

    def confidence_from_probs(self, probabilities: np.ndarray) -> float:
        return float(np.max(probabilities))

    def save(self, path: Path) -> None:
        if self.bundle is None:
            raise RuntimeError("No trained model to save.")
        joblib.dump(self.bundle, path)

    def load(self, path: Path) -> None:
        self.bundle = joblib.load(path)

    def _compute_metrics(
        self,
        model_name: str,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> ModelMetrics:
        labels = sorted(np.unique(y_true))
        ll = log_loss(y_true, probabilities, labels=labels)
        brier = float(np.mean([brier_score_loss((y_true == label).astype(int), probabilities[:, idx]) for idx, label in enumerate(labels)]))
        acc = accuracy_score(y_true, predictions)
        max_probs = probabilities.max(axis=1)
        observed = (predictions == y_true).astype(float)
        try:
            frac, mean_pred = calibration_curve(observed, max_probs, n_bins=8, strategy="quantile")
            fractions = frac.tolist()
            mean_predicted = mean_pred.tolist()
            mean_observed = frac.tolist()
        except ValueError:
            fractions, mean_predicted, mean_observed = [], [], []

        return ModelMetrics(
            model_name=model_name,
            log_loss=float(ll),
            brier_score=float(brier),
            accuracy=float(acc),
            calibration_fractions=fractions,
            calibration_mean_predicted=mean_predicted,
            calibration_mean_observed=mean_observed,
        )
