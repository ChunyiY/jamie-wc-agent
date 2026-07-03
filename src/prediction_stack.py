"""Unified prediction stack: Elo, Dixon–Coles, ordered logit, multinomial ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib

from src.dixon_coles import DixonColesParams
from src.ensemble import PredictionEnsemble
from src.model import FootballOutcomeModel
from src.ordered_logit import OrderedLogitModel
from src.walk_forward import WalkForwardFold


@dataclass
class PredictionStack:
    ensemble: PredictionEnsemble
    multinomial: FootballOutcomeModel
    walk_forward_folds: list[WalkForwardFold] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "ensemble": self.ensemble,
                "multinomial": self.multinomial.bundle,
                "walk_forward_folds": self.walk_forward_folds,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PredictionStack":
        data = joblib.load(path)
        ml = FootballOutcomeModel()
        ml.bundle = data["multinomial"]
        return cls(
            ensemble=data["ensemble"],
            multinomial=ml,
            walk_forward_folds=data.get("walk_forward_folds", []),
        )
