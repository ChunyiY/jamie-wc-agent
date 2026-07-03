"""Tests for walk-forward validation."""

import pandas as pd

from src.walk_forward import optimize_ensemble_weights, ranked_probability_score


def test_optimize_ensemble_weights():
    preds_dc = [(0.5, 0.25, 0.25), (0.4, 0.3, 0.3)]
    preds_ol = [[0.48, 0.27, 0.25], [0.42, 0.28, 0.30]]
    preds_ml = [[0.52, 0.23, 0.25], [0.38, 0.32, 0.30]]
    y = [0, 1]
    import numpy as np

    w = optimize_ensemble_weights(preds_dc, np.array(preds_ol), np.array(preds_ml), np.array(y))
    assert abs(sum(w) - 1.0) < 1e-6
    assert all(x >= 0 for x in w)


def test_ranked_probability_score():
    import numpy as np

    y = np.array([0, 1, 2])
    probs = np.array([[0.6, 0.2, 0.2], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    rps = ranked_probability_score(y, probs)
    assert rps >= 0
