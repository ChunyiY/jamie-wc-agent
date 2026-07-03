"""Fast football model loading with optional progress reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.dixon_coles import fit_dixon_coles, predict_dixon_coles
from src.elo import EloSystem
from src.ensemble import PredictionEnsemble
from src.features import (
    FEATURE_COLUMNS,
    TRAINING_FEATURE_COLUMNS,
    build_training_frame_point_in_time,
    elo_lookup_from_history,
)
from src.match_analyzer import MatchAnalyzer
from src.model import FootballOutcomeModel
from src.ordered_logit import OrderedLogitModel
from src.player_stats import load_goalscorers
from src.prediction_stack import PredictionStack
from src.match_stats import (
    FIFA_RANKINGS_FILENAME,
    MATCH_STATS_FILENAME,
    load_fifa_rankings,
    load_intl_match_stats,
)
from src.shootouts import load_shootouts
from src.squad_data import load_per90_csv, load_squads_csv, merge_squads_per90
from src.time_decay import match_sample_weights
from src.walk_forward import optimize_ensemble_weights, run_walk_forward_validation

ProgressCallback = Callable[[float, str], None]

MODEL_FILENAME = "logistic_regression.joblib"
ENSEMBLE_FILENAME = "ensemble_stack.joblib"
META_FILENAME = "model_meta.json"


@dataclass
class LoadSummary:
    source: str
    elo_matches: int
    training_matches: int
    teams: int
    calibrated: bool
    seconds_hint: str
    ensemble_weights: tuple[float, float, float] = (0.35, 0.30, 0.35)
    walk_forward_folds: int = 0


def _noop_progress(_pct: float, _msg: str) -> None:
    return


def results_fingerprint(results: pd.DataFrame, config: AppConfig) -> str:
    payload = (
        f"{len(results)}|{config.modeling_match_limit}|{config.elo_match_limit}|"
        f"{config.fast_train_skip_calibration}|ensemble_v3_competition_tactics"
    )
    if not results.empty and "date" in results.columns:
        payload += f"|{results['date'].iloc[-1]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _model_path(config: AppConfig) -> Path:
    root = Path(__file__).resolve().parent.parent / "models"
    return root / MODEL_FILENAME


def _ensemble_path(config: AppConfig) -> Path:
    root = Path(__file__).resolve().parent.parent / "models"
    return root / ENSEMBLE_FILENAME


def _meta_file(config: AppConfig) -> Path:
    root = Path(__file__).resolve().parent.parent / "models"
    return root / META_FILENAME


def _load_squad_merged(config: AppConfig) -> pd.DataFrame:
    squads = load_squads_csv(config.wc_squads_csv)
    per90 = load_per90_csv(config.wc_per90_csv)
    return merge_squads_per90(squads, per90)


def _load_goalscorers_frame(config: AppConfig) -> pd.DataFrame:
    return load_goalscorers(config.goalscorers_csv)


def _load_shootouts_frame(config: AppConfig) -> pd.DataFrame:
    path = config.results_csv.parent / "shootouts.csv"
    return load_shootouts(path)


def _load_intl_stats_frame(config: AppConfig) -> pd.DataFrame:
    return load_intl_match_stats(config.results_csv.parent / MATCH_STATS_FILENAME)


def _load_fifa_rankings_frame(config: AppConfig) -> pd.DataFrame:
    return load_fifa_rankings(config.results_csv.parent / FIFA_RANKINGS_FILENAME)


def _analyzer_kwargs(config: AppConfig) -> dict:
    return {
        "goalscorers": _load_goalscorers_frame(config),
        "merged_squads": _load_squad_merged(config),
        "shootouts": _load_shootouts_frame(config),
        "intl_match_stats": _load_intl_stats_frame(config),
        "fifa_rankings": _load_fifa_rankings_frame(config),
    }


def _save_meta(
    config: AppConfig,
    fingerprint: str,
    *,
    calibrated: bool,
    training_matches: int,
    ensemble_weights: tuple[float, float, float],
    walk_forward_folds: int,
    baseline: dict | None = None,
) -> None:
    path = _meta_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "calibrated": calibrated,
        "training_matches": training_matches,
        "ensemble_weights": list(ensemble_weights),
        "walk_forward_folds": walk_forward_folds,
    }
    if baseline:
        payload["baseline"] = baseline
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_matches(config: AppConfig, fingerprint: str) -> bool:
    path = _meta_file(config)
    model_path = _model_path(config)
    ensemble_path = _ensemble_path(config)
    if not path.exists() or not model_path.exists() or not ensemble_path.exists():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return meta.get("fingerprint") == fingerprint


def _elo_frame(results: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    limit = config.elo_match_limit
    if len(results) > limit:
        return results.sort_values("date").tail(limit)
    return results.sort_values("date")


def _modeling_frame(results: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    limit = config.modeling_match_limit
    if len(results) > limit:
        return results.sort_values("date").tail(limit)
    return results.sort_values("date")


def _optimize_weights_on_holdout(
    training: pd.DataFrame,
    dc_params,
    ol: OrderedLogitModel,
    ml: FootballOutcomeModel,
    *,
    test_fraction: float = 0.2,
) -> tuple[float, float, float]:
    frame = training.sort_values("date").reset_index(drop=True)
    split = int(len(frame) * (1 - test_fraction))
    if split < 50 or len(frame) - split < 20:
        return (0.35, 0.30, 0.35)

    test = frame.iloc[split:]
    preds_dc: list[tuple[float, float, float]] = []
    preds_ol = []
    preds_ml = []
    y_test = []

    for _, row in test.iterrows():
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

    return optimize_ensemble_weights(
        preds_dc,
        np.array(preds_ol),
        np.array(preds_ml),
        np.array(y_test),
    )


def load_from_disk(
    results: pd.DataFrame,
    config: AppConfig,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[EloSystem, FootballOutcomeModel, MatchAnalyzer, LoadSummary]:
    report = progress or _noop_progress
    report(0.15, "Loading saved Elo ratings…")
    elo = EloSystem(config)
    if config.elo_table_path.exists():
        elo.load(config.elo_table_path)
    else:
        elo_frame = _elo_frame(results, config)
        elo.fit_history(elo_frame)
        elo.save(config.elo_table_path)

    report(0.45, "Loading saved ML ensemble…")
    model = FootballOutcomeModel()
    model.load(_model_path(config))
    stack = PredictionStack.load(_ensemble_path(config))

    report(0.85, "Preparing match analyzer…")
    extra = _analyzer_kwargs(config)
    analyzer = MatchAnalyzer(
        results, elo, model, ensemble=stack.ensemble,
        goalscorers=extra["goalscorers"],
        merged_squads=extra["merged_squads"],
        shootouts=extra["shootouts"],
        intl_match_stats=extra["intl_match_stats"],
        fifa_rankings=extra["fifa_rankings"],
    )
    meta = json.loads(_meta_file(config).read_text(encoding="utf-8"))
    weights = tuple(meta.get("ensemble_weights", [0.35, 0.30, 0.35]))
    report(1.0, "Ready — loaded from cache on disk.")
    return elo, model, analyzer, LoadSummary(
        source="disk_cache",
        elo_matches=len(_elo_frame(results, config)),
        training_matches=int(meta.get("training_matches", 0)),
        teams=len(elo.ratings),
        calibrated=bool(meta.get("calibrated", False)),
        seconds_hint="< 1s",
        ensemble_weights=weights,  # type: ignore[arg-type]
        walk_forward_folds=int(meta.get("walk_forward_folds", 0)),
    )


def train_football_stack(
    results: pd.DataFrame,
    config: AppConfig | None = None,
    *,
    calibrate: bool | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[EloSystem, FootballOutcomeModel, MatchAnalyzer, LoadSummary]:
    """Train Elo + Dixon–Coles + ordered logit + multinomial ensemble."""
    config = config or load_config()
    report = progress or _noop_progress
    if calibrate is not None:
        use_calibration = calibrate
    else:
        use_calibration = not config.fast_train_skip_calibration

    fingerprint = results_fingerprint(results, config)
    elo_frame = _elo_frame(results, config)
    modeling_frame = _modeling_frame(results, config)

    report(0.05, f"Computing Elo for {len(elo_frame):,} recent matches…")
    elo = EloSystem(config)
    elo_history = elo.fit_history(elo_frame)
    elo.save(config.elo_table_path)
    lookup = elo_lookup_from_history(elo_history)
    report(0.20, f"Elo done — {len(elo.ratings)} teams rated.")

    goalscorers = _load_goalscorers_frame(config)
    merged_squads = _load_squad_merged(config)

    report(0.25, f"Building features for {len(modeling_frame):,} matches…")

    def feature_progress(done: int, total: int) -> None:
        pct = 0.25 + (done / max(total, 1)) * 0.35
        report(pct, f"Feature engineering… {done:,}/{total:,} matches")

    training = build_training_frame_point_in_time(
        modeling_frame,
        elo_lookup=lookup,
        progress_callback=feature_progress,
        goalscorers=goalscorers,
        merged_squads=merged_squads,
    )
    report(0.62, f"Features ready — {len(training):,} training rows.")

    sw_results = match_sample_weights(modeling_frame, prediction_context="world_cup")
    report(0.65, "Fitting Dixon–Coles score model…")
    dc_params = fit_dixon_coles(modeling_frame, sample_weights=sw_results)

    sw_train = match_sample_weights(training, prediction_context="world_cup")
    x_train = training[TRAINING_FEATURE_COLUMNS].to_numpy()
    y_train = training["outcome"].to_numpy().astype(int)

    report(0.72, "Training ordered logit (draw-aware)…")
    ol = OrderedLogitModel()
    ol.fit(x_train, y_train, sample_weight=sw_train, feature_columns=TRAINING_FEATURE_COLUMNS.copy())

    model = FootballOutcomeModel()
    model_path = _model_path(config)
    report(0.78, "Training multinomial logistic regression…")
    try:
        model.train(
            training,
            model_name="logistic_regression",
            calibrate=use_calibration,
            sample_weight=sw_train,
        )
        model.save(model_path)
    except Exception:
        if model_path.exists():
            model.load(model_path)
        else:
            raise

    report(0.85, "Optimizing ensemble weights…")
    weights = _optimize_weights_on_holdout(training, dc_params, ol, model)
    ensemble = PredictionEnsemble(
        dixon_coles=dc_params,
        ordered_logit=ol,
        multinomial=model,
        weights=weights,
    )

    from src.model_validation import compare_to_baselines

    baseline_dict = None
    try:
        cmp = compare_to_baselines(training, model)
        baseline_dict = {
            "model_log_loss": cmp.model_log_loss,
            "model_accuracy": cmp.model_accuracy,
            "elo_log_loss": cmp.elo_log_loss,
            "elo_accuracy": cmp.elo_accuracy,
            "majority_accuracy": cmp.majority_accuracy,
            "log_loss_vs_elo": cmp.log_loss_vs_elo,
            "accuracy_vs_elo": cmp.accuracy_vs_elo,
        }
    except Exception:
        baseline_dict = None

    walk_folds = []
    if use_calibration and len(modeling_frame) >= 2000:
        report(0.90, "Running walk-forward validation (3 folds)…")
        try:
            walk_folds, wf_weights = run_walk_forward_validation(
                modeling_frame,
                training,
                n_folds=3,
                min_train=min(1500, len(modeling_frame) // 3),
            )
            if walk_folds:
                weights = wf_weights
                ensemble.weights = weights
        except Exception:
            walk_folds = []

    stack = PredictionStack(ensemble=ensemble, multinomial=model, walk_forward_folds=walk_folds)
    stack.save(_ensemble_path(config))
    _save_meta(
        config,
        fingerprint,
        calibrated=use_calibration,
        training_matches=len(modeling_frame),
        ensemble_weights=weights,
        walk_forward_folds=len(walk_folds),
        baseline=baseline_dict,
    )

    report(0.95, "Wiring match analyzer…")
    analyzer = MatchAnalyzer(
        results, elo, model, ensemble=ensemble,
        **_analyzer_kwargs(config),
    )
    report(1.0, "Model ready.")
    return elo, model, analyzer, LoadSummary(
        source="trained",
        elo_matches=len(elo_frame),
        training_matches=len(modeling_frame),
        teams=len(elo.ratings),
        calibrated=use_calibration,
        seconds_hint="~20–60s",
        ensemble_weights=weights,
        walk_forward_folds=len(walk_folds),
    )


def load_or_train_football_stack(
    results: pd.DataFrame,
    config: AppConfig | None = None,
    *,
    force_retrain: bool = False,
    calibrate: bool | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[EloSystem, FootballOutcomeModel, MatchAnalyzer, LoadSummary]:
    config = config or load_config()
    fingerprint = results_fingerprint(results, config)
    if not force_retrain and _meta_matches(config, fingerprint):
        return load_from_disk(results, config, progress=progress)
    return train_football_stack(results, config, calibrate=calibrate, progress=progress)
