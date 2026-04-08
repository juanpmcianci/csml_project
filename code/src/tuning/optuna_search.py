"""Hyperparameter tuning with Optuna: per-model objectives + walk-forward CV."""

import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import rmse

logger = logging.getLogger(__name__)


# =====================================================================
# Walk-forward cross-validation scorer
# =====================================================================

def walk_forward_rmse(model_cls, params: dict, X, y, n_splits: int = 5, gap: int = 1) -> float:
    """Evaluate a model config via walk-forward expanding-window CV.

    Returns mean RMSE across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
        y_tr = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
        X_va = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
        y_va = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx]

        model = model_cls(**params)
        model.fit(X_tr, y_tr, X_va, y_va)
        preds = model.predict(X_va)

        # Handle models that return padded NaNs (e.g. LSTM)
        y_va_arr = np.asarray(y_va, dtype=float)
        preds_arr = np.asarray(preds, dtype=float)
        mask = np.isfinite(y_va_arr) & np.isfinite(preds_arr)

        if mask.sum() > 0:
            scores.append(rmse(y_va_arr[mask], preds_arr[mask]))

    return float(np.mean(scores)) if scores else float("inf")


# =====================================================================
# Per-model Optuna objectives
# =====================================================================

def ridge_objective(trial, X, y, n_splits=5):
    from src.models.baselines import RidgeModel
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    return walk_forward_rmse(RidgeModel, {"alpha": alpha}, X, y, n_splits)


def lasso_objective(trial, X, y, n_splits=5):
    from src.models.baselines import LassoModel
    alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    return walk_forward_rmse(LassoModel, {"alpha": alpha}, X, y, n_splits)


def lgbm_objective(trial, X, y, n_splits=5):
    from src.models.trees import LightGBMModel
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    return walk_forward_rmse(LightGBMModel, params, X, y, n_splits)


def xgboost_objective(trial, X, y, n_splits=5):
    from src.models.trees import XGBoostModel
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    return walk_forward_rmse(XGBoostModel, params, X, y, n_splits)


def lstm_objective(trial, X, y, n_splits=3):
    from src.models.lstm import LSTMModel
    params = {
        "seq_len": trial.suggest_categorical("seq_len", [7, 14, 21, 30]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "max_epochs": 30,
        "batch_size": 64,
        "patience": 5,
    }
    return walk_forward_rmse(LSTMModel, params, X, y, n_splits)


def tft_objective(trial, X, y, n_splits=3):
    from src.models.tft import TFTModel
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 64, 128]),
        "attention_head_size": trial.suggest_int("attention_head_size", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 32, 64]),
        "max_epochs": 20,
        "batch_size": 64,
        "patience": 5,
    }
    return walk_forward_rmse(TFTModel, params, X, y, n_splits)


# =====================================================================
# Objective registry
# =====================================================================

OBJECTIVE_REGISTRY = {
    "ridge": {"fn": ridge_objective, "n_trials": 50},
    "lasso": {"fn": lasso_objective, "n_trials": 50},
    "lgbm": {"fn": lgbm_objective, "n_trials": 100},
    "xgboost": {"fn": xgboost_objective, "n_trials": 50},
    "lstm": {"fn": lstm_objective, "n_trials": 50},
    "tft": {"fn": tft_objective, "n_trials": 30},
}


# =====================================================================
# Main tuning runner
# =====================================================================

def tune_model(
    model_name: str,
    X,
    y,
    n_trials: int | None = None,
    n_splits: int = 5,
    storage: str | None = None,
    study_name: str | None = None,
) -> optuna.Study:
    """Run Optuna hyperparameter search for a given model.

    Parameters
    ----------
    model_name : key from OBJECTIVE_REGISTRY
    X, y : training data (full — CV is done internally)
    n_trials : override default trial count
    n_splits : number of walk-forward CV folds
    storage : Optuna storage URL (e.g. "sqlite:///optuna.db")
    study_name : name for the study (defaults to model_name)

    Returns
    -------
    optuna.Study with best trial accessible via study.best_trial
    """
    if model_name not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"No objective for '{model_name}'. Available: {list(OBJECTIVE_REGISTRY.keys())}"
        )

    spec = OBJECTIVE_REGISTRY[model_name]
    objective_fn = spec["fn"]
    n_trials = n_trials or spec["n_trials"]
    study_name = study_name or model_name

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        return objective_fn(trial, X, y, n_splits)

    logger.info("Tuning %s: %d trials, %d CV folds", model_name, n_trials, n_splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best %s: RMSE=%.6f, params=%s",
        model_name,
        study.best_value,
        study.best_params,
    )
    return study


def tune_all(
    X,
    y,
    models: list[str] | None = None,
    n_splits: int = 5,
    storage: str = "sqlite:///optuna_studies.db",
) -> dict[str, optuna.Study]:
    """Tune all (or selected) models sequentially.

    Returns dict of model_name -> optuna.Study.
    """
    models = models or list(OBJECTIVE_REGISTRY.keys())
    studies = {}

    for name in models:
        logger.info("=== Tuning %s ===", name)
        studies[name] = tune_model(name, X, y, n_splits=n_splits, storage=storage)

    # Summary
    logger.info("=== Tuning Summary ===")
    for name, study in studies.items():
        logger.info("  %s: best RMSE=%.6f", name, study.best_value)

    return studies
