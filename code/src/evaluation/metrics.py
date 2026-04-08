"""Forecast evaluation metrics: regression, directional, and grouped."""

import numpy as np
import pandas as pd


# =====================================================================
# Regression metrics
# =====================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. Uses eps to avoid division by zero."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² (1 - SSres/SStot)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def theils_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil's U statistic relative to naive persistence (predict zero change).

    U < 1 means model beats persistence, U > 1 means worse.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    model_mse = np.mean((yt - yp) ** 2)
    naive_mse = np.mean(yt ** 2)  # persistence predicts 0 return
    if naive_mse == 0:
        return float("inf")
    return float(np.sqrt(model_mse / naive_mse))


# =====================================================================
# Directional metrics
# =====================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct sign predictions."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def weighted_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy weighted by magnitude of the true move."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    correct = (np.sign(yt) == np.sign(yp)).astype(float)
    weights = np.abs(yt)
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float((correct * weights).sum() / total_weight)


def hit_rate_by_quintile(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Directional accuracy broken down by quintile of predicted magnitude."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    df = pd.DataFrame({"y_true": yt, "y_pred": yp, "abs_pred": np.abs(yp)})
    df["quintile"] = pd.qcut(df["abs_pred"], 5, labels=False, duplicates="drop")
    result = df.groupby("quintile").apply(
        lambda g: pd.Series({
            "n": len(g),
            "hit_rate": (np.sign(g["y_true"]) == np.sign(g["y_pred"])).mean(),
            "mean_abs_pred": g["abs_pred"].mean(),
        })
    )
    return result


# =====================================================================
# Aggregate report
# =====================================================================

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2_oos": r2_oos(y_true, y_pred),
        "theils_u": theils_u(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "weighted_directional_accuracy": weighted_directional_accuracy(y_true, y_pred),
    }


def compare_models(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare multiple models side-by-side.

    Parameters
    ----------
    y_true : array
    predictions : dict mapping model_name -> predicted array

    Returns
    -------
    DataFrame with models as rows, metrics as columns.
    """
    rows = []
    for name, y_pred in predictions.items():
        metrics = compute_all_metrics(y_true, y_pred)
        metrics["model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model")
