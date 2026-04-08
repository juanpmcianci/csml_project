"""Statistical tests: Diebold-Mariano, Model Confidence Set, Mincer-Zarnowitz."""

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# =====================================================================
# Diebold-Mariano test
# =====================================================================

def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0 where d_t = L(e_a,t) - L(e_b,t).
    Uses Harvey, Leybourne & Newbold (1997) small-sample correction.

    Parameters
    ----------
    loss : "squared" or "absolute"
    h : forecast horizon (for HAC variance correction)

    Returns
    -------
    dict with "dm_stat", "p_value", "better_model" ("a", "b", or "tied")
    """
    mask = np.isfinite(y_true) & np.isfinite(pred_a) & np.isfinite(pred_b)
    yt = y_true[mask]
    ea = yt - pred_a[mask]
    eb = yt - pred_b[mask]

    if loss == "squared":
        d = ea ** 2 - eb ** 2
    else:
        d = np.abs(ea) - np.abs(eb)

    n = len(d)
    if n < 3:
        return {"dm_stat": np.nan, "p_value": np.nan, "better_model": "tied"}

    d_mean = d.mean()
    # Newey-West HAC variance estimate
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += gamma_k
    var_d = (gamma_0 + 2 * gamma_sum) / n

    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "better_model": "tied"}

    dm_raw = d_mean / np.sqrt(var_d)

    # Harvey, Leybourne & Newbold correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = dm_raw * correction

    p_value = 2 * sp_stats.t.sf(np.abs(dm_stat), df=n - 1)

    if p_value < 0.05:
        better = "a" if d_mean < 0 else "b"
    else:
        better = "tied"

    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "better_model": better}


def dm_pairwise(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    loss: str = "squared",
) -> pd.DataFrame:
    """Pairwise Diebold-Mariano tests between all model pairs.

    Returns DataFrame with (model_a, model_b) as index and DM stat + p-value.
    """
    names = list(predictions.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = diebold_mariano(
                y_true, predictions[names[i]], predictions[names[j]], loss=loss
            )
            rows.append({
                "model_a": names[i],
                "model_b": names[j],
                **result,
            })
    return pd.DataFrame(rows)


def dm_pvalue_matrix(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    loss: str = "squared",
) -> pd.DataFrame:
    """Return a symmetric matrix of DM p-values for heatmap plotting."""
    names = list(predictions.keys())
    n = len(names)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            result = diebold_mariano(y_true, predictions[names[i]], predictions[names[j]], loss=loss)
            mat[i, j] = result["p_value"]
            mat[j, i] = result["p_value"]

    return pd.DataFrame(mat, index=names, columns=names)


# =====================================================================
# Model Confidence Set
# =====================================================================

def model_confidence_set(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    alpha: float = 0.10,
    loss: str = "squared",
) -> dict:
    """Model Confidence Set (Hansen et al., 2011).

    Uses the arch package if available, otherwise falls back to
    a simple sequential elimination based on DM tests.

    Returns
    -------
    dict with "surviving_models" (list), "eliminated" (list), "method" (str)
    """
    try:
        from arch.bootstrap import MCS as ArchMCS

        # Build loss matrix: rows = observations, columns = models
        names = list(predictions.keys())
        mask = np.isfinite(y_true)
        for p in predictions.values():
            mask &= np.isfinite(p)

        yt = y_true[mask]
        if loss == "squared":
            losses = np.column_stack([(yt - predictions[n][mask]) ** 2 for n in names])
        else:
            losses = np.column_stack([np.abs(yt - predictions[n][mask]) for n in names])

        loss_df = pd.DataFrame(losses, columns=names)
        mcs = ArchMCS(loss_df, size=alpha)
        mcs.compute()

        surviving = mcs.included
        eliminated = mcs.excluded

        return {
            "surviving_models": list(surviving),
            "eliminated": list(eliminated),
            "method": "arch.MCS",
        }

    except ImportError:
        logger.warning("arch.bootstrap.MCS not available; using DM-based fallback")

    # Fallback: sequential elimination
    remaining = list(predictions.keys())
    eliminated = []

    while len(remaining) > 1:
        worst = None
        worst_losses = -np.inf
        for name in remaining:
            mask = np.isfinite(y_true) & np.isfinite(predictions[name])
            errs = y_true[mask] - predictions[name][mask]
            avg_loss = np.mean(errs ** 2) if loss == "squared" else np.mean(np.abs(errs))
            if avg_loss > worst_losses:
                worst_losses = avg_loss
                worst = name

        # Test if worst is significantly worse than all others
        significantly_worse = True
        for other in remaining:
            if other == worst:
                continue
            dm = diebold_mariano(y_true, predictions[worst], predictions[other], loss=loss)
            if dm["p_value"] > alpha or dm["better_model"] != "b":
                significantly_worse = False
                break

        if significantly_worse:
            remaining.remove(worst)
            eliminated.append(worst)
        else:
            break

    return {
        "surviving_models": remaining,
        "eliminated": eliminated,
        "method": "dm_sequential",
    }


# =====================================================================
# Mincer-Zarnowitz regression
# =====================================================================

def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mincer-Zarnowitz test: regress realized on predicted.

    y_true = alpha + beta * y_pred + epsilon
    H0: alpha = 0, beta = 1 (unbiased forecasts)

    Returns
    -------
    dict with alpha, beta, their p-values, and joint F-test p-value.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    if len(yt) < 5:
        return {"alpha": np.nan, "beta": np.nan, "f_pvalue": np.nan}

    X = sm.add_constant(yp)
    model = sm.OLS(yt, X).fit()

    alpha_hat = model.params[0]
    beta_hat = model.params[1]

    # Joint test: H0: alpha=0, beta=1
    r_matrix = np.array([[1, 0], [0, 1]])
    q = np.array([0, 1])
    try:
        f_test = model.f_test((r_matrix, q))
        f_pvalue = float(f_test.pvalue)
    except Exception:
        f_pvalue = np.nan

    return {
        "alpha": float(alpha_hat),
        "alpha_pvalue": float(model.pvalues[0]),
        "beta": float(beta_hat),
        "beta_pvalue": float(model.pvalues[1]),
        "f_pvalue": f_pvalue,
        "r_squared": float(model.rsquared),
    }


def mincer_zarnowitz_all(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Run Mincer-Zarnowitz test for all models."""
    rows = []
    for name, y_pred in predictions.items():
        result = mincer_zarnowitz(y_true, y_pred)
        result["model"] = name
        rows.append(result)
    return pd.DataFrame(rows).set_index("model")
