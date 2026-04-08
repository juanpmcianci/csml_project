"""Publication-quality figure generators for the paper."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _apply_style():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })


# =====================================================================
# Fig 1 — Sample contract price trajectories
# =====================================================================

def plot_price_trajectories(
    histories: dict[str, pd.DataFrame],
    price_col: str = "price",
    n_contracts: int = 6,
    savepath: str | None = None,
) -> plt.Figure:
    """Plot price trajectories for a set of contracts."""
    _apply_style()
    items = list(histories.items())[:n_contracts]
    n = len(items)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (name, df) in enumerate(items):
        ax = axes_flat[i]
        ax.plot(df.index, df[price_col], linewidth=0.8, color="C0")
        ax.set_title(name[:45], fontsize=9)
        ax.set_ylabel("Price")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Sample Contract Price Trajectories", fontsize=12, y=1.01)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 2 — Feature correlation heatmap
# =====================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    top_n: int = 30,
    savepath: str | None = None,
) -> plt.Figure:
    """Plot correlation heatmap for top N numeric features."""
    _apply_style()
    numeric = df.select_dtypes(include="number")
    # Select top features by variance
    variances = numeric.var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    corr = numeric[top_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f"Feature Correlation (top {top_n} by variance)")
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 3 — Model comparison bar chart
# =====================================================================

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metrics: list[str] = ["mae", "rmse"],
    savepath: str | None = None,
) -> plt.Figure:
    """Bar chart comparing models across metrics."""
    _apply_style()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        values = metrics_df[metric].sort_values()
        colors = ["C2" if v == values.min() else "C0" for v in values]
        values.plot.barh(ax=ax, color=colors, edgecolor="k", alpha=0.8)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Model Comparison — {metric.upper()}")
        ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 4 — Ablation: endogenous-only vs full feature set
# =====================================================================

def plot_ablation(
    endo_metrics: pd.DataFrame,
    full_metrics: pd.DataFrame,
    metric: str = "rmse",
    savepath: str | None = None,
) -> plt.Figure:
    """Grouped bar chart: endogenous-only vs full feature set per model."""
    _apply_style()
    models = endo_metrics.index.intersection(full_metrics.index)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, endo_metrics.loc[models, metric], width,
           label="Endogenous only", color="C0", edgecolor="k", alpha=0.8)
    ax.bar(x + width / 2, full_metrics.loc[models, metric], width,
           label="Full (endo + exog)", color="C1", edgecolor="k", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Ablation Study — {metric.upper()}: Endogenous vs Full Features")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 5 — Feature importance (horizontal bar)
# =====================================================================

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    savepath: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    _apply_style()
    top = importance.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    top.plot.barh(ax=ax, color="C0", edgecolor="k", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 6 — SHAP summary plot
# =====================================================================

def plot_shap_summary(
    model,
    X_test,
    top_n: int = 20,
    savepath: str | None = None,
) -> plt.Figure:
    """SHAP beeswarm summary plot for a tree model."""
    import shap

    _apply_style()
    explainer = shap.TreeExplainer(model.model)

    if isinstance(X_test, pd.DataFrame):
        X_arr = X_test.values
        feature_names = list(X_test.columns)
    else:
        X_arr = np.asarray(X_test)
        feature_names = None

    shap_values = explainer.shap_values(X_arr)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    shap.summary_plot(
        shap_values, X_arr, feature_names=feature_names,
        max_display=top_n, show=False,
    )
    fig = plt.gcf()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# =====================================================================
# Fig 7 — Equity curves (delegates to trading_sim)
# =====================================================================

def plot_equity_curves(
    sims: dict[str, pd.DataFrame],
    top_n: int = 5,
    savepath: str | None = None,
) -> plt.Figure:
    """Equity curves for top models + buy-and-hold."""
    from src.evaluation.trading_sim import plot_equity_curves as _plot_eq

    fig = _plot_eq(sims, title="Equity Curves — Model Strategies vs Buy-and-Hold", top_n=top_n)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    return fig


# =====================================================================
# Fig 8 — DM p-value heatmap
# =====================================================================

def plot_dm_heatmap(
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
    savepath: str | None = None,
) -> plt.Figure:
    """Heatmap of Diebold-Mariano p-values."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.eye(len(pvalue_matrix), dtype=bool)
    sns.heatmap(
        pvalue_matrix, ax=ax, annot=True, fmt=".3f",
        cmap="RdYlGn_r", vmin=0, vmax=0.2,
        mask=mask, square=True, linewidths=0.5,
        cbar_kws={"label": "p-value"},
    )
    ax.set_title(f"Diebold-Mariano p-values (sig. at {alpha})")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
