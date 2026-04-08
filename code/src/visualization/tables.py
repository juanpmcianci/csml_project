"""LaTeX table formatters for the paper."""

import pandas as pd


def _fmt_float(x, decimals=4):
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


# =====================================================================
# Table 1 — Descriptive statistics
# =====================================================================

def descriptive_stats_table(
    df: pd.DataFrame,
    contract_col: str = "condition_id",
    price_col: str = "price",
    vol_col: str = "volume",
) -> pd.DataFrame:
    """Generate descriptive statistics table (Table 1 in paper)."""
    stats = {}

    if contract_col in df.columns:
        stats["Contracts"] = df[contract_col].nunique()
    stats["Observations"] = len(df)
    stats["Date range"] = f"{df.index.min().date()} — {df.index.max().date()}"

    if price_col in df.columns:
        p = df[price_col]
        stats["Price mean"] = p.mean()
        stats["Price std"] = p.std()
        stats["Price min"] = p.min()
        stats["Price max"] = p.max()
        stats["Price median"] = p.median()

    if vol_col in df.columns:
        v = df[vol_col]
        stats["Volume mean"] = v.mean()
        stats["Volume median"] = v.median()

    return pd.DataFrame({"Statistic": stats.keys(), "Value": stats.values()}).set_index("Statistic")


# =====================================================================
# Table 2 — Forecast accuracy comparison
# =====================================================================

def forecast_accuracy_table(
    metrics_df: pd.DataFrame,
    bold_best: bool = True,
) -> str:
    """Format model comparison DataFrame as LaTeX table.

    Bolds the best (lowest) value in each metric column.
    """
    df = metrics_df.copy()

    if bold_best:
        for col in df.select_dtypes(include="number").columns:
            best_idx = df[col].idxmin()
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
            df.at[best_idx, col] = f"\\textbf{{{df.at[best_idx, col]}}}"

    latex = df.to_latex(
        escape=False,
        caption="Out-of-sample forecast accuracy across models.",
        label="tab:forecast_accuracy",
        column_format="l" + "r" * len(df.columns),
    )
    return latex


# =====================================================================
# Table 3 — Feature importance ranking
# =====================================================================

def feature_importance_table(
    importances: dict[str, pd.Series],
    top_n: int = 20,
) -> pd.DataFrame:
    """Side-by-side feature importance from multiple methods.

    Parameters
    ----------
    importances : dict mapping method_name -> pd.Series of importances

    Returns
    -------
    DataFrame with features as rows, methods as columns (ranked).
    """
    frames = []
    for name, imp in importances.items():
        top = imp.sort_values(ascending=False).head(top_n)
        ranked = pd.DataFrame({
            f"{name}_feature": top.index,
            f"{name}_score": top.values,
        }).reset_index(drop=True)
        ranked.index = range(1, len(ranked) + 1)
        ranked.index.name = "rank"
        frames.append(ranked)

    return pd.concat(frames, axis=1)


# =====================================================================
# Table 4 — Trading simulation summary
# =====================================================================

def trading_summary_table(
    metrics_df: pd.DataFrame,
    bold_best: bool = True,
) -> str:
    """Format trading metrics as LaTeX table."""
    cols = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]
    available = [c for c in cols if c in metrics_df.columns]
    df = metrics_df[available].copy()

    if bold_best:
        best_map = {
            "total_return": "max", "sharpe_ratio": "max",
            "max_drawdown": "max",  # least negative
            "win_rate": "max", "profit_factor": "max",
        }
        for col in available:
            best_fn = best_map.get(col, "max")
            best_idx = df[col].idxmax() if best_fn == "max" else df[col].idxmin()
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
            df.at[best_idx, col] = f"\\textbf{{{df.at[best_idx, col]}}}"

    latex = df.to_latex(
        escape=False,
        caption="Trading simulation performance across models.",
        label="tab:trading_sim",
        column_format="l" + "r" * len(available),
    )
    return latex


# =====================================================================
# Generic export
# =====================================================================

def export_table(
    df: pd.DataFrame,
    path: str,
    caption: str = "",
    label: str = "",
    fmt: str = "latex",
):
    """Export a DataFrame as LaTeX or CSV."""
    if fmt == "latex":
        latex = df.to_latex(
            caption=caption,
            label=label,
            column_format="l" + "r" * len(df.columns),
        )
        with open(path, "w") as f:
            f.write(latex)
    elif fmt == "csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")
