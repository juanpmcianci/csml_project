"""Trading simulation: strategy execution, P&L, and performance metrics."""

import numpy as np
import pandas as pd


def simulate_trading(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy: str = "binary",
    threshold: float = 0.0,
    transaction_cost: float = 0.01,
    max_position: float = 1.0,
) -> pd.DataFrame:
    """Simulate a trading strategy based on model predictions.

    Parameters
    ----------
    y_true : array of realized returns
    y_pred : array of predicted returns
    strategy : "binary" (long/flat) or "proportional" (position ~ signal magnitude)
    threshold : minimum predicted return to trigger a trade
    transaction_cost : round-trip cost as fraction (e.g. 0.01 = 1%)
    max_position : max position size

    Returns
    -------
    DataFrame with columns: position, gross_return, cost, net_return, equity
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = len(y_true)

    positions = np.zeros(n)
    for i in range(n):
        if not mask[i]:
            continue
        if strategy == "binary":
            positions[i] = max_position if y_pred[i] > threshold else 0.0
        elif strategy == "proportional":
            positions[i] = np.clip(y_pred[i] / (np.abs(y_pred[mask]).std() + 1e-8), -max_position, max_position)

    # Costs: charged when position changes
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * transaction_cost / 2  # half round-trip on each leg

    gross_returns = positions * y_true
    net_returns = gross_returns - costs

    equity = np.cumprod(1 + net_returns)

    return pd.DataFrame({
        "position": positions,
        "gross_return": gross_returns,
        "cost": costs,
        "net_return": net_returns,
        "equity": equity,
    })


def trading_metrics(sim: pd.DataFrame, periods_per_year: float = 252) -> dict[str, float]:
    """Compute performance metrics from a simulation DataFrame."""
    net = sim["net_return"]
    equity = sim["equity"]

    # Sharpe ratio (annualized)
    mean_ret = net.mean()
    std_ret = net.std()
    sharpe = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    trades = net[net != 0]
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0.0

    # Profit factor
    gross_profit = trades[trades > 0].sum()
    gross_loss = -trades[trades < 0].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Total return
    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "n_trades": int((sim["position"].diff().abs() > 0).sum()),
        "total_cost": float(sim["cost"].sum()),
    }


def compare_strategies(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    strategy: str = "binary",
    threshold: float = 0.0,
    transaction_cost: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run trading sim for all models + buy-and-hold baseline.

    Returns
    -------
    (metrics_df, simulations_dict)
    """
    sims = {}
    rows = []

    # Model strategies
    for name, y_pred in predictions.items():
        sim = simulate_trading(y_true, y_pred, strategy=strategy,
                               threshold=threshold, transaction_cost=transaction_cost)
        sims[name] = sim
        metrics = trading_metrics(sim)
        metrics["model"] = name
        rows.append(metrics)

    # Buy-and-hold baseline
    bh_sim = simulate_trading(y_true, np.ones_like(y_true), strategy="binary",
                              threshold=-np.inf, transaction_cost=0.0)
    sims["buy_and_hold"] = bh_sim
    bh_metrics = trading_metrics(bh_sim)
    bh_metrics["model"] = "buy_and_hold"
    rows.append(bh_metrics)

    metrics_df = pd.DataFrame(rows).set_index("model")
    return metrics_df, sims


def plot_equity_curves(
    sims: dict[str, pd.DataFrame],
    title: str = "Equity Curves",
    top_n: int | None = None,
):
    """Plot equity curves for all simulations.

    Returns matplotlib Figure for saving.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Optionally limit to top N by final equity
    items = list(sims.items())
    if top_n:
        items.sort(key=lambda x: x[1]["equity"].iloc[-1], reverse=True)
        items = items[:top_n]

    for name, sim in items:
        ax.plot(sim["equity"].values, label=name, linewidth=1.2)

    ax.axhline(1.0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel("Period")
    ax.set_ylabel("Equity")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
