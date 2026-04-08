"""Evaluate trained models: metrics, statistical tests, trading simulation."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import compare_models, hit_rate_by_quintile
from src.evaluation.statistical_tests import (
    dm_pairwise, dm_pvalue_matrix, model_confidence_set,
    mincer_zarnowitz_all,
)
from src.evaluation.trading_sim import compare_strategies, plot_equity_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--transaction-cost", type=float, default=0.01)
    parser.add_argument("--strategy", type=str, default="binary", choices=["binary", "proportional"])
    parser.add_argument("--mcs-alpha", type=float, default=0.10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load predictions
    pred_path = results_dir / "predictions.parquet"
    if not pred_path.exists():
        logger.error("No predictions found at %s. Run 03_train_models.py first.", pred_path)
        sys.exit(1)

    pred_df = pd.read_parquet(pred_path)
    y_true = pred_df["y_true"].values
    model_cols = [c for c in pred_df.columns if c != "y_true"]
    predictions = {name: pred_df[name].values for name in model_cols}

    logger.info("Loaded predictions for %d models, %d observations", len(model_cols), len(y_true))

    # ── 1. Metrics ──────────────────────────────────────────────────
    logger.info("=== Forecast Metrics ===")
    metrics_df = compare_models(y_true, predictions)
    print(metrics_df.to_string())
    metrics_df.to_parquet(results_dir / "metrics.parquet")
    logger.info("Saved metrics.parquet")

    # Hit rate by quintile for best model
    best_model = metrics_df["rmse"].idxmin()
    quintiles = hit_rate_by_quintile(y_true, predictions[best_model])
    print(f"\nHit rate by quintile ({best_model}):")
    print(quintiles.to_string())

    # ── 2. Statistical Tests ────────────────────────────────────────
    logger.info("=== Statistical Tests ===")

    # Diebold-Mariano pairwise
    dm_pairs = dm_pairwise(y_true, predictions, loss="squared")
    print("\nDiebold-Mariano pairwise:")
    print(dm_pairs.to_string())
    dm_pairs.to_parquet(results_dir / "dm_pairwise.parquet")

    # DM p-value matrix (for heatmap)
    dm_mat = dm_pvalue_matrix(y_true, predictions)
    dm_mat.to_parquet(results_dir / "dm_pvalues.parquet")
    logger.info("Saved dm_pvalues.parquet")

    # Model Confidence Set
    mcs = model_confidence_set(y_true, predictions, alpha=args.mcs_alpha)
    print(f"\nModel Confidence Set (alpha={args.mcs_alpha}):")
    print(f"  Surviving: {mcs['surviving_models']}")
    print(f"  Eliminated: {mcs['eliminated']}")
    print(f"  Method: {mcs['method']}")

    # Mincer-Zarnowitz
    mz = mincer_zarnowitz_all(y_true, predictions)
    print("\nMincer-Zarnowitz regression:")
    print(mz.to_string())
    mz.to_parquet(results_dir / "mincer_zarnowitz.parquet")

    # ── 3. Trading Simulation ───────────────────────────────────────
    logger.info("=== Trading Simulation ===")
    trading_metrics, sims = compare_strategies(
        y_true, predictions,
        strategy=args.strategy,
        transaction_cost=args.transaction_cost,
    )
    print("\nTrading performance:")
    print(trading_metrics.to_string())
    trading_metrics.to_parquet(results_dir / "trading_metrics.parquet")

    # Save equity curves
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_equity_curves(sims, top_n=5)
    fig.savefig(results_dir / "equity_curves.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(results_dir / "equity_curves.png", bbox_inches="tight", dpi=150)
    logger.info("Saved equity curve plots")

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("=== Summary ===")
    logger.info("Best model by RMSE: %s (%.6f)", best_model, metrics_df.loc[best_model, "rmse"])
    logger.info("MCS surviving models: %s", mcs["surviving_models"])
    best_trader = trading_metrics["sharpe_ratio"].idxmax()
    logger.info("Best Sharpe: %s (%.3f)", best_trader, trading_metrics.loc[best_trader, "sharpe_ratio"])
    logger.info("All results saved to %s", results_dir)


if __name__ == "__main__":
    main()
