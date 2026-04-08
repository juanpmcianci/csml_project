"""Generate all paper figures and tables from saved results."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.plots import (
    plot_price_trajectories,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_ablation,
    plot_feature_importance,
    plot_equity_curves,
    plot_dm_heatmap,
)
from src.visualization.tables import (
    descriptive_stats_table,
    forecast_accuracy_table,
    feature_importance_table,
    trading_summary_table,
    export_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with saved results")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")
    args = parser.parse_args()

    results = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)

    logger.info("Generating figures to %s", out)

    # Load results if they exist
    metrics_path = results / "metrics.parquet"
    if metrics_path.exists():
        metrics_df = pd.read_parquet(metrics_path)
        logger.info("Loaded metrics: %s", metrics_df.index.tolist())

        # Fig 3 — Model comparison
        plot_model_comparison(metrics_df, savepath=out / "fig3_model_comparison.pdf")
        logger.info("Saved fig3")

        # Table 2 — Forecast accuracy
        latex = forecast_accuracy_table(metrics_df)
        (out / "tables" / "tab2_forecast_accuracy.tex").write_text(latex)
        logger.info("Saved table 2")

    trading_path = results / "trading_metrics.parquet"
    if trading_path.exists():
        trading_df = pd.read_parquet(trading_path)

        # Table 4 — Trading summary
        latex = trading_summary_table(trading_df)
        (out / "tables" / "tab4_trading_sim.tex").write_text(latex)
        logger.info("Saved table 4")

    dm_path = results / "dm_pvalues.parquet"
    if dm_path.exists():
        dm_mat = pd.read_parquet(dm_path)
        plot_dm_heatmap(dm_mat, savepath=out / "fig8_dm_heatmap.pdf")
        logger.info("Saved fig8")

    logger.info("Done. Check %s for outputs.", out)


if __name__ == "__main__":
    main()
