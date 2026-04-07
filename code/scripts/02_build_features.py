"""Build feature matrices from preprocessed data."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.builder import FeatureMatrixBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument(
        "--interim-dir", type=str, default="data/interim",
        help="Directory with preprocessed parquets",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for feature matrices",
    )
    parser.add_argument(
        "--external-dir", type=str, default="data/external",
        help="Directory with exogenous data",
    )
    args = parser.parse_args()

    interim = Path(args.interim_dir)
    output = Path(args.output_dir)
    external = Path(args.external_dir)
    output.mkdir(parents=True, exist_ok=True)

    builder = FeatureMatrixBuilder()

    # Load exogenous data if available
    trends = _load_parquet(external / "google_trends.parquet")
    fred = _load_parquet(external / "fred.parquet")
    polling = _load_parquet(external / "polling.parquet")

    # Process each contract parquet
    parquets = sorted(interim.glob("*.parquet"))
    logger.info("Found %d contract parquets in %s", len(parquets), interim)

    for path in parquets:
        logger.info("Processing %s", path.name)
        df = pd.read_parquet(path)

        if df.empty:
            continue

        category = df.get("category", pd.Series([None])).iloc[0]
        expiration = df.get("end_date", pd.Series([None])).iloc[0]

        # Look for sentiment data matching the contract
        sentiment = _find_sentiment(external, df)

        result = builder.build(
            df,
            category=category,
            expiration=expiration,
            trends=trends,
            sentiment=sentiment,
            fred=fred,
            polling=polling,
        )

        out_path = output / path.name
        result.to_parquet(out_path)
        logger.info("Saved: %s (%d rows, %d cols)", out_path.name, len(result), len(result.columns))

    logger.info("Done. Feature matrices saved to %s", output)


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _find_sentiment(external: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    """Try to find a matching news sentiment parquet."""
    sentiment_files = list(external.glob("news_*.parquet"))
    if sentiment_files:
        return pd.read_parquet(sentiment_files[0])
    return None


if __name__ == "__main__":
    main()
