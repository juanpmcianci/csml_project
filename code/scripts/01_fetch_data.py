"""End-to-end data ingestion: Polymarket, Kalshi, exogenous sources."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.polymarket import ingest_polymarket
from src.data.kalshi import ingest_kalshi
from src.data.exogenous import fetch_all_exogenous

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch all data sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["polymarket", "kalshi", "exogenous"],
        choices=["polymarket", "kalshi", "exogenous"],
        help="Which sources to fetch",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=100_000,
        help="Minimum contract volume in USD",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if "polymarket" in args.sources:
        logger.info("=== Polymarket ingestion ===")
        poly_df = ingest_polymarket(
            raw_dir=data_dir / "raw" / "polymarket",
            interim_dir=data_dir / "interim",
            min_volume=args.min_volume,
        )
        logger.info("Polymarket: %d rows", len(poly_df))

    if "kalshi" in args.sources:
        logger.info("=== Kalshi ingestion ===")
        kalshi_df = ingest_kalshi(
            raw_dir=data_dir / "raw" / "kalshi",
            interim_dir=data_dir / "interim",
            min_volume=args.min_volume,
        )
        logger.info("Kalshi: %d rows", len(kalshi_df))

    if "exogenous" in args.sources:
        logger.info("=== Exogenous data ===")
        exog = fetch_all_exogenous(
            output_dir=data_dir / "external",
            trend_keywords=["prediction market", "Polymarket", "Kalshi"],
            news_queries=["prediction market", "election betting"],
        )
        logger.info("Exogenous sources fetched: %s", list(exog.keys()))

    logger.info("Done.")


if __name__ == "__main__":
    main()
