"""Kalshi data ingestion via public REST API (v2)."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MIN_REQUEST_INTERVAL = 0.06  # ~17 req/s, staying under the 20/s basic-tier cap


class KalshiClient:
    """Fetches market metadata and candlestick history from Kalshi."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/kalshi",
        interim_dir: str | Path = "data/interim",
        min_volume: float = 100_000,
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.min_volume = min_volume
        self._last_request = 0.0

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    # ── Rate limiting ─────────────────────────────────────────────

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, params: dict | None = None) -> dict:
        self._throttle()
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Historical cutoff ─────────────────────────────────────────

    def get_cutoff(self) -> dict:
        """Return timestamps separating live from historical data."""
        data = self._get(f"{BASE_URL}/historical/cutoff")
        logger.info("Historical cutoff: %s", data)
        return data

    # ── Market discovery ──────────────────────────────────────────

    def _paginate_markets(self, endpoint: str, **extra) -> list[dict]:
        markets: list[dict] = []
        cursor = None

        while True:
            params = {"limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor

            data = self._get(endpoint, params=params)
            batch = data.get("markets", [])
            markets.extend(batch)

            cursor = data.get("cursor")
            if not cursor or not batch:
                break

            logger.info("Fetched %d Kalshi markets so far", len(markets))

        return markets

    def fetch_all_markets(self) -> list[dict]:
        """Fetch markets from both live and historical endpoints."""
        live = self._paginate_markets(f"{BASE_URL}/markets")
        historical = self._paginate_markets(f"{BASE_URL}/historical/markets")

        # deduplicate by ticker
        seen = set()
        combined = []
        for m in live + historical:
            ticker = m.get("ticker")
            if ticker and ticker not in seen:
                seen.add(ticker)
                combined.append(m)

        logger.info(
            "Discovered %d unique markets (live=%d, historical=%d)",
            len(combined),
            len(live),
            len(historical),
        )
        return combined

    def filter_markets(self, markets: list[dict]) -> list[dict]:
        """Keep markets above volume threshold."""
        out = []
        for m in markets:
            try:
                vol = float(m.get("volume_fp", 0))
            except (ValueError, TypeError):
                continue
            if vol >= self.min_volume:
                out.append(m)

        logger.info(
            "Kept %d / %d markets (volume >= $%s)",
            len(out),
            len(markets),
            f"{self.min_volume:,.0f}",
        )
        return out

    # ── Series / event metadata ───────────────────────────────────

    def _get_series_ticker(self, market: dict) -> str | None:
        """Derive series_ticker from the market ticker.

        Kalshi tickers follow the pattern SERIES-DATE or SERIES-PARAM.
        The series_ticker is everything before the last hyphen-delimited segment,
        but this is fragile. Prefer event_ticker -> event -> series lookup.
        """
        event_ticker = market.get("event_ticker")
        if event_ticker:
            try:
                event = self._get(f"{BASE_URL}/events/{event_ticker}")
                return event.get("event", {}).get("series_ticker")
            except requests.HTTPError:
                pass
        return None

    def fetch_series_metadata(self, series_ticker: str) -> dict:
        """Fetch series info (contains category)."""
        try:
            data = self._get(f"{BASE_URL}/series/{series_ticker}")
            return data.get("series", {})
        except requests.HTTPError:
            return {}

    # ── Candlestick history ───────────────────────────────────────

    def fetch_candlesticks(
        self,
        market: dict,
        start_ts: int,
        end_ts: int,
        period: int = 1440,
        series_ticker: str | None = None,
    ) -> list[dict]:
        """Fetch candlestick data, querying both live and historical endpoints."""
        ticker = market["ticker"]
        cutoff = self.get_cutoff()
        market_cutoff_ts = cutoff.get("market_settled_ts", 0)

        candles: list[dict] = []

        # Historical portion
        if start_ts < market_cutoff_ts:
            hist_end = min(end_ts, market_cutoff_ts)
            try:
                data = self._get(
                    f"{BASE_URL}/historical/markets/{ticker}/candlesticks",
                    params={
                        "start_ts": start_ts,
                        "end_ts": hist_end,
                        "period_interval": period,
                    },
                )
                candles.extend(data.get("candlesticks", []))
            except requests.HTTPError as exc:
                logger.warning("Historical candlesticks failed for %s: %s", ticker, exc)

        # Live portion
        if end_ts > market_cutoff_ts and series_ticker:
            live_start = max(start_ts, market_cutoff_ts)
            try:
                data = self._get(
                    f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks",
                    params={
                        "start_ts": live_start,
                        "end_ts": end_ts,
                        "period_interval": period,
                    },
                )
                candles.extend(data.get("candlesticks", []))
            except requests.HTTPError as exc:
                logger.warning("Live candlesticks failed for %s: %s", ticker, exc)

        return candles

    @staticmethod
    def _candles_to_dataframe(
        candles: list[dict], market: dict, category: str | None = None
    ) -> pd.DataFrame:
        """Convert raw candlestick dicts to a tidy DataFrame."""
        if not candles:
            return pd.DataFrame()

        rows = []
        for c in candles:
            price = c.get("price", {})
            rows.append(
                {
                    "timestamp": c.get("end_period_ts"),
                    "open": float(price.get("open_dollars", 0)),
                    "high": float(price.get("high_dollars", 0)),
                    "low": float(price.get("low_dollars", 0)),
                    "close": float(price.get("close_dollars", 0)),
                    "mean": float(price.get("mean_dollars", 0)),
                    "volume": float(c.get("volume_fp", 0)),
                    "open_interest": float(c.get("open_interest_fp", 0)),
                }
            )

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()

        df["ticker"] = market["ticker"]
        df["event_ticker"] = market.get("event_ticker")
        df["status"] = market.get("status")
        df["result"] = market.get("result")
        df["category"] = category
        return df

    # ── Main pipeline ─────────────────────────────────────────────

    def run(
        self,
        start_ts: int = 1672531200,   # 2023-01-01
        end_ts: int = 1767225600,     # 2025-12-31
        period: int = 1440,
    ) -> pd.DataFrame:
        """Discover markets, fetch candlestick histories, save parquets."""
        # 1 — discover & filter
        all_markets = self.fetch_all_markets()
        markets = self.filter_markets(all_markets)

        # cache raw metadata
        import json
        with open(self.raw_dir / "markets.json", "w") as f:
            json.dump(markets, f, indent=2, default=str)

        # 2 — fetch candlesticks per market
        all_frames: list[pd.DataFrame] = []
        series_cache: dict[str, dict] = {}

        for market in markets:
            ticker = market["ticker"]

            # resolve series for category + live endpoint
            series_ticker = self._get_series_ticker(market)
            category = None
            if series_ticker and series_ticker not in series_cache:
                series_cache[series_ticker] = self.fetch_series_metadata(
                    series_ticker
                )
            if series_ticker:
                category = series_cache.get(series_ticker, {}).get("category")

            candles = self.fetch_candlesticks(
                market,
                start_ts=start_ts,
                end_ts=end_ts,
                period=period,
                series_ticker=series_ticker,
            )

            df = self._candles_to_dataframe(candles, market, category)
            if not df.empty:
                path = self.interim_dir / f"kalshi_{ticker}.parquet"
                df.to_parquet(path)
                all_frames.append(df)

            logger.info("Done: %s (%d rows)", ticker, len(df))

        if not all_frames:
            logger.warning("No Kalshi data collected")
            return pd.DataFrame()

        combined = pd.concat(all_frames)
        logger.info(
            "Kalshi ingestion complete: %d rows, %d markets",
            len(combined),
            combined["ticker"].nunique(),
        )
        return combined


# ── Convenience entry point ───────────────────────────────────────

def ingest_kalshi(
    raw_dir: str = "data/raw/kalshi",
    interim_dir: str = "data/interim",
    min_volume: float = 100_000,
) -> pd.DataFrame:
    client = KalshiClient(
        raw_dir=raw_dir, interim_dir=interim_dir, min_volume=min_volume
    )
    return client.run()
