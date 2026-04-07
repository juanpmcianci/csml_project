"""Polymarket data ingestion via CLOB and Gamma APIs."""

import asyncio
import json
import logging
from pathlib import Path

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"


class PolymarketClient:
    """Fetches market metadata and price history from Polymarket."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/polymarket",
        interim_dir: str | Path = "data/interim",
        min_volume: float = 100_000,
        max_concurrent: int = 5,
        fidelity: int = 60,
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.min_volume = min_volume
        self.fidelity = fidelity
        self._sem = asyncio.Semaphore(max_concurrent)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    # ── Market discovery (Gamma API) ──────────────────────────────

    async def fetch_all_markets(
        self, session: aiohttp.ClientSession
    ) -> list[dict]:
        """Paginate through Gamma /markets endpoint."""
        markets: list[dict] = []
        offset = 0
        limit = 100

        while True:
            async with session.get(
                f"{GAMMA_BASE}/markets",
                params={"limit": limit, "offset": offset},
            ) as resp:
                resp.raise_for_status()
                batch = await resp.json()

            if not batch:
                break
            markets.extend(batch)
            offset += limit
            logger.info("Discovered %d markets so far", len(markets))

        return markets

    def filter_markets(self, markets: list[dict]) -> list[dict]:
        """Keep only markets above the minimum volume threshold."""
        out = []
        for m in markets:
            try:
                vol = float(m.get("volumeNum") or 0)
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

    # ── Price history (CLOB API) ──────────────────────────────────

    async def _fetch_history(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
    ) -> list[dict]:
        """Fetch price history for one token with retries."""
        params = {
            "market": token_id,
            "interval": "max",
            "fidelity": self.fidelity,
        }
        async with self._sem:
            for attempt in range(3):
                try:
                    async with session.get(
                        f"{CLOB_BASE}/prices-history", params=params
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get("history", [])
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    wait = 2**attempt
                    logger.warning(
                        "Token %s attempt %d failed: %s — retrying in %ds",
                        token_id[:12],
                        attempt + 1,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)

        logger.error("Failed to fetch history for token %s", token_id[:12])
        return []

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_field(raw) -> list:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return []
        return raw if isinstance(raw, list) else []

    @staticmethod
    def _extract_metadata(market: dict) -> dict:
        return {
            "condition_id": market.get("conditionId"),
            "question": market.get("question"),
            "category": market.get("category"),
            "slug": market.get("slug"),
            "end_date": market.get("endDate"),
            "active": market.get("active"),
            "closed": market.get("closed"),
            "volume": float(market.get("volumeNum") or 0),
            "liquidity": float(market.get("liquidityNum") or 0),
        }

    @staticmethod
    def _to_dataframe(
        history: list[dict],
        token_id: str,
        outcome: str,
        meta: dict,
    ) -> pd.DataFrame:
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df.columns = ["timestamp", "price"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp").sort_index()
        df["token_id"] = token_id
        df["outcome"] = outcome
        for k, v in meta.items():
            df[k] = v
        return df

    # ── Main pipeline ─────────────────────────────────────────────

    async def run(self) -> pd.DataFrame:
        """Discover markets, fetch histories, save raw + interim parquets."""
        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1 — discover & filter
            raw_markets = await self.fetch_all_markets(session)
            markets = self.filter_markets(raw_markets)

            # cache raw metadata
            with open(self.raw_dir / "markets.json", "w") as f:
                json.dump(markets, f, indent=2)

            # 2 — fetch price history per contract
            all_frames: list[pd.DataFrame] = []

            for market in markets:
                meta = self._extract_metadata(market)
                token_ids = self._parse_json_field(market.get("clobTokenIds"))
                outcomes = self._parse_json_field(market.get("outcomes"))

                if not token_ids:
                    continue

                contract_frames: list[pd.DataFrame] = []
                tasks = []
                for tid, outcome in zip(token_ids, outcomes):
                    tasks.append(
                        self._fetch_and_convert(session, tid, outcome, meta)
                    )

                results = await asyncio.gather(*tasks)
                for df in results:
                    if not df.empty:
                        contract_frames.append(df)

                if contract_frames:
                    contract_df = pd.concat(contract_frames)
                    path = self.interim_dir / f"polymarket_{meta['condition_id']}.parquet"
                    contract_df.to_parquet(path)
                    all_frames.append(contract_df)

                logger.info(
                    "Done: %s (%.0f rows)",
                    (meta["question"] or "")[:60],
                    sum(len(f) for f in contract_frames),
                )

        if not all_frames:
            logger.warning("No data collected")
            return pd.DataFrame()

        combined = pd.concat(all_frames)
        logger.info(
            "Polymarket ingestion complete: %d rows, %d contracts",
            len(combined),
            combined["condition_id"].nunique(),
        )
        return combined

    async def _fetch_and_convert(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
        outcome: str,
        meta: dict,
    ) -> pd.DataFrame:
        history = await self._fetch_history(session, token_id)
        # cache raw
        with open(self.raw_dir / f"{token_id}.json", "w") as f:
            json.dump(history, f)
        return self._to_dataframe(history, token_id, outcome, meta)


# ── Convenience entry point ───────────────────────────────────────

def ingest_polymarket(
    raw_dir: str = "data/raw/polymarket",
    interim_dir: str = "data/interim",
    min_volume: float = 100_000,
) -> pd.DataFrame:
    """Synchronous wrapper around the async pipeline."""
    client = PolymarketClient(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        min_volume=min_volume,
    )
    return asyncio.run(client.run())
