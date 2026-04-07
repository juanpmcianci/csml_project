"""Exogenous data fetchers: FRED macro, Google Trends, news sentiment, polls."""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


# =====================================================================
# FRED  (macro: VIX, S&P 500, 10Y yield, CPI, NFP, unemployment)
# =====================================================================

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs for key macro indicators
FRED_SERIES = {
    "vix": "VIXCLS",
    "sp500": "SP500",
    "treasury_10y": "DGS10",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "fed_funds": "FEDFUNDS",
}


def fetch_fred(
    series_id: str,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch a single FRED series. Returns DataFrame indexed by date."""
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED_API_KEY not set. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").rename(columns={"value": series_id})
    return df


def fetch_all_fred(
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch all FRED macro series and join on date."""
    frames = []
    for name, sid in FRED_SERIES.items():
        try:
            df = fetch_fred(sid, start, end, api_key)
            frames.append(df)
            logger.info("FRED %s (%s): %d obs", name, sid, len(df))
        except Exception as exc:
            logger.warning("Failed to fetch FRED %s: %s", name, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# =====================================================================
# Google Trends
# =====================================================================

def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "2023-01-01 2025-12-31",
    geo: str = "US",
) -> pd.DataFrame:
    """Fetch Google Trends interest-over-time for a list of keywords."""
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)

    # Google Trends allows max 5 keywords per request
    frames = []
    for i in range(0, len(keywords), 5):
        batch = keywords[i : i + 5]
        pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    # Deduplicate columns if overlapping batches
    combined = combined.loc[:, ~combined.columns.duplicated()]
    logger.info("Google Trends: %d rows, keywords=%s", len(combined), keywords)
    return combined


# =====================================================================
# News sentiment  (NewsAPI + VADER)
# =====================================================================

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def fetch_news_sentiment(
    query: str,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    api_key: str | None = None,
    page_size: int = 100,
    max_pages: int = 5,
) -> pd.DataFrame:
    """Fetch headlines from NewsAPI and score with VADER.

    Returns daily aggregates: headline count, mean/std sentiment.
    """
    key = api_key or os.environ.get("NEWSAPI_KEY")
    if not key:
        raise ValueError("NEWSAPI_KEY not set")

    analyzer = SentimentIntensityAnalyzer()
    articles: list[dict] = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": start,
            "to": end,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": key,
        }
        resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning("NewsAPI page %d error: %s", page, resp.text[:200])
            break

        data = resp.json()
        batch = data.get("articles", [])
        if not batch:
            break
        articles.extend(batch)

    if not articles:
        logger.warning("No articles found for query '%s'", query)
        return pd.DataFrame()

    rows = []
    for a in articles:
        title = a.get("title") or ""
        scores = analyzer.polarity_scores(title)
        rows.append(
            {
                "date": pd.to_datetime(a.get("publishedAt", "")).date(),
                "compound": scores["compound"],
            }
        )

    df = pd.DataFrame(rows)
    daily = (
        df.groupby("date")
        .agg(
            headline_count=("compound", "count"),
            sentiment_mean=("compound", "mean"),
            sentiment_std=("compound", "std"),
        )
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    logger.info(
        "News sentiment for '%s': %d articles, %d days",
        query,
        len(articles),
        len(daily),
    )
    return daily


# =====================================================================
# Polling data  (FiveThirtyEight public CSVs)
# =====================================================================

FIVETHIRTYEIGHT_POLLS_URL = (
    "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
)


def fetch_polling(
    url: str = FIVETHIRTYEIGHT_POLLS_URL,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Fetch FiveThirtyEight polling data and compute daily averages."""
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        logger.warning("Could not fetch polling CSV: %s", exc)
        return pd.DataFrame()

    # Normalize date column
    date_col = None
    for col in ["end_date", "enddate", "created_at"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        logger.warning("No date column found in polling data")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    # Attempt to extract candidate-level polling averages
    if "pct" in df.columns and "candidate_name" in df.columns:
        pivot = (
            df.groupby(["date", "candidate_name"])["pct"]
            .mean()
            .unstack(fill_value=None)
        )
        pivot.columns = [f"poll_{c}" for c in pivot.columns]
        logger.info("Polling data: %d days, %d candidates", len(pivot), len(pivot.columns))
        return pivot

    logger.warning("Polling data columns not as expected: %s", list(df.columns)[:10])
    return pd.DataFrame()


# =====================================================================
# Master fetch  — pull all exogenous sources, save to data/external/
# =====================================================================

def fetch_all_exogenous(
    output_dir: str | Path = "data/external",
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    trend_keywords: list[str] | None = None,
    news_queries: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch all available exogenous data sources and save as parquets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    # FRED
    try:
        fred_df = fetch_all_fred(start, end)
        if not fred_df.empty:
            fred_df.to_parquet(output_dir / "fred.parquet")
            results["fred"] = fred_df
    except Exception as exc:
        logger.warning("FRED fetch failed: %s", exc)

    # Google Trends
    if trend_keywords:
        try:
            trends_df = fetch_google_trends(
                trend_keywords, timeframe=f"{start} {end}"
            )
            if not trends_df.empty:
                trends_df.to_parquet(output_dir / "google_trends.parquet")
                results["google_trends"] = trends_df
        except Exception as exc:
            logger.warning("Google Trends fetch failed: %s", exc)

    # News sentiment
    if news_queries:
        for query in news_queries:
            try:
                news_df = fetch_news_sentiment(query, start, end)
                if not news_df.empty:
                    safe_name = query.replace(" ", "_").lower()[:40]
                    news_df.to_parquet(output_dir / f"news_{safe_name}.parquet")
                    results[f"news_{safe_name}"] = news_df
            except Exception as exc:
                logger.warning("News sentiment failed for '%s': %s", query, exc)

    # Polling
    try:
        polls_df = fetch_polling(start=start, end=end)
        if not polls_df.empty:
            polls_df.to_parquet(output_dir / "polling.parquet")
            results["polling"] = polls_df
    except Exception as exc:
        logger.warning("Polling fetch failed: %s", exc)

    logger.info(
        "Exogenous data collection done. Sources retrieved: %s",
        list(results.keys()),
    )
    return results
