"""Exogenous feature engineering: polling, macro, search/social, category-specific."""

import numpy as np
import pandas as pd


# =====================================================================
# Universal features (all categories)
# =====================================================================

def add_trends_features(df: pd.DataFrame, trends: pd.DataFrame,
                        momentum_window: int = 7, lag: int = 1) -> pd.DataFrame:
    """Add Google Trends features: level + momentum, lagged."""
    if trends.empty:
        return df

    trends_shifted = trends.shift(lag)
    for col in trends_shifted.columns:
        df[f"trend_{col}"] = trends_shifted[col].reindex(df.index, method="ffill")
        df[f"trend_{col}_mom"] = trends_shifted[col].diff(momentum_window).reindex(df.index, method="ffill")
    return df


def add_sentiment_features(df: pd.DataFrame, sentiment: pd.DataFrame,
                           momentum_window: int = 3, lag: int = 1) -> pd.DataFrame:
    """Add news sentiment features: count, mean, std, momentum, lagged."""
    if sentiment.empty:
        return df

    sent = sentiment.shift(lag)

    for col in ["headline_count", "sentiment_mean", "sentiment_std"]:
        if col in sent.columns:
            df[f"news_{col}"] = sent[col].reindex(df.index, method="ffill")

    if "sentiment_mean" in sent.columns:
        df["news_sentiment_mom"] = (
            sent["sentiment_mean"].diff(momentum_window).reindex(df.index, method="ffill")
        )
    return df


def add_macro_features(df: pd.DataFrame, fred: pd.DataFrame,
                       lag: int = 1) -> pd.DataFrame:
    """Add FRED macro control features: levels and deltas, lagged."""
    if fred.empty:
        return df

    fred_shifted = fred.shift(lag)

    for col in fred_shifted.columns:
        series = fred_shifted[col].reindex(df.index, method="ffill")
        df[f"macro_{col}"] = series
        df[f"macro_{col}_delta"] = series.diff()

    return df


# =====================================================================
# Category-specific: Political
# =====================================================================

def add_polling_features(df: pd.DataFrame, polling: pd.DataFrame,
                         price_col: str = "price",
                         momentum_windows: list[int] = [3, 7],
                         lag: int = 1) -> pd.DataFrame:
    """Add polling features for political contracts."""
    if polling.empty:
        return df

    polled = polling.shift(lag)

    for col in polled.columns:
        series = polled[col].reindex(df.index, method="ffill")
        df[f"{col}"] = series

        # Polling momentum
        for w in momentum_windows:
            df[f"{col}_mom_{w}d"] = series.diff(w)

    # Polling-price divergence (use first polling column as proxy)
    first_poll = polled.columns[0]
    poll_vals = polled[first_poll].reindex(df.index, method="ffill") / 100.0
    if price_col in df.columns:
        df["poll_price_divergence"] = poll_vals - df[price_col]

    return df


# =====================================================================
# Category-specific: Economic
# =====================================================================

def add_economic_features(df: pd.DataFrame, fred: pd.DataFrame,
                          lag: int = 1) -> pd.DataFrame:
    """Add economic-specific features from FRED data.

    These go beyond the universal macro controls to include
    economic surprise proxies.
    """
    if fred.empty:
        return df

    shifted = fred.shift(lag)

    # CPI momentum (inflation trend)
    if "CPIAUCSL" in shifted.columns:
        cpi = shifted["CPIAUCSL"].reindex(df.index, method="ffill")
        df["cpi_mom_3m"] = cpi.pct_change(63)  # ~3 months of trading days

    # Unemployment rate change
    if "UNRATE" in shifted.columns:
        unemp = shifted["UNRATE"].reindex(df.index, method="ffill")
        df["unemployment_delta"] = unemp.diff()

    # Fed funds rate change
    if "FEDFUNDS" in shifted.columns:
        ff = shifted["FEDFUNDS"].reindex(df.index, method="ffill")
        df["fed_funds_delta"] = ff.diff()

    return df


# =====================================================================
# Master function
# =====================================================================

def build_exogenous_features(
    df: pd.DataFrame,
    category: str | None = None,
    price_col: str = "price",
    trends: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    fred: pd.DataFrame | None = None,
    polling: pd.DataFrame | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Apply exogenous feature engineering based on contract category.

    Universal features (trends, sentiment, macro controls) are always added.
    Category-specific features are added when `category` matches.
    """
    df = df.copy()

    # Universal
    if trends is not None:
        df = add_trends_features(df, trends, lag=lag)
    if sentiment is not None:
        df = add_sentiment_features(df, sentiment, lag=lag)
    if fred is not None:
        df = add_macro_features(df, fred, lag=lag)

    # Category-specific
    if category and category.lower() in ("political", "us-current-affairs", "politics"):
        if polling is not None:
            df = add_polling_features(df, polling, price_col=price_col, lag=lag)

    if category and category.lower() in ("economic", "economics", "finance"):
        if fred is not None:
            df = add_economic_features(df, fred, lag=lag)

    return df
