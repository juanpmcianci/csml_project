"""Preprocessing: temporal alignment, filtering, cleaning, train/test split."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize as _winsorize

logger = logging.getLogger(__name__)


# =====================================================================
# 1.  Temporal alignment
# =====================================================================

def resample_to_grid(
    df: pd.DataFrame,
    freq: str = "1D",
    price_col: str = "price",
) -> pd.DataFrame:
    """Resample a time-indexed DataFrame to a uniform grid.

    Forward-fills prices (markets don't trade continuously) and
    sums volume where available.
    """
    numeric = df.select_dtypes(include="number")
    non_numeric = df.select_dtypes(exclude="number")

    # Price-like columns: forward-fill
    price_cols = [c for c in numeric.columns if "price" in c.lower() or c in (
        price_col, "open", "high", "low", "close", "mean",
    )]
    vol_cols = [c for c in numeric.columns if "volume" in c.lower()]
    other_num = [c for c in numeric.columns if c not in price_cols + vol_cols]

    agg: dict = {}
    for c in price_cols:
        agg[c] = "last"
    for c in vol_cols:
        agg[c] = "sum"
    for c in other_num:
        agg[c] = "last"

    resampled = numeric.resample(freq).agg(agg)
    resampled[price_cols] = resampled[price_cols].ffill()

    # Carry forward categorical columns
    for c in non_numeric.columns:
        resampled[c] = non_numeric[c].resample(freq).last().ffill()

    return resampled


def align_exogenous(
    price_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """Merge exogenous features onto price data with a lag.

    Shifts exogenous data forward by `lag` periods so that features at
    time t only use information available before time t.
    """
    exog_shifted = exog_df.shift(lag)
    merged = price_df.join(exog_shifted, how="left")
    return merged


# =====================================================================
# 2.  Filtering
# =====================================================================

def filter_contracts(
    df: pd.DataFrame,
    contract_col: str = "condition_id",
    price_col: str = "price",
    min_trading_days: int = 30,
    max_missing_pct: float = 0.20,
    trivial_threshold: float = 0.95,
    trivial_pct: float = 0.80,
) -> pd.DataFrame:
    """Remove contracts that are too short, too sparse, or trivially resolved."""
    keep = []
    contracts = df.groupby(contract_col)

    for cid, group in contracts:
        # (a) minimum trading days
        n_days = (group.index.max() - group.index.min()).days
        if n_days < min_trading_days:
            logger.debug("Dropping %s: only %d days", cid, n_days)
            continue

        # (b) too many missing observations
        total_slots = len(
            pd.date_range(group.index.min(), group.index.max(), freq="D")
        )
        if total_slots > 0:
            missing = 1 - len(group) / total_slots
            if missing > max_missing_pct:
                logger.debug("Dropping %s: %.0f%% missing", cid, missing * 100)
                continue

        # (c) trivially resolved (stuck near 0 or 1)
        if price_col in group.columns:
            prices = group[price_col].dropna()
            if len(prices) > 0:
                trivial_frac = (
                    (prices >= trivial_threshold) | (prices <= 1 - trivial_threshold)
                ).mean()
                if trivial_frac >= trivial_pct:
                    logger.debug("Dropping %s: trivially resolved (%.0f%%)", cid, trivial_frac * 100)
                    continue

        keep.append(cid)

    filtered = df[df[contract_col].isin(keep)]
    logger.info(
        "Filtering: kept %d / %d contracts",
        len(keep),
        contracts.ngroups,
    )
    return filtered


# =====================================================================
# 3.  Cleaning
# =====================================================================

def compute_returns(
    df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    """Add log_return and simple_return columns."""
    prices = df[price_col]
    df = df.copy()
    df["simple_return"] = prices.pct_change()
    df["log_return"] = np.log(prices / prices.shift(1))
    return df


def winsorize_returns(
    df: pd.DataFrame,
    return_col: str = "log_return",
    limits: tuple[float, float] = (0.01, 0.01),
) -> pd.DataFrame:
    """Clip extreme returns at the given percentiles."""
    df = df.copy()
    mask = df[return_col].notna()
    df.loc[mask, return_col] = _winsorize(
        df.loc[mask, return_col].values, limits=limits
    )
    return df


def flag_wash_trades(
    df: pd.DataFrame,
    price_col: str = "price",
    vol_col: str = "volume",
    window: int = 5,
    zero_change_threshold: int = 4,
) -> pd.DataFrame:
    """Flag suspected wash trading: repeated buy-sell at same price.

    Marks rows where the price is unchanged for >= `zero_change_threshold`
    of the last `window` periods while volume remains positive.
    """
    df = df.copy()
    price_unchanged = (df[price_col].diff().abs() < 1e-8).astype(int)
    unchanged_count = price_unchanged.rolling(window, min_periods=1).sum()

    has_volume = True
    if vol_col in df.columns:
        has_volume = df[vol_col] > 0

    df["wash_flag"] = (unchanged_count >= zero_change_threshold) & has_volume
    n_flagged = df["wash_flag"].sum()
    if n_flagged > 0:
        logger.info("Flagged %d potential wash-trade rows", n_flagged)
    return df


# =====================================================================
# 4.  Train / test split
# =====================================================================

def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2025-06-30",
    test_start: str = "2025-07-01",
    test_end: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date. Assumes a datetime index."""
    train = df.loc[:train_end]
    test = df.loc[test_start:test_end]
    logger.info("Temporal split: train=%d rows, test=%d rows", len(train), len(test))
    return train, test


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 1,
    min_train_periods: int = 60,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate expanding-window walk-forward train/val splits.

    Each fold expands the training set and validates on the next month.
    `gap` periods are skipped between train and val to avoid leakage.
    """
    dates = df.index.normalize().unique().sort_values()
    n = len(dates)
    fold_size = (n - min_train_periods) // n_splits

    if fold_size < 1:
        raise ValueError(
            f"Not enough data ({n} dates) for {n_splits} folds "
            f"with min_train={min_train_periods}"
        )

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        val_end_idx = min_train_periods + (i + 1) * fold_size
        val_start_idx = min_train_periods + i * fold_size + gap
        train_end_idx = min_train_periods + i * fold_size

        if val_end_idx > n:
            break

        train_end_date = dates[train_end_idx - 1]
        val_start_date = dates[val_start_idx]
        val_end_date = dates[val_end_idx - 1]

        train = df.loc[:train_end_date]
        val = df.loc[val_start_date:val_end_date]

        if len(train) > 0 and len(val) > 0:
            splits.append((train, val))
            logger.debug(
                "Fold %d: train up to %s (%d), val %s–%s (%d)",
                i + 1,
                train_end_date.date(),
                len(train),
                val_start_date.date(),
                val_end_date.date(),
                len(val),
            )

    logger.info("Walk-forward: generated %d splits", len(splits))
    return splits


# =====================================================================
# Master preprocessing pipeline
# =====================================================================

def preprocess(
    df: pd.DataFrame,
    contract_col: str = "condition_id",
    price_col: str = "price",
    freq: str = "1D",
    min_trading_days: int = 30,
    max_missing_pct: float = 0.20,
    winsorize_limits: tuple[float, float] = (0.01, 0.01),
    exog_df: pd.DataFrame | None = None,
    exog_lag: int = 1,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on raw price data."""
    logger.info("Starting preprocessing (%d rows)", len(df))

    # Resample each contract to a uniform grid
    groups = []
    for cid, group in df.groupby(contract_col):
        resampled = resample_to_grid(group, freq=freq, price_col=price_col)
        resampled[contract_col] = cid
        groups.append(resampled)

    if not groups:
        return pd.DataFrame()

    df = pd.concat(groups)

    # Filter
    df = filter_contracts(
        df,
        contract_col=contract_col,
        price_col=price_col,
        min_trading_days=min_trading_days,
        max_missing_pct=max_missing_pct,
    )

    # Returns + cleaning
    df = compute_returns(df, price_col=price_col)
    df = winsorize_returns(df, limits=winsorize_limits)
    df = flag_wash_trades(df, price_col=price_col)

    # Merge exogenous
    if exog_df is not None:
        df = align_exogenous(df, exog_df, lag=exog_lag)

    logger.info("Preprocessing complete: %d rows", len(df))
    return df
