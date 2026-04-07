"""Endogenous feature engineering: price, volume, microstructure, cross-contract."""

import numpy as np
import pandas as pd


# =====================================================================
# Price-based features
# =====================================================================

def add_returns(df: pd.DataFrame, price_col: str = "price", windows: list[int] = [1, 4, 24, 72, 168]) -> pd.DataFrame:
    """Add simple and log returns at multiple horizons (in periods)."""
    p = df[price_col]
    for w in windows:
        df[f"ret_{w}"] = p.pct_change(w)
        df[f"logret_{w}"] = np.log(p / p.shift(w))
    return df


def add_moving_averages(df: pd.DataFrame, price_col: str = "price",
                        sma_windows: list[int] = [7, 14],
                        ema_windows: list[int] = [7, 14]) -> pd.DataFrame:
    """Add SMA, EMA, and price-relative-to-MA features."""
    p = df[price_col]
    for w in sma_windows:
        sma = p.rolling(w).mean()
        df[f"sma_{w}"] = sma
        df[f"price_rel_sma_{w}"] = p / sma - 1
    for w in ema_windows:
        df[f"ema_{w}"] = p.ewm(span=w, adjust=False).mean()
    return df


def add_bollinger(df: pd.DataFrame, price_col: str = "price", window: int = 14, n_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band position: (price - lower) / (upper - lower)."""
    p = df[price_col]
    sma = p.rolling(window).mean()
    std = p.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    df["bb_position"] = (p - lower) / (upper - lower).replace(0, np.nan)
    return df


def add_rsi(df: pd.DataFrame, price_col: str = "price", period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


def add_macd(df: pd.DataFrame, price_col: str = "price",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram."""
    p = df[price_col]
    ema_fast = p.ewm(span=fast, adjust=False).mean()
    ema_slow = p.ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_roc(df: pd.DataFrame, price_col: str = "price", windows: list[int] = [1, 3, 7]) -> pd.DataFrame:
    """Add Rate of Change at multiple windows."""
    p = df[price_col]
    for w in windows:
        df[f"roc_{w}"] = p.pct_change(w)
    return df


# =====================================================================
# Volume-based features
# =====================================================================

def add_volume_features(df: pd.DataFrame, vol_col: str = "volume",
                        price_col: str = "price", ma_window: int = 7,
                        abnormal_window: int = 14, abnormal_sigma: float = 2.0) -> pd.DataFrame:
    """Add volume-derived features."""
    if vol_col not in df.columns:
        return df

    v = df[vol_col]

    df["log_volume"] = np.log1p(v)

    # Volume MA ratio
    vol_ma = v.rolling(ma_window).mean()
    df["vol_ma_ratio"] = v / vol_ma.replace(0, np.nan)

    # On-balance volume
    direction = np.sign(df[price_col].diff()).fillna(0)
    df["obv"] = (v * direction).cumsum()

    # Abnormal volume flag
    vol_mean = v.rolling(abnormal_window).mean()
    vol_std = v.rolling(abnormal_window).std()
    df["abnormal_volume"] = (v > vol_mean + abnormal_sigma * vol_std).astype(int)

    return df


# =====================================================================
# Microstructure features
# =====================================================================

def add_spread_features(df: pd.DataFrame, bid_col: str = "yes_bid",
                        ask_col: str = "yes_ask") -> pd.DataFrame:
    """Add bid-ask spread and spread MA ratio."""
    if bid_col not in df.columns or ask_col not in df.columns:
        return df

    df["spread"] = df[ask_col] - df[bid_col]
    spread_ma = df["spread"].rolling(7).mean()
    df["spread_ma_ratio"] = df["spread"] / spread_ma.replace(0, np.nan)
    return df


def add_amihud(df: pd.DataFrame, price_col: str = "price",
               vol_col: str = "volume", window: int = 14) -> pd.DataFrame:
    """Add Amihud illiquidity ratio (rolling)."""
    if vol_col not in df.columns:
        return df

    abs_ret = df[price_col].pct_change().abs()
    dollar_vol = df[vol_col].replace(0, np.nan)
    ratio = abs_ret / dollar_vol
    df["amihud"] = ratio.rolling(window).mean()
    return df


def add_kyle_lambda(df: pd.DataFrame, price_col: str = "price",
                    vol_col: str = "volume", window: int = 14) -> pd.DataFrame:
    """Estimate Kyle's lambda: rolling regression of price change on signed volume."""
    if vol_col not in df.columns:
        return df

    dp = df[price_col].diff()
    signed_vol = np.sign(dp) * df[vol_col]

    lambdas = []
    for i in range(len(df)):
        if i < window:
            lambdas.append(np.nan)
            continue
        y = dp.iloc[i - window:i].values
        x = signed_vol.iloc[i - window:i].values
        mask = np.isfinite(y) & np.isfinite(x) & (x != 0)
        if mask.sum() < 3:
            lambdas.append(np.nan)
            continue
        # Simple OLS: lambda = cov(dp, sv) / var(sv)
        cov = np.cov(y[mask], x[mask])
        lam = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan
        lambdas.append(lam)

    df["kyle_lambda"] = lambdas
    return df


def add_time_features(df: pd.DataFrame, expiration: str | None = None) -> pd.DataFrame:
    """Add time-to-expiration and calendar features."""
    idx = df.index

    if expiration is not None:
        exp_dt = pd.Timestamp(expiration, tz=idx.tz)
        tte = (exp_dt - idx).total_seconds() / 86400
        df["time_to_exp"] = np.maximum(tte, 0)
        df["time_to_exp_sq"] = df["time_to_exp"] ** 2

    if hasattr(idx, "hour"):
        df["hour"] = idx.hour
    if hasattr(idx, "dayofweek"):
        df["dayofweek"] = idx.dayofweek

    return df


# =====================================================================
# Cross-contract features
# =====================================================================

def add_yes_no_spread(df_yes: pd.DataFrame, df_no: pd.DataFrame,
                      price_col: str = "price") -> pd.DataFrame:
    """Spread between Yes/No implied probabilities (should sum to ~1)."""
    merged = df_yes[[price_col]].rename(columns={price_col: "price_yes"})
    merged = merged.join(
        df_no[[price_col]].rename(columns={price_col: "price_no"}),
        how="outer",
    ).ffill()
    merged["prob_sum"] = merged["price_yes"] + merged["price_no"]
    merged["prob_deviation"] = merged["prob_sum"] - 1.0
    return merged


def add_cross_platform_spread(df_poly: pd.DataFrame, df_kalshi: pd.DataFrame,
                              price_col: str = "price") -> pd.DataFrame:
    """Polymarket price - Kalshi price for matched contracts."""
    merged = df_poly[[price_col]].rename(columns={price_col: "price_poly"})
    merged = merged.join(
        df_kalshi[[price_col]].rename(columns={price_col: "price_kalshi"}),
        how="outer",
    ).ffill()
    merged["xplatform_spread"] = merged["price_poly"] - merged["price_kalshi"]
    return merged


# =====================================================================
# Master function
# =====================================================================

def build_endogenous_features(
    df: pd.DataFrame,
    price_col: str = "price",
    vol_col: str = "volume",
    return_windows: list[int] = [1, 4, 24, 72, 168],
    sma_windows: list[int] = [7, 14],
    ema_windows: list[int] = [7, 14],
    rsi_period: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    roc_windows: list[int] = [1, 3, 7],
    expiration: str | None = None,
) -> pd.DataFrame:
    """Apply all endogenous feature engineering to a single contract DataFrame."""
    df = df.copy()
    df = add_returns(df, price_col, return_windows)
    df = add_moving_averages(df, price_col, sma_windows, ema_windows)
    df = add_bollinger(df, price_col)
    df = add_rsi(df, price_col, rsi_period)
    df = add_macd(df, price_col, *macd_params)
    df = add_roc(df, price_col, roc_windows)
    df = add_volume_features(df, vol_col, price_col)
    df = add_spread_features(df)
    df = add_amihud(df, price_col, vol_col)
    df = add_kyle_lambda(df, price_col, vol_col)
    df = add_time_features(df, expiration)
    return df
