"""Unified feature matrix constructor: joins endogenous + exogenous, adds targets."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.endogenous import build_endogenous_features
from src.features.exogenous import build_exogenous_features

logger = logging.getLogger(__name__)


# =====================================================================
# Target variable construction
# =====================================================================

def add_targets(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Add target columns for regression and classification."""
    p = df[price_col]

    # y1: next-period log return (regression)
    df["target_logret"] = np.log(p.shift(-1) / p)

    # y2: next-period direction (classification: -1, 0, +1)
    ret = p.shift(-1) / p - 1
    df["target_direction"] = np.sign(ret)

    # y3: next-period price level (regression)
    df["target_price"] = p.shift(-1)

    return df


# =====================================================================
# Multicollinearity checks
# =====================================================================

def flag_high_correlation(df: pd.DataFrame, threshold: float = 0.90) -> list[tuple[str, str, float]]:
    """Find feature pairs with |correlation| above threshold."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr().abs()

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if val >= threshold:
                pairs.append((corr.columns[i], corr.columns[j], val))

    if pairs:
        logger.info("Found %d feature pairs with |corr| >= %.2f", len(pairs), threshold)
    return pairs


def compute_vif(df: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
    """Compute Variance Inflation Factors for numeric features.

    Only computes on the first `max_features` columns to avoid
    excessive runtime on wide feature matrices.
    """
    numeric = df.select_dtypes(include="number").dropna()

    if numeric.shape[1] > max_features:
        logger.warning(
            "VIF computation limited to first %d features (of %d)",
            max_features,
            numeric.shape[1],
        )
        numeric = numeric.iloc[:, :max_features]

    if numeric.shape[0] < numeric.shape[1]:
        logger.warning("More features than observations — skipping VIF")
        return pd.DataFrame()

    vifs = []
    cols = numeric.columns.tolist()
    X = numeric.values

    for i in range(len(cols)):
        try:
            vif_val = variance_inflation_factor(X, i)
        except Exception:
            vif_val = np.nan
        vifs.append({"feature": cols[i], "vif": vif_val})

    result = pd.DataFrame(vifs).sort_values("vif", ascending=False)
    high = result[result["vif"] > 10]
    if len(high) > 0:
        logger.info("%d features with VIF > 10", len(high))
    return result


# =====================================================================
# Feature metadata
# =====================================================================

def generate_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a metadata table describing each feature."""
    target_cols = [c for c in df.columns if c.startswith("target_")]
    id_cols = ["condition_id", "ticker", "token_id", "outcome",
               "question", "category", "slug", "event_ticker",
               "status", "result"]

    meta = []
    for col in df.columns:
        if col in id_cols:
            group = "identifier"
        elif col in target_cols:
            group = "target"
        elif col.startswith(("ret_", "logret_", "sma_", "ema_", "price_rel",
                             "bb_", "rsi", "macd", "roc_")):
            group = "endogenous/price"
        elif col.startswith(("vol_", "log_vol", "obv", "abnormal_vol")):
            group = "endogenous/volume"
        elif col.startswith(("spread", "amihud", "kyle", "time_to")):
            group = "endogenous/microstructure"
        elif col.startswith(("hour", "dayof")):
            group = "endogenous/calendar"
        elif col.startswith("trend_"):
            group = "exogenous/trends"
        elif col.startswith("news_"):
            group = "exogenous/sentiment"
        elif col.startswith("macro_"):
            group = "exogenous/macro"
        elif col.startswith("poll"):
            group = "exogenous/polling"
        elif col.startswith(("cpi_", "unemployment", "fed_funds")):
            group = "exogenous/economic"
        elif col.startswith(("prob_", "xplatform")):
            group = "endogenous/cross-contract"
        else:
            group = "other"

        meta.append({
            "feature": col,
            "group": group,
            "dtype": str(df[col].dtype),
            "null_pct": df[col].isna().mean(),
            "nunique": df[col].nunique(),
        })

    return pd.DataFrame(meta)


# =====================================================================
# Master builder
# =====================================================================

class FeatureMatrixBuilder:
    """Builds the unified feature matrix for a set of contracts."""

    def __init__(
        self,
        price_col: str = "price",
        vol_col: str = "volume",
        return_windows: list[int] = [1, 4, 24, 72, 168],
        sma_windows: list[int] = [7, 14],
        ema_windows: list[int] = [7, 14],
        rsi_period: int = 14,
        macd_params: tuple[int, int, int] = (12, 26, 9),
        roc_windows: list[int] = [1, 3, 7],
        exog_lag: int = 1,
        max_abs_corr: float = 0.90,
        max_vif: float = 10.0,
    ):
        self.price_col = price_col
        self.vol_col = vol_col
        self.return_windows = return_windows
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.rsi_period = rsi_period
        self.macd_params = macd_params
        self.roc_windows = roc_windows
        self.exog_lag = exog_lag
        self.max_abs_corr = max_abs_corr
        self.max_vif = max_vif

    def build(
        self,
        df: pd.DataFrame,
        category: str | None = None,
        expiration: str | None = None,
        trends: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
        fred: pd.DataFrame | None = None,
        polling: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the full feature matrix for a single contract."""
        logger.info("Building features (%d rows, category=%s)", len(df), category)

        # Endogenous
        out = build_endogenous_features(
            df,
            price_col=self.price_col,
            vol_col=self.vol_col,
            return_windows=self.return_windows,
            sma_windows=self.sma_windows,
            ema_windows=self.ema_windows,
            rsi_period=self.rsi_period,
            macd_params=self.macd_params,
            roc_windows=self.roc_windows,
            expiration=expiration,
        )

        # Exogenous
        out = build_exogenous_features(
            out,
            category=category,
            price_col=self.price_col,
            trends=trends,
            sentiment=sentiment,
            fred=fred,
            polling=polling,
            lag=self.exog_lag,
        )

        # Targets
        out = add_targets(out, price_col=self.price_col)

        n_features = out.select_dtypes(include="number").shape[1]
        logger.info("Feature matrix: %d rows x %d numeric features", len(out), n_features)
        return out

    def build_and_save(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        **kwargs,
    ) -> pd.DataFrame:
        """Build features and save as parquet."""
        result = self.build(df, **kwargs)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path)
        logger.info("Saved feature matrix to %s", output_path)
        return result

    def diagnostics(self, df: pd.DataFrame) -> dict:
        """Run multicollinearity diagnostics on a feature matrix."""
        high_corr = flag_high_correlation(df, self.max_abs_corr)
        vif = compute_vif(df)
        meta = generate_feature_metadata(df)

        return {
            "high_correlation_pairs": high_corr,
            "vif": vif,
            "metadata": meta,
        }
