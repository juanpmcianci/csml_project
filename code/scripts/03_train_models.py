"""Train all models on processed feature matrices."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import get_model, list_models
from src.data.preprocessing import temporal_split
from src.tuning.optuna_search import tune_model, OBJECTIVE_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "target_logret"

# Columns that are not features
EXCLUDE_COLS = {
    TARGET, "target_direction", "target_price",
    "condition_id", "ticker", "token_id", "outcome", "question",
    "category", "slug", "event_ticker", "status", "result",
    "end_date", "active", "closed", "wash_flag",
    "group_id", "time_idx",
}


def load_feature_matrix(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all processed parquets."""
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets in {data_dir}")

    frames = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(frames).sort_index()
    logger.info("Loaded %d rows from %d files", len(df), len(parquets))
    return df


def prepare_Xy(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target y, dropping NaNs."""
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c not in EXCLUDE_COLS]
    subset = df[feature_cols + [TARGET]].dropna()
    X = subset[feature_cols]
    y = subset[TARGET].values
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--models", nargs="+",
        default=["persistence", "historical_mean", "ridge", "lasso", "lgbm", "xgboost"],
        help="Models to train",
    )
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--tune-trials", type=int, default=None, help="Override trial count")
    parser.add_argument("--train-end", type=str, default="2025-06-30")
    parser.add_argument("--test-start", type=str, default="2025-07-01")
    parser.add_argument("--test-end", type=str, default="2025-12-31")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    # Load data
    df = load_feature_matrix(data_dir)

    # Temporal split
    train_df, test_df = temporal_split(df, args.train_end, args.test_start, args.test_end)
    X_train, y_train, feature_cols = prepare_Xy(train_df)
    X_test, y_test, _ = prepare_Xy(test_df)

    logger.info("Train: %d rows, %d features", len(X_train), len(feature_cols))
    logger.info("Test:  %d rows", len(X_test))

    # Optionally split a validation set from training data (last 15%)
    val_split = int(len(X_train) * 0.85)
    X_tr = X_train.iloc[:val_split]
    y_tr = y_train[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train[val_split:]

    predictions = {}
    trained_models = {}

    for name in args.models:
        logger.info("=== %s ===", name)

        # Optuna tuning
        best_params = {}
        if args.tune and name in OBJECTIVE_REGISTRY:
            study = tune_model(
                name, X_train.values, y_train,
                n_trials=args.tune_trials,
                storage=f"sqlite:///{output_dir / 'optuna.db'}",
            )
            best_params = study.best_params
            logger.info("Best params for %s: %s", name, best_params)

        # Train
        model = get_model(name, **best_params)
        model.fit(X_tr, y_tr, X_val, y_val)

        # Predict on test
        preds = model.predict(X_test)
        predictions[name] = preds
        trained_models[name] = model

        # Save model
        try:
            model.save(str(output_dir / "models" / name))
            logger.info("Saved model: %s", name)
        except Exception as e:
            logger.warning("Could not save %s: %s", name, e)

        # Quick RMSE
        mask = np.isfinite(preds) & np.isfinite(y_test)
        if mask.sum() > 0:
            test_rmse = np.sqrt(np.mean((preds[mask] - y_test[mask]) ** 2))
            logger.info("%s test RMSE: %.6f", name, test_rmse)

    # Save predictions
    pred_df = pd.DataFrame(predictions, index=X_test.index)
    pred_df["y_true"] = y_test
    pred_df.to_parquet(output_dir / "predictions.parquet")
    logger.info("Saved predictions to %s", output_dir / "predictions.parquet")

    # Save feature importance where available
    importance_frames = {}
    for name, model in trained_models.items():
        imp = model.get_feature_importance()
        if len(imp) > 0:
            importance_frames[name] = imp

    if importance_frames:
        imp_df = pd.DataFrame(importance_frames)
        imp_df.to_parquet(output_dir / "feature_importances.parquet")
        logger.info("Saved feature importances")

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
