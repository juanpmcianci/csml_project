"""Temporal Fusion Transformer wrapper using pytorch-forecasting."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class TFTModel:
    """Temporal Fusion Transformer with unified fit/predict interface.

    Uses pytorch-forecasting's TemporalFusionTransformer under the hood.
    """

    def __init__(
        self,
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        hidden_size: int = 32,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        known_reals: list[str] | None = None,
        unknown_reals: list[str] | None = None,
        target: str = "target_logret",
        time_idx: str = "time_idx",
        group_id: str = "group_id",
    ):
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.known_reals = known_reals or ["time_to_exp"]
        self.unknown_reals = unknown_reals or []
        self.target = target
        self.time_idx = time_idx
        self.group_id = group_id

        self.model = None
        self.trainer = None
        self._training_dataset = None

    def _prepare_dataset(self, df: pd.DataFrame, is_train: bool = True):
        """Convert a flat DataFrame into a pytorch-forecasting TimeSeriesDataSet."""
        from pytorch_forecasting import TimeSeriesDataSet

        # Ensure required columns
        if self.time_idx not in df.columns:
            df = df.copy()
            df[self.time_idx] = np.arange(len(df))
        if self.group_id not in df.columns:
            df = df.copy()
            df[self.group_id] = "0"

        # Auto-detect unknown reals if not specified
        if not self.unknown_reals:
            exclude = {self.target, self.time_idx, self.group_id} | set(self.known_reals)
            self.unknown_reals = [
                c for c in df.select_dtypes(include="number").columns
                if c not in exclude
            ]

        # Fill NaN for TFT compatibility
        df = df.fillna(0.0)

        kwargs = dict(
            data=df,
            time_idx=self.time_idx,
            target=self.target,
            group_ids=[self.group_id],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=[r for r in self.known_reals if r in df.columns],
            time_varying_unknown_reals=[r for r in self.unknown_reals if r in df.columns],
            allow_missing_timesteps=True,
        )

        if is_train:
            dataset = TimeSeriesDataSet(**kwargs)
            self._training_dataset = dataset
        else:
            dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, df, stop_randomization=True)

        return dataset

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import MAE

        # X_train should be a DataFrame with features + target
        if isinstance(X_train, pd.DataFrame) and self.target not in X_train.columns:
            df_train = X_train.copy()
            df_train[self.target] = y_train
        elif isinstance(X_train, pd.DataFrame):
            df_train = X_train
        else:
            raise ValueError("TFTModel.fit expects a DataFrame for X_train")

        train_ds = self._prepare_dataset(df_train, is_train=True)
        train_dl = train_ds.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)

        val_dl = None
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame) and self.target not in X_val.columns:
                df_val = X_val.copy()
                df_val[self.target] = y_val
            else:
                df_val = X_val
            val_ds = self._prepare_dataset(df_val, is_train=False)
            val_dl = val_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        self.model = TemporalFusionTransformer.from_dataset(
            train_ds,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            learning_rate=self.learning_rate,
            loss=MAE(),
            log_interval=0,
            reduce_on_plateau_patience=5,
        )

        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, mode="min"),
        ]

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks if val_dl else [],
            gradient_clip_val=0.1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        self.trainer.fit(self.model, train_dl, val_dl)
        return self

    def predict(self, X_test) -> np.ndarray:
        if isinstance(X_test, pd.DataFrame) and self.target not in X_test.columns:
            df_test = X_test.copy()
            df_test[self.target] = 0.0  # placeholder
        else:
            df_test = X_test

        test_ds = self._prepare_dataset(df_test, is_train=False)
        test_dl = test_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        preds = self.model.predict(test_dl)
        return preds.numpy().flatten()

    def get_feature_importance(self) -> pd.Series:
        """Extract variable selection weights from the TFT."""
        if self.model is None:
            return pd.Series(dtype=float)
        try:
            interpretation = self.model.interpret_output(
                self.model.predict(
                    self._training_dataset.to_dataloader(train=False, batch_size=128, num_workers=0),
                    return_x=True,
                ),
                reduction="mean",
            )
            encoder_imp = interpretation.get("encoder_variables", {})
            if isinstance(encoder_imp, dict):
                return pd.Series(encoder_imp).sort_values(ascending=False)
        except Exception as e:
            logger.warning("Could not extract TFT feature importance: %s", e)
        return pd.Series(dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.with_suffix(".pt"))
        path.write_bytes(pickle.dumps({
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "dropout": self.dropout,
            "hidden_continuous_size": self.hidden_continuous_size,
            "max_encoder_length": self.max_encoder_length,
            "known_reals": self.known_reals,
            "unknown_reals": self.unknown_reals,
            "target": self.target,
        }))

    def load(self, path):
        path = Path(path)
        data = pickle.loads(path.read_bytes())
        for k, v in data.items():
            setattr(self, k, v)
        logger.info("TFT config loaded from %s — call fit() with data to rebuild model", path)
        return self
