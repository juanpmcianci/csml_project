"""LSTM / GRU models in PyTorch Lightning."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# =====================================================================
# Dataset
# =====================================================================

class SequenceDataset(Dataset):
    """Sliding-window dataset: (X_seq, y_next)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 14):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_next = self.y[idx + self.seq_len]
        return x_seq, y_next


# =====================================================================
# Lightning Module
# =====================================================================

class LSTMModule(pl.LightningModule):
    """LSTM/GRU regressor."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        cell_type: str = "lstm",
    ):
        super().__init__()
        self.save_hyperparameters()

        RNNClass = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = RNNClass(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.functional.mse_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


# =====================================================================
# Wrapper with unified interface
# =====================================================================

class LSTMModel:
    """LSTM model with sklearn-like fit/predict interface."""

    def __init__(
        self,
        seq_len: int = 14,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        cell_type: str = "lstm",
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.cell_type = cell_type

        self.module: LSTMModule | None = None
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = self._to_numpy(X_train)
        y_train = self._to_numpy(y_train)

        # Normalize using training stats only
        self._train_mean = np.nanmean(X_train, axis=0)
        self._train_std = np.nanstd(X_train, axis=0)
        self._train_std[self._train_std == 0] = 1.0
        X_train = (X_train - self._train_mean) / self._train_std

        n_features = X_train.shape[1]
        train_ds = SequenceDataset(X_train, y_train, self.seq_len)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        self.module = LSTMModule(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
            cell_type=self.cell_type,
        )

        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, mode="min"),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
        ]

        val_dl = None
        if X_val is not None and y_val is not None:
            X_val = self._to_numpy(X_val)
            y_val = self._to_numpy(y_val)
            X_val = (X_val - self._train_mean) / self._train_std
            val_ds = SequenceDataset(X_val, y_val, self.seq_len)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(self.module, train_dl, val_dl)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test = self._to_numpy(X_test)
        X_test = (X_test - self._train_mean) / self._train_std

        self.module.eval()
        preds = []
        with torch.no_grad():
            for i in range(len(X_test) - self.seq_len):
                x = torch.tensor(
                    X_test[i : i + self.seq_len], dtype=torch.float32
                ).unsqueeze(0)
                pred = self.module(x).item()
                preds.append(pred)

        # Pad beginning with NaN to match input length
        return np.array([np.nan] * self.seq_len + preds)

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), path.with_suffix(".pt"))
        path.write_bytes(pickle.dumps({
            "seq_len": self.seq_len, "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers, "dropout": self.dropout,
            "cell_type": self.cell_type,
            "train_mean": self._train_mean, "train_std": self._train_std,
        }))

    def load(self, path):
        path = Path(path)
        data = pickle.loads(path.read_bytes())
        self.seq_len = data["seq_len"]
        self.hidden_dim = data["hidden_dim"]
        self.n_layers = data["n_layers"]
        self.dropout = data["dropout"]
        self.cell_type = data["cell_type"]
        self._train_mean = data["train_mean"]
        self._train_std = data["train_std"]

        n_features = len(self._train_mean)
        self.module = LSTMModule(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
        )
        self.module.load_state_dict(torch.load(path.with_suffix(".pt"), weights_only=True))
        return self

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, pd.DataFrame):
            return x.values.astype(np.float32)
        if isinstance(x, pd.Series):
            return x.values.astype(np.float32)
        return np.asarray(x, dtype=np.float32)
