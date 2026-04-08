"""Baseline models: persistence, historical mean, Ridge, Lasso."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler


class PersistenceModel:
    """Naive persistence: predict y(t+1) = y(t), i.e. zero return."""

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X_test) -> np.ndarray:
        return np.zeros(len(X_test))

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(dtype=float)

    def save(self, path):
        pass

    def load(self, path):
        return self


class HistoricalMeanModel:
    """Predict y(t+1) = rolling mean of recent targets."""

    def __init__(self, window: int = 14):
        self.window = window
        self._mean = 0.0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._mean = np.nanmean(y_train[-self.window:])
        return self

    def predict(self, X_test) -> np.ndarray:
        return np.full(len(X_test), self._mean)

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(dtype=float)

    def save(self, path):
        Path(path).write_bytes(pickle.dumps({"window": self.window, "mean": self._mean}))

    def load(self, path):
        data = pickle.loads(Path(path).read_bytes())
        self.window = data["window"]
        self._mean = data["mean"]
        return self


class RidgeModel:
    """Ridge regression with feature scaling."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self._feature_names: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train, self._feature_names = self._to_array(X_train)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test, _ = self._to_array(X_test)
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        coefs = pd.Series(
            np.abs(self.model.coef_),
            index=self._feature_names or range(len(self.model.coef_)),
        )
        return coefs.sort_values(ascending=False)

    def save(self, path):
        Path(path).write_bytes(pickle.dumps({
            "model": self.model, "scaler": self.scaler,
            "feature_names": self._feature_names,
        }))

    def load(self, path):
        data = pickle.loads(Path(path).read_bytes())
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._feature_names = data["feature_names"]
        return self

    @staticmethod
    def _to_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        return np.asarray(X), []


class LassoModel:
    """Lasso regression with feature scaling."""

    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Lasso(alpha=alpha, max_iter=5000)
        self._feature_names: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train, self._feature_names = self._to_array(X_train)
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test, _ = self._to_array(X_test)
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        coefs = pd.Series(
            np.abs(self.model.coef_),
            index=self._feature_names or range(len(self.model.coef_)),
        )
        return coefs.sort_values(ascending=False)

    def save(self, path):
        Path(path).write_bytes(pickle.dumps({
            "model": self.model, "scaler": self.scaler,
            "feature_names": self._feature_names,
        }))

    def load(self, path):
        data = pickle.loads(Path(path).read_bytes())
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._feature_names = data["feature_names"]
        return self

    @staticmethod
    def _to_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        return np.asarray(X), []
