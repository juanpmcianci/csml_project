"""Tree ensemble models: LightGBM, XGBoost."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


class LightGBMModel:
    """LightGBM regressor with Optuna-friendly interface."""

    def __init__(self, **params):
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        defaults.update(params)
        self.params = defaults
        self.model = lgb.LGBMRegressor(**defaults)
        self._feature_names: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train, self._feature_names = self._to_array(X_train)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_val, _ = self._to_array(X_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test, _ = self._to_array(X_test)
        return self.model.predict(X_test)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type=importance_type)
        names = self._feature_names or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path):
        path = Path(path)
        self.model.booster_.save_model(str(path.with_suffix(".lgb")))
        path.write_bytes(pickle.dumps({"params": self.params, "feature_names": self._feature_names}))

    def load(self, path):
        path = Path(path)
        data = pickle.loads(path.read_bytes())
        self.params = data["params"]
        self._feature_names = data["feature_names"]
        booster = lgb.Booster(model_file=str(path.with_suffix(".lgb")))
        self.model = lgb.LGBMRegressor(**self.params)
        self.model._Booster = booster
        return self

    @staticmethod
    def _to_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        return np.asarray(X), []


class XGBoostModel:
    """XGBoost regressor with Optuna-friendly interface."""

    def __init__(self, **params):
        import xgboost as xgb

        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
        defaults.update(params)
        self.params = defaults
        self.model = xgb.XGBRegressor(**defaults)
        self._feature_names: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train, self._feature_names = self._to_array(X_train)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            X_val, _ = self._to_array(X_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X_test) -> np.ndarray:
        X_test, _ = self._to_array(X_test)
        return self.model.predict(X_test)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        imp = self.model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False)

    def save(self, path):
        import xgboost as xgb
        path = Path(path)
        self.model.save_model(str(path.with_suffix(".xgb")))
        path.write_bytes(pickle.dumps({"params": self.params, "feature_names": self._feature_names}))

    def load(self, path):
        import xgboost as xgb
        path = Path(path)
        data = pickle.loads(path.read_bytes())
        self.params = data["params"]
        self._feature_names = data["feature_names"]
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(str(path.with_suffix(".xgb")))
        return self

    @staticmethod
    def _to_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        return np.asarray(X), []
