"""Model registry: factory pattern for experiment dispatch."""

import importlib

# Lazy registry: (module_path, class_name) — avoids importing torch at module load
_REGISTRY_SPEC = {
    "persistence": ("src.models.baselines", "PersistenceModel"),
    "historical_mean": ("src.models.baselines", "HistoricalMeanModel"),
    "ridge": ("src.models.baselines", "RidgeModel"),
    "lasso": ("src.models.baselines", "LassoModel"),
    "lgbm": ("src.models.trees", "LightGBMModel"),
    "xgboost": ("src.models.trees", "XGBoostModel"),
    "lstm": ("src.models.lstm", "LSTMModel"),
    "tft": ("src.models.tft", "TFTModel"),
}


def _resolve(name: str):
    module_path, class_name = _REGISTRY_SPEC[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_model(name: str, **kwargs):
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        Key from the registry.
    **kwargs
        Passed to the model constructor.

    Returns
    -------
    Model instance with .fit(), .predict(), .get_feature_importance(),
    .save(), .load() methods.
    """
    if name not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY_SPEC.keys())}"
        )
    cls = _resolve(name)
    return cls(**kwargs)


def list_models() -> list[str]:
    """Return available model names."""
    return list(_REGISTRY_SPEC.keys())
