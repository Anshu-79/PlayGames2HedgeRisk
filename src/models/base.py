# src/models/base.py

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """All VaR/CVaR models implement this interface."""

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model on (X, y)."""
        pass

    @abstractmethod
    def predict_var(self, X: np.ndarray) -> np.ndarray:
        """Return VaR forecasts."""
        pass

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        """
        Return CVaR forecasts.
        Override in models that predict CVaR natively.
        Default: raise NotImplementedError (use two-stage approach).
        """
        raise NotImplementedError(f"{self.__class__.__name__} has no native CVaR.")

    def predict(self, X: np.ndarray) -> dict:
        """Return dict with var and (optionally) cvar."""
        out = {"var": self.predict_var(X)}
        try:
            out["cvar"] = self.predict_cvar(X)
        except NotImplementedError:
            out["cvar"] = None
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(q={self.quantile})"


class BaseRLAgent(ABC):
    """RL agents implement this interface."""

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile

    @abstractmethod
    def train(self, env) -> None:
        pass

    @abstractmethod
    def evaluate(self, env) -> dict:
        """Return metrics dict."""
        pass

    @abstractmethod
    def predict_var(self, obs: np.ndarray) -> float:
        pass

    def predict_cvar(self, obs: np.ndarray) -> float:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
