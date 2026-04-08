# src/models/statistical/models.py

import numpy as np
from scipy import stats
from arch import arch_model
from src.models.base import BaseModel


class HistoricalSimulation(BaseModel):
    """Historical Simulation VaR and CVaR."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._train_returns = y
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        q = 1 - self.quantile
        return np.array([np.quantile(self._train_returns, q)] * len(X))

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        q = 1 - self.quantile
        var = np.quantile(self._train_returns, q)
        cvar = self._train_returns[self._train_returns <= var].mean()
        return np.array([cvar] * len(X))


class ParametricVaR(BaseModel):
    """
    Gaussian or Student-t parametric VaR/CVaR.
    dist: 'normal' or 'student-t'
    """

    def __init__(self, quantile: float = 0.95, dist: str = "normal"):
        super().__init__(quantile)
        self.dist = dist
        self._mu = None
        self._sigma = None
        self._df = None  # for student-t only

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mu = np.mean(y)
        self._sigma = np.std(y)
        if self.dist == "student-t":
            self._df, _, _ = stats.t.fit(y)
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        q = 1 - self.quantile
        if self.dist == "normal":
            var = self._mu + self._sigma * stats.norm.ppf(q)
        else:
            var = self._mu + self._sigma * stats.t.ppf(q, df=self._df)
        return np.full(len(X), var)

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        q = 1 - self.quantile
        if self.dist == "normal":
            cvar = self._mu - self._sigma * (
                stats.norm.pdf(stats.norm.ppf(q)) / q
            )
        else:
            # Analytical CVaR for Student-t
            t_ppf = stats.t.ppf(q, df=self._df)
            cvar = self._mu - self._sigma * (
                stats.t.pdf(t_ppf, df=self._df) * (self._df + t_ppf**2) /
                ((self._df - 1) * q)
            )
        return np.full(len(X), cvar)


class GARCHModel(BaseModel):
    """
    GARCH(1,1) for VaR/CVaR.
    Uses arch library.
    """

    def __init__(self, quantile: float = 0.95, p: int = 1, q: int = 1):
        super().__init__(quantile)
        self.p = p
        self.q = q
        self._result = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        returns_pct = y * 100   # arch expects pct returns
        am = arch_model(returns_pct, vol="Garch", p=self.p, q=self.q, dist="Normal")
        self._result = am.fit(disp="off")
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        fcast = self._result.forecast(horizon=len(X), reindex=False)
        sigma = np.sqrt(fcast.variance.values[-1]) / 100
        mu = 0.0
        q = 1 - self.quantile
        return mu + sigma * stats.norm.ppf(q)

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        fcast = self._result.forecast(horizon=len(X), reindex=False)
        sigma = np.sqrt(fcast.variance.values[-1]) / 100
        q = 1 - self.quantile
        return -(sigma * stats.norm.pdf(stats.norm.ppf(q)) / q)


class CAViaR(BaseModel):
    """
    CAViaR (Conditional Autoregressive VaR) — quantile regression.
    Simplified Symmetric Absolute Value specification.
    CAViaR: VaR_t = beta0 + beta1 * VaR_{t-1} + beta2 * |r_{t-1}|
    Optimized via quantile loss minimization.
    """

    def __init__(self, quantile: float = 0.95, n_iter: int = 1000):
        super().__init__(quantile)
        self.n_iter = n_iter
        self._beta = None

    def _quantile_loss(self, beta: np.ndarray, returns: np.ndarray) -> float:
        q = 1 - self.quantile
        n = len(returns)
        var = np.zeros(n)
        var[0] = np.quantile(returns, q)
        for t in range(1, n):
            var[t] = beta[0] + beta[1] * var[t - 1] + beta[2] * abs(returns[t - 1])
        resid = returns - var
        loss = np.where(resid < 0, q * resid, (q - 1) * resid)
        return np.mean(loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from scipy.optimize import minimize
        beta0 = np.array([0.01, 0.8, 0.1])
        result = minimize(
            self._quantile_loss,
            beta0,
            args=(y,),
            method="Nelder-Mead",
            options={"maxiter": self.n_iter},
        )
        self._beta = result.x
        self._last_var = np.quantile(y, 1 - self.quantile)
        self._last_ret = y[-1]
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        b = self._beta
        preds = []
        var_t = self._last_var
        ret_t = self._last_ret
        for _ in range(len(X)):
            var_t = b[0] + b[1] * var_t + b[2] * abs(ret_t)
            preds.append(var_t)
            ret_t = 0.0   # no future return known; use 0 for multi-step
        return np.array(preds)
