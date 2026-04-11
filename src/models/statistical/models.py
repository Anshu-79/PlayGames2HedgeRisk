# src/models/statistical/models.py

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from arch import arch_model
from src.models.base import BaseModel
from numba import jit


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
        if self.dist == "normal":
            self._mu = np.mean(y)
            self._sigma = np.std(y)
        else:
            # df = degrees of freedom (nu)
            # loc = center (mu)
            # scale = dispersion (gamma)
            self._df, self._mu, self._sigma = stats.t.fit(y)
        
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
            cvar = self._mu - self._sigma * (stats.norm.pdf(stats.norm.ppf(q)) / q)
        else:
            # Analytical CVaR for Student-t
            t_ppf = stats.t.ppf(q, df=self._df)
            cvar = self._mu - self._sigma * (
                stats.t.pdf(t_ppf, df=self._df)
                * (self._df + t_ppf**2)
                / ((self._df - 1) * q)
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
        returns_pct = y * 100  # arch expects pct returns
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


@jit(nopython=True)
def caviar_quantile_loss(beta, returns, q, initial_var):
    """
    C-level code for computing pinball loss for CAViaR model
    """

    n = len(returns)
    var = np.zeros(n)
    var[0] = initial_var

    # Recursive VaR (linear in past return, not abs)
    for t in range(1, n):
        var[t] = beta[0] + beta[1] * var[t - 1] + beta[2] * returns[t - 1]

    # Correct pinball loss
    loss = 0.0
    for i in range(n):
        resid = returns[i] - var[i]

        if resid < 0:
            loss += (q - 1.0) * resid
        else:
            loss += q * resid

    return loss / n


@jit(nopython=True)
def compute_fitted_series(beta, returns, initial_var):
    n = len(returns)
    var = np.zeros(n)
    var[0] = initial_var

    for t in range(1, n):
        var[t] = beta[0] + beta[1] * var[t - 1] + beta[2] * returns[t - 1]

    return var


class CAViaR(BaseModel):
    """
    CAViaR (Conditional Autoregressive VaR)

    VaR_t = beta0 + beta1 * VaR_{t-1} + beta2 * r_{t-1}
    """

    def __init__(self, quantile: float = 0.95, n_iter: int = 1000):
        super().__init__(quantile)
        self.n_iter = n_iter
        self._beta = None
        self._cvar_slope = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        q = 1 - self.quantile

        # Initial VaR (should already be negative for left tail)
        initial_var = np.quantile(y, q)

        beta0 = np.array([initial_var, 0.95, -0.05])

        # Enforce correct sign behavior
        bounds = [
            (-1.0, 0.0),  # beta0 (negative)
            (0.0, 0.999),  # persistence
            (-1.0, 0.0),  # response to returns (negative)
        ]

        result = minimize(
            caviar_quantile_loss,
            x0=beta0,
            args=(y, q, initial_var),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.n_iter},
        )

        self._beta = result.x

        fitted_var = compute_fitted_series(self._beta, y, initial_var)

        self._last_var = fitted_var[-1]
        self._last_ret = y[-1]

        # --- CVaR slope ---
        tail_mask = y <= fitted_var
        if np.any(tail_mask) and np.mean(fitted_var[tail_mask]) != 0:
            self._cvar_slope = np.mean(y[tail_mask]) / np.mean(fitted_var[tail_mask])
        else:
            self._cvar_slope = 1.1

        self.is_fitted = True

    def _compute_var_series(self, returns, beta, initial_var):
        n = len(returns)
        var = np.zeros(n)
        var[0] = initial_var
        for t in range(1, n):
            var[t] = beta[0] + beta[1] * var[t - 1] + beta[2] * returns[t - 1]
        return var

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        b = self._beta
        v_next = b[0] + b[1] * self._last_var + b[2] * self._last_ret
        return np.full(len(X), v_next)

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        var_pred = self.predict_var(X)
        return var_pred * self._cvar_slope
