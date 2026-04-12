# src/models/ml/models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from src.models.base import BaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SVRModel(BaseModel):
    """Support Vector Regression for VaR."""

    def __init__(self, quantile: float = 0.95, kernel: str = "rbf"):
        super().__init__(quantile)
        self.kernel = kernel
        self._scaler = StandardScaler()
        self._model = SVR(kernel=kernel, C=1.0, epsilon=0.01)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s = self._scaler.fit_transform(X)
        # Target: indicator whether return <= VaR threshold (regression proxy)
        q = 1 - self.quantile
        var_target = np.quantile(y, q) * np.ones(len(y))
        self._model.fit(X_s, var_target)
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X)
        return self._model.predict(X_s)


class QuantileGradientBoosting(BaseModel):
    """
    QGB: XGBoost with quantile objective for VaR.
    Two-stage: XGBoost VaR -> XGBoost tail mean for CVaR.
    """

    def __init__(self, quantile: float = 0.95, n_estimators: int = 200):
        super().__init__(quantile)
        self.n_estimators = n_estimators
        alpha = 1 - quantile
        self._var_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            tree_method="hist",
            # device='cuda'
        )
        self._cvar_model = XGBRegressor(
            n_estimators=n_estimators,
            objective="reg:squarederror",
            tree_method="hist",
            # device='cuda'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._var_model.fit(X, y)

        # CVaR: train on tail samples only
        var_preds = self._var_model.predict(X)
        X_cvar = np.concatenate([X, var_preds.reshape(-1, 1)], axis=1)

        tail_mask = y <= var_preds

        if tail_mask.mean() > 0.01:
            self._cvar_model.fit(X_cvar[tail_mask], y[tail_mask])
        else:
            self._cvar_model.fit(X_cvar, y)

        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        return self._var_model.predict(X)

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        var_preds = self._var_model.predict(X)
        X_cvar = np.concatenate([X, var_preds.reshape(-1, 1)], axis=1)

        cvar_preds = self._cvar_model.predict(X_cvar)

        return np.minimum(cvar_preds, var_preds)


class GARCHSVRModel(BaseModel):
    """
    GARCH-SVR hybrid.
    GARCH filters volatility -> SVR predicts VaR from GARCH residuals + features.
    """

    def __init__(self, quantile: float = 0.95):
        super().__init__(quantile)
        from arch import arch_model as _arch

        self._arch = _arch
        self._garch_result = None
        self._svr = SVR(kernel="rbf")
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Step 1: fit GARCH on returns
        am = self._arch(y * 100, vol="Garch", p=1, q=1)
        self._garch_result = am.fit(disp="off")
        garch_vol = np.sqrt(self._garch_result.conditional_volatility) / 100

        # Step 2: stack garch vol with X features
        X_aug = np.hstack([X, garch_vol.reshape(-1, 1)])
        X_s = self._scaler.fit_transform(X_aug)

        q = 1 - self.quantile
        var_target = np.full(len(y), np.quantile(y, q))
        self._svr.fit(X_s, var_target)
        self._last_vol = garch_vol[-1]
        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        vol = np.full((len(X), 1), self._last_vol)
        X_aug = np.hstack([X, vol])
        X_s = self._scaler.transform(X_aug)
        return self._svr.predict(X_s)


class MixtureDensityNetwork(BaseModel):
    """
    Mixture Density Network (MDN) — probabilistic model.
    Outputs mixture of Gaussians; VaR/CVaR computed using Monte Carlo simulations
    """

    def __init__(
        self,
        quantile: float = 0.95,
        n_components: int = 5,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        super().__init__(quantile)
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self._net = None
        self._scaler = StandardScaler()

    def _build_net(self, input_dim: int):
        import torch.nn as nn

        K = self.n_components
        self._net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3 * K),
        ).to(DEVICE)

    def _mdn_loss(self, pi, mu, sigma, y):
        import torch.distributions as D

        mix = D.Categorical(pi)
        comp = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, comp)
        return -gmm.log_prob(y).mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch.nn.functional as F

        X_s = self._scaler.fit_transform(X).astype(np.float32)
        y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        X_t = torch.tensor(X_s).to(DEVICE)

        self._build_net(X_s.shape[1])
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        K = self.n_components
        for epoch in range(self.epochs):
            self._net.train()
            out = self._net(X_t)
            pi_raw, mu, log_sigma = out[:, :K], out[:, K : 2 * K], out[:, 2 * K :]
            pi = F.softmax(pi_raw, dim=-1)
            sigma = torch.exp(log_sigma).clamp(min=1e-6)
            loss = self._mdn_loss(pi, mu, sigma, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._last_X = X_t[-1:]
        self.is_fitted = True

    def _sample_var_cvar(self, X_t):
        import torch.nn.functional as F

        K = self.n_components
        self._net.eval()
        with torch.no_grad():
            out = self._net(X_t)
        pi_raw, mu, log_sigma = out[:, :K], out[:, K : 2 * K], out[:, 2 * K :]
        pi = F.softmax(pi_raw, dim=-1).cpu().numpy()
        mu = mu.cpu().numpy()
        sigma = torch.exp(log_sigma).cpu().numpy()

        # Monte Carlo: sample from GMM
        N = 100_000
        results = []
        for i in range(len(X_t)):
            comp_idx = np.random.choice(K, size=N, p=pi[i])
            samples = np.random.normal(mu[i][comp_idx], sigma[i][comp_idx])
            results.append(samples)
        return np.array(results)

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_s).to(DEVICE)
        samples = self._sample_var_cvar(X_t)
        q = 1 - self.quantile
        return np.quantile(samples, q, axis=1)

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X_s).to(DEVICE)
        samples = self._sample_var_cvar(X_t)
        q = 1 - self.quantile
        var = np.quantile(samples, q, axis=1)
        cvar = np.array(
            [samples[i][samples[i] <= var[i]].mean() for i in range(len(var))]
        )
        return cvar
