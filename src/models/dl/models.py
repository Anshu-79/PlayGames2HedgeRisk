# src/models/dl/models.py

import numpy as np
import torch
import torch.nn as nn
from src.models.base import BaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantile_loss(preds: torch.Tensor, targets: torch.Tensor, q: float) -> torch.Tensor:
    errors = targets - preds
    return torch.mean(torch.where(errors >= 0, q * errors, (q - 1) * errors))


def es_loss(
    var_pred: torch.Tensor, cvar_pred: torch.Tensor, targets: torch.Tensor, q: float
) -> torch.Tensor:
    """Expected Shortfall loss for joint VaR-CVaR training."""
    exceed = (targets < var_pred).float()
    es_term = exceed * (targets - cvar_pred) ** 2
    return torch.mean(es_term)


class _QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMQuantile(BaseModel):
    """Quantile LSTM for VaR."""

    def __init__(
        self,
        quantile=0.95,
        hidden_dim=64,
        num_layers=2,
        seq_len=20,
        epochs=50,
        lr=1e-3,
        dropout=0.1,
    ):
        super().__init__(quantile)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self._net = None

    def _make_sequences(self, X, y=None):
        xs = []
        ys = [] if y is not None else None
        for i in range(self.seq_len, len(X)):
            xs.append(X[i - self.seq_len : i])
            if y is not None:
                ys.append(y[i])
        Xt = torch.tensor(np.array(xs), dtype=torch.float32).to(DEVICE)
        yt = (
            torch.tensor(np.array(ys), dtype=torch.float32).to(DEVICE)
            if y is not None
            else None
        )
        return Xt, yt

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xt, yt = self._make_sequences(X, y)
        self._net = _QuantileLSTM(
            X.shape[1], self.hidden_dim, self.num_layers, self.dropout
        ).to(DEVICE)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        q = 1 - self.quantile

        for _ in range(self.epochs):
            self._net.train()
            pred = self._net(Xt)
            loss = quantile_loss(pred, yt, q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        Xt, _ = self._make_sequences(X)
        with torch.no_grad():
            return self._net(Xt).cpu().numpy()


class _AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (attn_w * out).sum(dim=1)  # (B, H)
        return self.head(context).squeeze(-1)


class AttentionLSTM(LSTMQuantile):
    """Attention-LSTM for VaR."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xt, yt = self._make_sequences(X, y)
        self._net = _AttentionLSTM(X.shape[1], self.hidden_dim, self.num_layers).to(
            DEVICE
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        q = 1 - self.quantile

        for _ in range(self.epochs):
            self._net.train()
            pred = self._net(Xt)
            loss = quantile_loss(pred, yt, q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.is_fitted = True


class _TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class TransformerEncoder(LSTMQuantile):
    """Transformer Encoder for VaR."""

    def __init__(
        self,
        quantile=0.95,
        d_model=64,
        nhead=4,
        num_layers=2,
        seq_len=20,
        epochs=50,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__(
            quantile,
            hidden_dim=d_model,
            num_layers=num_layers,
            seq_len=seq_len,
            epochs=epochs,
            lr=lr,
        )
        self.d_model = d_model
        self.nhead = nhead

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xt, yt = self._make_sequences(X, y)
        self._net = _TransformerEncoder(
            X.shape[1], self.d_model, self.nhead, self.num_layers, self.seq_len
        ).to(DEVICE)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        q = 1 - self.quantile

        for _ in range(self.epochs):
            self._net.train()
            pred = self._net(Xt)
            loss = quantile_loss(pred, yt, q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.is_fitted = True


class _JointTransformer(nn.Module):
    """2-head output: VaR and CVaR."""

    def __init__(self, input_dim, d_model, nhead, num_layers):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.var_head = nn.Linear(d_model, 1)
        self.cvar_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        h = x[:, -1, :]
        return self.var_head(h).squeeze(-1), self.cvar_head(h).squeeze(-1)


class JointTransformer(TransformerEncoder):
    """
    Transformer with 2-head joint VaR + CVaR output.
    Joint loss = quantile_loss(VaR) + es_loss(CVaR).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xt, yt = self._make_sequences(X, y)
        self._net = _JointTransformer(
            X.shape[1], self.d_model, self.nhead, self.num_layers
        ).to(DEVICE)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        q = 1 - self.quantile

        for _ in range(self.epochs):
            self._net.train()
            var_pred, cvar_pred = self._net(Xt)
            loss = quantile_loss(var_pred, yt, q) + es_loss(var_pred, cvar_pred, yt, q)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.is_fitted = True

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        Xt, _ = self._make_sequences(X)
        with torch.no_grad():
            var, _ = self._net(Xt)
        return var.cpu().numpy()

    def predict_cvar(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        Xt, _ = self._make_sequences(X)
        with torch.no_grad():
            _, cvar = self._net(Xt)
        return cvar.cpu().numpy()
