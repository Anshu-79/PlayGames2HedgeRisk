# src/models/rl/agent.py
#
# PLACEHOLDER — RL agent skeleton.
# Implement policy network, reward shaping, and training loop here.
# Environment is fully wired; agent internals are stubs.

import numpy as np
import torch
import torch.nn as nn
from src.models.base import BaseRLAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class RiskEnv:
    """
    Custom RL environment for VaR/CVaR estimation.

    State : rolling window of features (returns, vols, drawdown, ...)
    Action: scalar VaR estimate (continuous)
    Reward: penalize VaR violations + CVaR over-estimation
    """

    def __init__(
        self,
        returns: np.ndarray,
        features: np.ndarray,
        window: int = 20,
        quantile: float = 0.95,
        cvar_penalty: float = 1.0,
        risk_penalty: float = 1.0,
    ):
        self.returns = returns
        self.features = features
        self.window = window
        self.quantile = quantile
        self.cvar_penalty = cvar_penalty
        self.risk_penalty = risk_penalty
        self.t = window
        self.done = False

    def reset(self) -> np.ndarray:
        self.t = self.window
        self.done = False
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return self.features[self.t - self.window : self.t].flatten()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        action: predicted VaR value (negative float, e.g. -0.02)
        """
        r_t = self.returns[self.t]
        var_pred = action

        # ---------- reward components ----------
        violation = float(r_t < var_pred)  # 1 if loss exceeds VaR

        # Quantile calibration reward (penalize wrong violation rate)
        q = 1 - self.quantile
        reward_var = -abs(violation - q) * self.risk_penalty

        # CVaR penalty: if violated, penalize magnitude of excess loss
        if violation:
            cvar_penalty = self.cvar_penalty * abs(r_t - var_pred)
        else:
            cvar_penalty = 0.0

        reward = reward_var - cvar_penalty
        # ---------------------------------------

        self.t += 1
        self.done = self.t >= len(self.returns)
        obs = (
            self._get_obs()
            if not self.done
            else np.zeros(self.window * self.features.shape[1])
        )
        info = {"r_t": r_t, "var_pred": var_pred, "violation": violation}
        return obs, reward, self.done, info

    @property
    def obs_dim(self) -> int:
        return self.window * self.features.shape[1]


# ---------------------------------------------------------------------------
# Policy Network (placeholder)
# ---------------------------------------------------------------------------


class _PolicyNet(nn.Module):
    """
    Simple MLP policy.
    TODO: replace with LSTM / Transformer policy for temporal dependencies.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # output: VaR estimate
            nn.Tanh(),  # bound output in [-1, 1]; scale downstream
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# RL Agent (placeholder — REINFORCE skeleton)
# ---------------------------------------------------------------------------


class RLVaRAgent(BaseRLAgent):
    """
    PLACEHOLDER RL agent for VaR/CVaR estimation.

    Algorithm : REINFORCE (vanilla policy gradient) — replace with PPO/SAC.
    Status    : skeleton only — training loop not implemented.

    TODO:
      - implement proper policy gradient update
      - add value network (actor-critic)
      - implement PPO clipping or SAC entropy bonus
      - add replay buffer for off-policy methods
      - implement CVaR-aware reward shaping
    """

    def __init__(
        self,
        quantile: float = 0.95,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        cvar_reward: bool = True,
        risk_penalty: bool = True,
    ):
        super().__init__(quantile)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.cvar_reward = cvar_reward
        self.risk_penalty = risk_penalty
        self._policy = None
        self._optimizer = None

    def _init_policy(self, obs_dim: int):
        self._policy = _PolicyNet(obs_dim, self.hidden_dim).to(DEVICE)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.lr)

    def train(self, env: RiskEnv, n_episodes: int = 100) -> None:
        """
        TODO: Implement full training loop.
        Current: random-action placeholder that logs episode rewards.
        """
        if self._policy is None:
            self._init_policy(env.obs_dim)

        for ep in range(n_episodes):
            obs = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                # --- PLACEHOLDER: random action ---
                # Replace with: action = self._policy(torch.tensor(obs)).item() * scale
                action = np.random.uniform(-0.05, 0.0)
                obs, reward, done, info = env.step(action)
                ep_reward += reward

            # TODO: collect trajectories, compute returns, update policy
            if ep % 10 == 0:
                print(
                    f"[RL] Episode {ep:4d} | Total Reward: {ep_reward:.4f}  [PLACEHOLDER]"
                )

    def evaluate(self, env: RiskEnv) -> dict:
        """Run policy on env, return VaR predictions and violation stats."""
        obs = env.reset()
        done = False
        var_preds, actuals, violations = [], [], []

        while not done:
            if self._policy is not None:
                with torch.no_grad():
                    action = (
                        self._policy(
                            torch.tensor(obs, dtype=torch.float32).to(DEVICE)
                        ).item()
                        * 0.05
                    )
            else:
                action = np.random.uniform(-0.05, 0.0)  # fallback

            obs, reward, done, info = env.step(action)
            var_preds.append(info["var_pred"])
            actuals.append(info["r_t"])
            violations.append(info["violation"])

        return {
            "var_preds": np.array(var_preds),
            "actuals": np.array(actuals),
            "violations": np.array(violations),
        }

    def predict_var(self, obs: np.ndarray) -> float:
        if self._policy is None:
            raise RuntimeError("Agent not trained yet.")
        with torch.no_grad():
            return (
                self._policy(torch.tensor(obs, dtype=torch.float32).to(DEVICE)).item()
                * 0.05
            )

    def save(self, path: str) -> None:
        torch.save(self._policy.state_dict(), path)
        print(f"Saved RL policy to {path}")

    def load(self, path: str) -> None:
        if self._policy is None:
            raise RuntimeError("Call train() or init_policy() first to set obs_dim.")
        self._policy.load_state_dict(torch.load(path))
        print(f"Loaded RL policy from {path}")
