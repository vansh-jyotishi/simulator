"""Gymnasium environment for the single-channel scan problem (contract C2).

``make_env(scenario, seed)`` builds a :class:`SpectrumEnv` whose hidden truth
is rendered in full at ``reset()`` (emitters are open-loop).  The scheduler
sees only ``O_t in {0, 1}`` and an info dict with exactly ``INFO_KEYS``; the
oracle and dashboard reach the truth through :meth:`SpectrumEnv.get_truth`.

Exact ``step(a)`` order for slot ``t``:

1. ``O_t, switched, blind, blind_remaining = receiver.observe(a, S[a,t], SNR[a,t], U[a,t])``
2. ``I_new = 1`` iff ``O_t == 1 and S[a,t] == 1`` and some emitter active in that
   cell was never truly detected before; all active ones are then marked seen.
3. ``R_t = R_NEW*I_new + R_HIT*O_t - C_EMPTY*(1-O_t) - C_TUNE*switched``
4. Write ``History[t]``; ``t += 1``; ``truncated = (t == T)``; build info.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from common.types import INFO_KEYS, History, MissionSpec, Truth
from env.emitters import render_truth
from env.receiver_model import ReceiverModel
from env.scenario import ScenarioConfig, load_scenario, to_mission_spec


class SpectrumEnv(gym.Env):
    """Single-channel receiver over ``N`` channels for ``T`` slots.  See module docstring."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: ScenarioConfig, seed: int, receiver: ReceiverModel | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.N = int(cfg.N)
        self.T = int(cfg.T)
        self._seed = int(seed)
        if receiver is None:
            rc = cfg.receiver
            receiver = ReceiverModel(rc.tau_switch, rc.pfa, rc.noise_model, rc.pd_fixed, rc.n_pulses)
        self.receiver = receiver
        self.action_space = Discrete(self.N)
        self.observation_space = Discrete(2)
        self.mission_spec: MissionSpec = to_mission_spec(cfg, tau_switch=receiver.tau_switch, pfa=receiver.pfa)
        self.initial_channel = int(cfg.receiver.initial_channel)
        rw = cfg.reward
        self._r_new, self._r_hit, self._c_empty, self._c_tune = rw.R_new, rw.R_hit, rw.C_empty, rw.C_tune

        self._truth: Truth | None = None
        self._truth_seed: int | None = None
        self._S: np.ndarray | None = None
        self._SNR: np.ndarray | None = None
        self._U: np.ndarray | None = None
        self._S_by: np.ndarray | None = None
        self._seen: np.ndarray = np.zeros(0, dtype=bool)
        self._t = 0
        self._done = True  # step() before reset() is an error

        T = self.T
        self._h_actions = np.full(T, -1, dtype=np.int32)
        self._h_obs = np.zeros(T, dtype=np.int8)
        self._h_blind = np.zeros(T, dtype=bool)
        self._h_switched = np.zeros(T, dtype=bool)
        self._h_new = np.zeros(T, dtype=bool)
        self._h_rewards = np.zeros(T, dtype=np.float32)

    # ------------------------------------------------------------------ props
    @property
    def seed(self) -> int:
        return self._seed

    @property
    def t(self) -> int:
        """Slots executed so far."""
        return self._t

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        """Start an episode.  ``seed=None`` reuses the ``make_env`` seed (never auto-advances)."""
        if seed is not None:
            self._seed = int(seed)
        super().reset(seed=self._seed)
        if self._truth is None or self._truth_seed != self._seed:
            self._truth = render_truth(self.cfg, self._seed)
            self._truth_seed = self._seed
            self._S = self._truth.S
            self._SNR = self._truth.SNR
            self._U = self._truth.U
            self._S_by = self._truth.S_by_emitter
        self._seen = np.zeros(self._S_by.shape[0], dtype=bool)
        self.receiver.reset(self.initial_channel)
        self._t = 0
        self._done = False
        self._h_actions.fill(-1)
        self._h_obs.fill(0)
        self._h_blind.fill(False)
        self._h_switched.fill(False)
        self._h_new.fill(False)
        self._h_rewards.fill(0.0)
        info0 = {
            "t": -1,
            "next_t": 0,
            "action": self.initial_channel,
            "switched": False,
            "blind": False,
            "blind_remaining": 0,
            "reward": 0.0,
            "reward_terms": {"new": 0.0, "hit": 0.0, "empty": 0.0, "tune": 0.0},
        }
        return 0, info0

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("step() called after the episode ended (or before reset())")
        a = int(action)
        if not (0 <= a < self.N):
            raise ValueError(f"action {a} outside [0, {self.N})")
        t = self._t
        s_at = int(self._S[a, t])
        o, switched, blind, blind_remaining = self.receiver.observe(a, s_at, float(self._SNR[a, t]), float(self._U[a, t]))

        i_new = 0
        if o and s_at:
            active = self._S_by[:, a, t] != 0
            if np.any(active & ~self._seen):
                i_new = 1
                self._seen |= active

        r_new = self._r_new if i_new else 0.0
        r_hit = self._r_hit if o else 0.0
        r_empty = 0.0 if o else -self._c_empty
        r_tune = -self._c_tune if switched else 0.0
        reward = r_new + r_hit + r_empty + r_tune

        self._h_actions[t] = a
        self._h_obs[t] = o
        self._h_blind[t] = blind
        self._h_switched[t] = switched
        self._h_new[t] = bool(i_new)
        self._h_rewards[t] = reward

        self._t = t + 1
        truncated = self._t == self.T
        self._done = truncated
        info = {
            "t": t,
            "next_t": t + 1,
            "action": a,
            "switched": bool(switched),
            "blind": bool(blind),
            "blind_remaining": int(blind_remaining),
            "reward": float(reward),
            "reward_terms": {"new": float(r_new), "hit": float(r_hit), "empty": float(r_empty), "tune": float(r_tune)},
        }
        return o, float(reward), False, truncated, info

    # ------------------------------------------------------------------ oracle access
    def get_truth(self) -> Truth:
        """ORACLE / DASHBOARD ONLY.  Never hand this to a scheduler."""
        if self._truth is None:
            self._truth = render_truth(self.cfg, self._seed)
            self._truth_seed = self._seed
            self._S, self._SNR, self._U, self._S_by = (self._truth.S, self._truth.SNR, self._truth.U,
                                                        self._truth.S_by_emitter)
        return self._truth

    def get_history(self) -> History:
        """Live views of the episode log; slots ``>= t`` are unfilled (-1 / 0)."""
        return History(
            actions=self._h_actions, obs=self._h_obs, blind=self._h_blind, switched=self._h_switched,
            new_detect=self._h_new, rewards=self._h_rewards, t=self._t,
        )

    # ------------------------------------------------------------------ render
    def render(self, width: int = 100) -> str:
        """ASCII waterfall of the last ``width`` slots.

        Rows are channels (0-based).  ``#`` truth on, ``.`` truth off; the receiver's
        track overlays ``@`` true positive, ``x`` blind dwell, ``?`` false alarm,
        ``o`` visited empty, ``m`` visited occupied but missed.
        """
        truth = self.get_truth()
        t1 = self._t
        t0 = max(0, t1 - width)
        if t1 == 0:
            t0, t1 = 0, min(width, self.T)
        S = truth.S[:, t0:t1]
        rows = []
        for n in range(self.N):
            chars = ["#" if s else "." for s in S[n]]
            rows.append(chars)
        for t in range(t0, min(t1, self._t)):
            a = int(self._h_actions[t])
            if a < 0:
                continue
            s = int(truth.S[a, t])
            o = int(self._h_obs[t])
            if self._h_blind[t]:
                c = "x"
            elif s and o:
                c = "@"
            elif s and not o:
                c = "m"
            elif o:
                c = "?"
            else:
                c = "o"
            rows[a][t - t0] = c
        header = f"SpectrumEnv {self.cfg.name} seed={self._seed} t={self._t}/{self.T} slots [{t0},{t1})"
        return "\n".join([header] + [f"{n:3d} |" + "".join(r) for n, r in enumerate(rows)])


def make_env(scenario: str | Path | dict, seed: int, receiver: ReceiverModel | None = None) -> SpectrumEnv:
    """Build the env.  ``receiver`` overrides the scenario's receiver block (its ``tau_switch``/``pfa`` win)."""
    cfg = load_scenario(scenario)
    return SpectrumEnv(cfg, seed, receiver)


__all__ = ["SpectrumEnv", "make_env", "INFO_KEYS"]
