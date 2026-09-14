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

``terminated`` is always ``False``; an episode ends only by truncation at
``t == T``.  The truth is rendered from the ``make_env`` seed and cached:
``reset()`` without a seed replays the same truth, ``reset(seed=s)`` re-renders
it.  Every ``step()`` info dict has exactly ``INFO_KEYS`` and never holds truth.
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
    """Single-channel receiver over ``N`` channels for ``T`` slots.

    One episode is ``T`` dwells.  Each :meth:`step` tunes the receiver to one
    channel and returns the binary observation ``O_t`` plus an info dict with
    exactly ``INFO_KEYS``.  The hidden truth is rendered once per seed at
    :meth:`reset` (emitters are open-loop) and is reachable only through
    :meth:`get_truth`.  See the module docstring for the exact step order and
    the reward formula.

    Parameters
    ----------
    cfg : ScenarioConfig
        Validated scenario (band size, receiver block, reward magnitudes,
        emitter list) as returned by :func:`env.scenario.load_scenario`.
    seed : int
        Seed the hidden truth is rendered from.  ``reset(seed=None)`` reuses
        it; it never auto-advances between episodes.
    receiver : ReceiverModel or None, optional
        Receiver to drive.  ``None`` (default) builds a :class:`ReceiverModel`
        from ``cfg.receiver``.  When given, its ``tau_switch`` / ``pfa`` are
        the ones reported in ``mission_spec``.

    Attributes
    ----------
    cfg : ScenarioConfig
        The scenario the env was built from.
    N : int
        Number of channels.
    T : int
        Number of slots per episode.
    receiver : ReceiverModel
        Receiver in use; sole owner of the previous-channel and blind state.
    action_space : gymnasium.spaces.Discrete
        ``Discrete(N)``: the channel index to dwell on.
    observation_space : gymnasium.spaces.Discrete
        ``Discrete(2)``: the binary detection ``O_t``.
    mission_spec : MissionSpec
        Operator-visible mission spec (never truth), with the receiver's
        ``tau_switch`` / ``pfa``.
    initial_channel : int
        Channel the receiver is parked on before the first dwell.
    metadata : dict
        Gymnasium metadata; ``render_modes = ["ansi"]``.

    Notes
    -----
    ``terminated`` is always ``False``; the episode ends by truncation when
    ``t == T``.  Calling :meth:`step` before :meth:`reset`, or after the
    episode has ended, raises ``RuntimeError``.
    """

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
        """Truth seed in use (the ``make_env`` seed unless ``reset(seed=...)`` replaced it).

        Returns
        -------
        int
            The seed the hidden truth is rendered from.
        """
        return self._seed

    @property
    def t(self) -> int:
        """Slots executed so far.

        Returns
        -------
        int
            ``0`` right after :meth:`reset`, ``T`` once the episode is over.
        """
        return self._t

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        """Start an episode.

        The hidden truth is rendered (or reused from the previous episode when
        the seed is unchanged), the receiver is parked on ``initial_channel``
        with no blindness, the seen-emitter set and the history log are
        cleared and ``t`` is set to ``0``.

        Parameters
        ----------
        seed : int or None, optional
            New truth seed.  ``None`` (default) reuses the ``make_env`` seed;
            the seed never auto-advances, so consecutive resets without a
            seed replay the same truth.
        options : dict or None, optional
            Accepted for Gymnasium API compatibility and ignored.

        Returns
        -------
        obs : int
            Always ``0``; no dwell has been executed yet.
        info : dict
            Info dict with exactly ``INFO_KEYS``: ``t = -1``, ``next_t = 0``,
            ``action = initial_channel``, ``switched`` and ``blind`` both
            ``False``, ``blind_remaining = 0``, ``reward = 0.0`` and all-zero
            ``reward_terms``.
        """
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
        """Execute one dwell on channel ``action`` at the current slot ``t``.

        Follows the exact order in the module docstring: the receiver observes
        the cell, ``I_new`` is computed against the set of emitters already
        truly detected, the reward is assembled, ``History[t]`` is written and
        ``t`` advances.

        Parameters
        ----------
        action : int
            Channel to dwell on, in ``[0, N)``.

        Returns
        -------
        obs : int
            ``O_t in {0, 1}``; always ``0`` while the receiver is blind after
            a retune.
        reward : float
            ``R_NEW*I_new + R_HIT*O_t - C_EMPTY*(1-O_t) - C_TUNE*switched``.
        terminated : bool
            Always ``False``.
        truncated : bool
            ``True`` iff this was slot ``T - 1``; the episode is then over.
        info : dict
            Exactly ``INFO_KEYS``: ``t`` (slot just executed), ``next_t``,
            ``action``, ``switched``, ``blind``, ``blind_remaining``,
            ``reward`` and ``reward_terms`` (``new`` / ``hit`` / ``empty`` /
            ``tune``, signed so they sum to ``reward``).  Never contains truth.

        Raises
        ------
        RuntimeError
            If called before :meth:`reset` or after the episode has ended.
        ValueError
            If ``action`` is outside ``[0, N)``.
        """
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
        """Return the hidden truth.  ORACLE / DASHBOARD ONLY.  Never hand this to a scheduler.

        Renders the truth for the current seed on first use, so it may be
        called before :meth:`reset`.

        Returns
        -------
        Truth
            Frozen :class:`common.types.Truth` for the current seed: ``S``,
            ``S_by_emitter``, ``SNR``, ``E``, ``U`` and the emitter names and
            types.
        """
        if self._truth is None:
            self._truth = render_truth(self.cfg, self._seed)
            self._truth_seed = self._seed
            self._S, self._SNR, self._U, self._S_by = (self._truth.S, self._truth.SNR, self._truth.U,
                                                        self._truth.S_by_emitter)
        return self._truth

    def get_history(self) -> History:
        """Return live views of the episode log.

        The arrays are the env's own buffers, not copies, so a
        :class:`History` fetched once stays current as the episode advances.
        Slots ``>= t`` are unfilled (``-1`` for ``actions``, ``0`` / ``False``
        elsewhere).

        Returns
        -------
        History
            ``actions``, ``obs``, ``blind``, ``switched``, ``new_detect`` and
            ``rewards`` arrays of length ``T`` plus ``t``, the number of
            slots executed so far.
        """
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

        Parameters
        ----------
        width : int, optional
            Number of slots to show, default ``100``.  The window is
            ``[max(0, t - width), t)``; before the first step it is the first
            ``min(width, T)`` slots of truth with no track overlay.

        Returns
        -------
        str
            A header ``SpectrumEnv <name> seed=<seed> t=<t>/<T> slots [t0,t1)``
            followed by one ``"<n> |<chars>"`` line per channel.

        Notes
        -----
        Uses :meth:`get_truth`, so this is for the dashboard and debugging
        only; never expose the output to a scheduler.
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
    """Build the env from a scenario file or dict.

    Parameters
    ----------
    scenario : str or Path or dict
        Scenario JSON path (relative paths resolve against the cwd, then the
        repo root) or an in-memory dict of the same shape; handed to
        :func:`env.scenario.load_scenario`.
    seed : int
        Truth seed handed to :class:`SpectrumEnv`.
    receiver : ReceiverModel or None, optional
        Overrides the scenario's receiver block (its ``tau_switch`` / ``pfa``
        win and are what ``mission_spec`` reports).  ``None`` (default) builds
        the receiver from the scenario.

    Returns
    -------
    SpectrumEnv
        Un-reset environment; call :meth:`SpectrumEnv.reset` before stepping.

    Raises
    ------
    ValueError
        From :func:`env.scenario.load_scenario` for every contract violation,
        naming the offending field.
    FileNotFoundError
        If ``scenario`` is a path found neither in the cwd nor the repo root.
    TypeError
        If ``scenario`` is neither a path nor a dict.
    """
    cfg = load_scenario(scenario)
    return SpectrumEnv(cfg, seed, receiver)


__all__ = ["SpectrumEnv", "make_env", "INFO_KEYS"]
