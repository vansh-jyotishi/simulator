"""Scripted schedulers and hand-built tiny fixtures for oracle / verification tests.

The scripted schedulers satisfy the C7 protocol.  They take constructor
arguments because they are test helpers, not candidates.

``tiny_truth()`` / ``tiny_history()`` are hand-built (N=4, T=30) so every
metric can be checked against numbers worked out by hand (theory guide Part 6
ratios: Pd 4/5, Pfa 1/20, POI 2/4, timing errors [2,4,1,3,5] -> 3.0,
4/5 predictions -> 80 %).
"""
from __future__ import annotations

import numpy as np

from common.protocol import BaseScheduler
from common.types import History, MissionSpec, Prediction, Truth


def play(scenario, seed: int, scheduler, receiver=None):
    """Minimal loop: run ``scheduler`` on ``make_env(scenario, seed)`` to the end and return the env.

    Applies ``sanitize_info`` like the real runner.  Test helper only.
    """
    from common.protocol import sanitize_info
    from env.spectrum_env import make_env

    env = make_env(scenario, seed, receiver) if receiver is not None else make_env(scenario, seed)
    obs, info = env.reset()
    if getattr(scheduler, "needs_truth", False):
        scheduler.set_truth(env.get_truth())
    scheduler.reset(env.mission_spec, seed)
    while True:
        a = scheduler.select_action(obs, sanitize_info(info))
        obs, r, term, trunc, info = env.step(a)
        if trunc:
            break
    return env


class StayPut(BaseScheduler):
    """Always dwell on one channel."""

    def __init__(self, channel: int = 0) -> None:
        super().__init__()
        self.channel = int(channel)

    def select_action(self, obs: int, info: dict) -> int:
        return self.channel


class Scripted(BaseScheduler):
    """Replay a fixed action list (the last action repeats if the episode is longer)."""

    def __init__(self, actions=()) -> None:
        super().__init__()
        self.actions = [int(a) for a in actions]
        self._i = 0

    def reset(self, spec: MissionSpec, seed: int) -> None:
        super().reset(spec, seed)
        self._i = 0

    def select_action(self, obs: int, info: dict) -> int:
        a = self.actions[min(self._i, len(self.actions) - 1)]
        self._i += 1
        return a


class SwitchEvery(BaseScheduler):
    """Cycle through ``channels``, holding each for ``k`` dwells."""

    def __init__(self, k: int = 2, channels=(0, 1)) -> None:
        super().__init__()
        self.k = int(k)
        self.channels = [int(c) for c in channels]
        self._t = 0

    def reset(self, spec: MissionSpec, seed: int) -> None:
        super().reset(spec, seed)
        self._t = 0

    def select_action(self, obs: int, info: dict) -> int:
        a = self.channels[(self._t // self.k) % len(self.channels)]
        self._t += 1
        return a


class FakePredictor(StayPut):
    """StayPut that also reports a fixed list of predictions."""

    def __init__(self, channel: int = 0, predictions=()) -> None:
        super().__init__(channel)
        self._fixed = list(predictions)

    def predictions(self) -> list[Prediction]:
        return list(self._fixed)


# ---------------------------------------------------------------------------
# Hand-built fixture: N=4, T=30, two emitters, tau_switch=1 (blind iff switched)
# ---------------------------------------------------------------------------
TINY_N, TINY_T = 4, 30
# emitter 0 "radar" (periodic) on channel 1: bursts [2,4], [13,15], [23,25]
# emitter 1 "hopper" (agile)   on channel 3: burst  [6,7]
TINY_BURSTS = [  # (emitter, type, channel, t_start, t_end) sorted by t_start
    (0, "periodic", 1, 2, 4),
    (1, "agile", 3, 6, 7),
    (0, "periodic", 1, 13, 15),
    (0, "periodic", 1, 23, 25),
]
TINY_ACTIONS = [0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
TINY_OBS_ONES = {3, 4, 13, 14, 16}  # 3,4,13,14 true positives; 16 false alarm
TINY_SWITCH_T = {2, 7, 10, 12, 22}  # blind iff switched (tau_switch = 1)

# Expected hand numbers
TINY_EXPECTED = dict(
    n_tp=4, n_fn=1, n_fp=1, n_tn=19, n_blind=5, n_switches=5,
    sensor_pd=4 / 5, sensor_pfa=1 / 20, effective_pd=4 / 7,
    n_bursts=4, n_caught=2, poi=2 / 4, poi_periodic=2 / 3, poi_agile=0.0,
    poi_time=4 / 11, mean_tti=0.5, median_tti=0.5, discovery_ratio=0.5,
    intercept_rate_per_dwell=2 / 30, intercept_rate_per_s=2 / (30 * 1e-3),
    total_reward=0.0, mean_reward=0.0, blind_fraction=5 / 30, switch_rate=5 / 30,
)

# Prediction sets (channel, t_pred, t_made, status)
PRED_PERFECT = [Prediction(1, 2, 0, "issued"), Prediction(1, 13, 5, "issued"), Prediction(1, 23, 16, "issued")]
PRED_OFF_BY_2 = [Prediction(1, 0, 0, "issued"), Prediction(1, 11, 5, "issued"), Prediction(1, 21, 16, "issued"),
                 Prediction(1, 5, 0, "dropped")]
PRED_THEORY_ERRORS = [Prediction(1, 4, 0, "issued"), Prediction(1, 17, 0, "issued"), Prediction(1, 24, 0, "issued"),
                      Prediction(1, 26, 0, "issued"), Prediction(1, 8, 0, "issued")]  # errors 2,4,1,3,5 -> 3.0; 3/5 correct
PRED_80PCT = [Prediction(1, 3, 0, "issued"), Prediction(1, 14, 0, "issued"), Prediction(1, 24, 0, "issued"),
              Prediction(3, 6, 0, "issued"), Prediction(1, 8, 0, "issued"), Prediction(1, 9, 0, "dropped")]


def tiny_truth() -> Truth:
    N, T = TINY_N, TINY_T
    S_by = np.zeros((2, N, T), dtype=np.int8)
    for m, _, c, s, e in TINY_BURSTS:
        S_by[m, c, s:e + 1] = 1
    S = S_by.any(axis=0).astype(np.int8)
    SNR = np.where(S == 1, 10.0, -np.inf).astype(np.float32)
    E = np.where(S == 1, S_by.argmax(axis=0), -1).astype(np.int16)
    U = np.zeros((N, T), dtype=np.float32)
    for arr in (S, S_by, SNR, E, U):
        arr.setflags(write=False)
    return Truth(S=S, S_by_emitter=S_by, SNR=SNR, E=E, U=U,
                 emitter_names=("radar", "hopper"), emitter_types=("periodic", "agile"))


def tiny_history(t: int | None = None) -> History:
    T = TINY_T
    actions = np.asarray(TINY_ACTIONS, dtype=np.int32)
    obs = np.array([1 if i in TINY_OBS_ONES else 0 for i in range(T)], dtype=np.int8)
    switched = np.array([i in TINY_SWITCH_T for i in range(T)], dtype=bool)
    blind = switched.copy()
    truth = tiny_truth()
    s_at = truth.S[actions, np.arange(T)].astype(bool)
    new_detect = np.zeros(T, dtype=bool)
    new_detect[3] = True  # first true detection of the radar; the hopper is never truly detected
    rewards = np.zeros(T, dtype=np.float32)
    for i in range(T):
        o = int(obs[i])
        rewards[i] = 10.0 * new_detect[i] + 2.0 * o - 0.5 * (1 - o) - 1.5 * switched[i]
    if t is not None:
        actions = actions.copy(); obs = obs.copy(); blind = blind.copy(); switched = switched.copy()
        new_detect = new_detect.copy(); rewards = rewards.copy()
        actions[t:] = -1; obs[t:] = 0; blind[t:] = False; switched[t:] = False
        new_detect[t:] = False; rewards[t:] = 0.0
    return History(actions=actions, obs=obs, blind=blind, switched=switched, new_detect=new_detect,
                   rewards=rewards, t=T if t is None else t)
