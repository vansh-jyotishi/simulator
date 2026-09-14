"""Shared types for the SIH 26055 RF environment simulator (contract C1).

This module is the only place where the environment (`env/`) and everything
downstream (`oracle/`, `baselines/`, `eval/`, the scheduler team, the
dashboard team) agree on data shapes.  It is frozen after tag ``contract-v1``:
adding a field with a default is allowed, renaming or removing is not.

Nothing here contains truth except :class:`Truth`, which is reachable only
through ``SpectrumEnv.get_truth()`` and must never be handed to a scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Closed set of emitter types; the scenario loader accepts no aliases.
EMITTER_TYPES = ("periodic", "agile", "scanning")

# Reward magnitudes (ML doc §0 / §9 / §14).  Signs are applied in the formula:
#   R_t = R_NEW*I_new + R_HIT*O_t - C_EMPTY*(1-O_t) - C_TUNE*switched
R_NEW, R_HIT, C_EMPTY, C_TUNE = 10.0, 2.0, 0.5, 1.5

# Exact key set of the per-step info dict.  Never contains truth.
INFO_KEYS = (
    "t",
    "next_t",
    "action",
    "switched",
    "blind",
    "blind_remaining",
    "reward",
    "reward_terms",
)


@dataclass(frozen=True)
class MissionSpec:
    """What an operator legitimately knows before the mission.  Never truth."""

    N: int
    T: int
    dwell_s: float
    tau_switch: int
    pfa: float
    initial_channel: int
    intel_weights: tuple[float, ...] | None  # len N, all >= 0, sum > 0, or None


@dataclass(frozen=True, eq=False)
class Truth:
    """Hidden truth.  ORACLE / DASHBOARD ONLY, via ``env.get_truth()``."""

    S: np.ndarray  # (N,T) int8, OR over S_by_emitter
    S_by_emitter: np.ndarray  # (M,N,T) int8
    SNR: np.ndarray  # (N,T) float32 dB, -inf where S==0, max over emitters on overlap
    E: np.ndarray  # (N,T) int16 emitter id with max SNR (ties: lowest id), -1 where S==0
    U: np.ndarray  # (N,T) float32 Uniform(0,1) pre-drawn noise = common random numbers
    emitter_names: tuple[str, ...]
    emitter_types: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class History:
    """Written by the env, read by the oracle.  Arrays have length T; slots >= t are unfilled (-1 / 0)."""

    actions: np.ndarray  # int32, -1 where unfilled
    obs: np.ndarray  # int8
    blind: np.ndarray  # bool
    switched: np.ndarray  # bool
    new_detect: np.ndarray  # bool (I_new)
    rewards: np.ndarray  # float32
    t: int  # slots executed so far


class Prediction(NamedTuple):
    """An ambush commitment made by a scheduler during the episode."""

    channel: int
    t_pred: int
    t_made: int
    status: str  # "issued" | "dropped"


@dataclass(frozen=True)
class Burst:
    """Maximal run of 1s for one (emitter, channel) in ``S_by_emitter``; ``t_end`` inclusive."""

    emitter: int
    emitter_type: str
    channel: int
    t_start: int
    t_end: int
