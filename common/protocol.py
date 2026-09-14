"""Scheduler protocol (contract C7).

Implemented by ``baselines/`` and by the scheduler team, consumed by ``eval/``.

Rules
-----
* Constructors take no required arguments (configuration goes through
  ``reset`` or the ``--scheduler-kwargs`` CLI option).
* ``select_action(obs, info)`` receives ``obs = O_{t-1}`` (0 at the first call)
  and ``info = sanitize_info(previous info)``; it returns ``a_t`` in ``[0, N)``.
* ``info["reward"]`` is training telemetry (``I_new`` is truth-derived).
  Runtime candidates must not condition decisions on it; offline benchmarks may.
* Never import ``env/`` internals and never call ``env.get_truth()``.
* Randomness is derived from ``np.random.default_rng([seed, 20_000])`` inside
  ``reset`` so that every scheduler sees the same seed stream layout.

Cold-start warning
------------------
The ML doc §7 one-dwell sweep ``a_t = t`` is 100 % blind at ``tau_switch = 1``.
A bootstrap sweep must dwell ``spec.tau_switch + 1`` slots per channel and drop
blind observations (``info["blind"]``) from tau=1 transition counts.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from common.types import INFO_KEYS, MissionSpec, Prediction

_INFO_KEY_SET = frozenset(INFO_KEYS)


def sanitize_info(info: dict) -> dict:
    """Whitelist ``INFO_KEYS``.  The runner applies this before anything reaches a scheduler.

    Any key outside the contract (for example an injected ``"S"``) is dropped.
    ``reward_terms`` is shallow-copied so a scheduler cannot mutate env state.
    """
    out = {}
    for k in INFO_KEYS:
        if k in info:
            v = info[k]
            if k == "reward_terms" and isinstance(v, dict):
                v = dict(v)
            out[k] = v
    return out


@runtime_checkable
class Scheduler(Protocol):
    """Structural type every scheduler must satisfy."""

    needs_truth: bool  # False for every candidate; True only for baselines.clairvoyant

    def reset(self, spec: MissionSpec, seed: int) -> None: ...

    def select_action(self, obs: int, info: dict) -> int: ...

    def predictions(self) -> list[Prediction]: ...


class BaseScheduler:
    """Working default: stores ``spec``, ``seed`` and ``rng``; ``predictions()`` returns ``[]``.

    Subclass it, implement ``select_action`` and (optionally) extend ``reset``.
    """

    needs_truth: bool = False

    def __init__(self) -> None:
        self.spec: MissionSpec | None = None
        self.seed: int | None = None
        self.rng: np.random.Generator | None = None
        self._predictions: list[Prediction] = []

    def reset(self, spec: MissionSpec, seed: int) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.rng = np.random.default_rng([int(seed), 20_000])
        self._predictions = []

    def select_action(self, obs: int, info: dict) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def predictions(self) -> list[Prediction]:
        return list(self._predictions)
