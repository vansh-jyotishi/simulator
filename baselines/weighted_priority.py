"""Priority-weighted sweep baseline (theory guide Part 7, "static weights from old intel").

Smooth weighted round-robin over ``spec.intel_weights``: every pick adds the
weight vector to a running credit, dwells on the argmax and subtracts the
total weight from that channel's credit.  Each pick is held for ``dwell_steps``
slots (default ``spec.tau_switch + 1``, same reason as :mod:`baselines.round_robin`).
With ``intel_weights = None`` the weights are uniform and the visit order is
identical to :class:`baselines.round_robin.RoundRobin`.  Open-loop, no rng.
"""
from __future__ import annotations

import numpy as np

from common.protocol import BaseScheduler
from common.types import MissionSpec


class WeightedPriority(BaseScheduler):
    needs_truth = False

    def __init__(self, dwell_steps: int | None = None) -> None:
        super().__init__()
        self.dwell_steps = None if dwell_steps is None else int(dwell_steps)
        self.d = 1
        self._t = 0
        self._w = np.ones(1)
        self._credit = np.zeros(1)
        self._current = 0

    def reset(self, spec: MissionSpec, seed: int) -> None:
        super().reset(spec, seed)
        self.d = self.dwell_steps if self.dwell_steps is not None else spec.tau_switch + 1
        if self.d < 1:
            raise ValueError(f"dwell_steps must be >= 1, got {self.d}")
        if spec.intel_weights is None:
            self._w = np.ones(spec.N, dtype=np.float64)
        else:
            self._w = np.asarray(spec.intel_weights, dtype=np.float64)
            if self._w.shape != (spec.N,) or np.any(self._w < 0) or self._w.sum() <= 0:
                raise ValueError("intel_weights must have length N, be >= 0 and sum > 0")
        self._wsum = float(self._w.sum())
        self._credit = np.zeros(spec.N, dtype=np.float64)
        self._t = 0
        self._current = 0

    def _pick(self) -> int:
        self._credit += self._w
        i = int(np.argmax(self._credit))  # ties -> lowest channel
        self._credit[i] -= self._wsum
        return i

    def select_action(self, obs: int, info: dict) -> int:
        if self._t % self.d == 0:
            self._current = self._pick()
        self._t += 1
        return self._current
