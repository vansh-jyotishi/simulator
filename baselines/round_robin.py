"""Round-robin baseline: the classic fixed sweep (theory guide Part 7, "old way").

``a_t = (t // d) % N`` with ``d = dwell_steps`` or, by default,
``spec.tau_switch + 1``.  The default matters: with ``tau_switch = 1`` a sweep
that switches every dwell is blind on *every* dwell (POI 0, Pd NaN), so each
channel is held for one blind slot plus one listening slot.  Open-loop: the
observation stream is ignored.
"""
from __future__ import annotations

from common.protocol import BaseScheduler
from common.types import MissionSpec


class RoundRobin(BaseScheduler):
    needs_truth = False

    def __init__(self, dwell_steps: int | None = None) -> None:
        super().__init__()
        self.dwell_steps = None if dwell_steps is None else int(dwell_steps)
        self.d = 1
        self._t = 0

    def reset(self, spec: MissionSpec, seed: int) -> None:
        super().reset(spec, seed)
        self.d = self.dwell_steps if self.dwell_steps is not None else spec.tau_switch + 1
        if self.d < 1:
            raise ValueError(f"dwell_steps must be >= 1, got {self.d}")
        self._t = 0

    def select_action(self, obs: int, info: dict) -> int:
        a = (self._t // self.d) % self.spec.N
        self._t += 1
        return int(a)
