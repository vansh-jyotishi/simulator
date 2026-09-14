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
    """Smooth weighted round-robin over ``spec.intel_weights``, one new pick every ``d`` slots.

    Every ``d`` slots the scheduler adds the weight vector to a running per-channel credit, dwells
    on the channel with the largest credit (ties resolve to the lowest index) and subtracts the
    total weight from that channel's credit.  Over a full cycle each channel is therefore visited
    in proportion to its weight while the visits stay evenly spread in time.  The scheduler is
    open-loop: ``obs`` and ``info`` are ignored and ``rng`` is never drawn.

    Parameters
    ----------
    dwell_steps : int or None, optional
        Number of consecutive slots to hold each pick.  ``None`` (the default) resolves to
        ``spec.tau_switch + 1`` at :meth:`reset`, for the same cold-start reason as
        :class:`baselines.round_robin.RoundRobin`.

    Attributes
    ----------
    needs_truth : bool
        Always ``False``; the scheduler never reads the hidden truth.
    dwell_steps : int or None
        The constructor argument, cast to ``int`` when given.
    d : int
        Effective dwell length in slots, resolved by :meth:`reset` (``1`` until then).

    Notes
    -----
    With ``spec.intel_weights = None`` the weights are uniform and the visit order is identical to
    :class:`baselines.round_robin.RoundRobin` with the same dwell.
    """

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
        """Store the mission spec, validate the weights and reset the credit vector to zero.

        Parameters
        ----------
        spec : MissionSpec
            Mission parameters.  ``spec.intel_weights`` (length ``N``, all ``>= 0``, sum ``> 0``)
            supplies the weights; ``None`` means uniform weights.  ``spec.tau_switch`` sets the
            default dwell.
        seed : int
            Episode seed, forwarded to :meth:`common.protocol.BaseScheduler.reset`.  Unused beyond
            that: the schedule is deterministic.

        Raises
        ------
        ValueError
            If the effective dwell ``d`` is < 1, or if ``spec.intel_weights`` is not ``None`` and has
            the wrong length, a negative entry or a non-positive sum.
        """
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
        """Advance the smooth weighted round-robin by one pick.

        Returns
        -------
        int
            Channel with the largest credit after adding the weights; ties go to the lowest index.
        """
        self._credit += self._w
        i = int(np.argmax(self._credit))  # ties -> lowest channel
        self._credit[i] -= self._wsum
        return i

    def select_action(self, obs: int, info: dict) -> int:
        """Return the channel to dwell on, choosing a new one every ``d`` slots.

        Parameters
        ----------
        obs : int
            Previous observation ``O_{t-1}`` (0 at the first call).  Ignored.
        info : dict
            Sanitised info dict from the previous step.  Ignored.

        Returns
        -------
        int
            Channel index in ``[0, N)``: a fresh pick when ``t % d == 0``, otherwise the current one.
        """
        if self._t % self.d == 0:
            self._current = self._pick()
        self._t += 1
        return self._current
