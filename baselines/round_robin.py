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
    """Fixed-order sweep over all ``N`` channels, holding each one for ``d`` consecutive slots.

    The action at step ``t`` is ``a_t = (t // d) % N``.  The scheduler is open-loop: ``obs`` and
    ``info`` are accepted only to satisfy the :class:`common.protocol.Scheduler` contract and are
    otherwise ignored.  ``needs_truth`` is ``False``, so it is a legal candidate for the eval runner.

    Parameters
    ----------
    dwell_steps : int or None, optional
        Number of consecutive slots to hold each channel.  ``None`` (the default) resolves to
        ``spec.tau_switch + 1`` at :meth:`reset`, the smallest dwell that guarantees one non-blind
        listening slot per visit when the receiver is blind for ``tau_switch`` slots after a retune.

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
    With the default dwell every visit to a channel consists of ``tau_switch`` blind slots followed
    by exactly one listening slot; a one-dwell sweep (``d = 1``) would be blind on every slot at
    ``tau_switch = 1`` (POI 0, Pd NaN).
    """

    needs_truth = False

    def __init__(self, dwell_steps: int | None = None) -> None:
        super().__init__()
        self.dwell_steps = None if dwell_steps is None else int(dwell_steps)
        self.d = 1
        self._t = 0

    def reset(self, spec: MissionSpec, seed: int) -> None:
        """Store the mission spec, resolve the dwell length and restart the sweep at channel 0.

        Parameters
        ----------
        spec : MissionSpec
            Mission parameters; ``spec.N`` sets the sweep length and ``spec.tau_switch`` the
            default dwell.
        seed : int
            Episode seed, forwarded to :meth:`common.protocol.BaseScheduler.reset`.  Unused beyond
            that: the sweep is deterministic.

        Raises
        ------
        ValueError
            If the effective dwell ``d`` (``dwell_steps`` or ``spec.tau_switch + 1``) is < 1.
        """
        super().reset(spec, seed)
        self.d = self.dwell_steps if self.dwell_steps is not None else spec.tau_switch + 1
        if self.d < 1:
            raise ValueError(f"dwell_steps must be >= 1, got {self.d}")
        self._t = 0

    def select_action(self, obs: int, info: dict) -> int:
        """Return the next channel of the sweep and advance the internal step counter.

        Parameters
        ----------
        obs : int
            Previous observation ``O_{t-1}`` (0 at the first call).  Ignored.
        info : dict
            Sanitised info dict from the previous step.  Ignored.

        Returns
        -------
        int
            Channel index ``a_t = (t // d) % N`` in ``[0, N)``.
        """
        a = (self._t // self.d) % self.spec.N
        self._t += 1
        return int(a)
