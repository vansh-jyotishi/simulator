"""CHEATING upper bound: a scheduler that reads the hidden truth.

``needs_truth = True``; the eval runner hands it ``env.get_truth()`` through
:meth:`Clairvoyant.set_truth` before ``reset``.  It is refused for ``--extra``
classes and is never a candidate, only the ceiling row in the table.

Policy (greedy, one step of look-ahead through the blind window): if an
uncaught burst is active *now* on the current channel, stay.  Otherwise switch
to the channel of the uncaught burst that can be listened to soonest after the
retune blindness (``t + tau_switch``); ties go to the earliest-starting burst.
If nothing is catchable yet, move early to the next future burst and wait.
Caught status is tracked from the receiver's own observations, so a missed
detection (Pd < 1) makes it try again.
"""
from __future__ import annotations

from common.protocol import BaseScheduler
from common.types import MissionSpec, Truth
from oracle.truth_tracker import segment_bursts


class Clairvoyant(BaseScheduler):
    """Truth-reading greedy scheduler used only as the upper-bound row of the eval table.

    ``needs_truth`` is ``True``: the eval runner must call :meth:`set_truth` with
    ``env.get_truth()`` before :meth:`reset`.  The scheduler is refused for ``--extra`` classes and
    is never a candidate.  The constructor takes no arguments.

    The policy is greedy with one step of look-ahead through the retune blind window:

    * if an uncaught burst is active *now* on the current channel, stay;
    * otherwise switch to the channel of the uncaught burst that can be listened to soonest after
      the blindness ends (``t + tau_switch``), ties going to the earliest-starting burst;
    * if nothing is catchable yet, move early to the next future burst and wait there.

    A burst counts as caught only once the receiver's own observation confirms it, so a missed
    detection (Pd < 1) makes the scheduler try again.

    Attributes
    ----------
    needs_truth : bool
        Always ``True`` (cheating / upper bound).
    """

    needs_truth = True  # CHEATING / upper bound

    def __init__(self) -> None:
        super().__init__()
        self._truth: Truth | None = None
        self._bursts: list[tuple[int, int, int]] = []  # (t_start, t_end, channel) sorted by t_start
        self._caught: list[bool] = []
        self._head = 0
        self._t = 0
        self._current = 0
        self._last_a = -1
        self._tau = 0

    def set_truth(self, truth: Truth) -> None:
        """Hand the hidden truth to the scheduler; must be called before :meth:`reset`.

        Parameters
        ----------
        truth : Truth
            Episode ground truth from ``env.get_truth()``.  ``truth.S`` (the ``(N, T)`` occupancy
            grid) and the per-emitter bursts segmented from it drive every decision.
        """
        self._truth = truth

    def reset(self, spec: MissionSpec, seed: int) -> None:
        """Segment the stored truth into bursts and restart on ``spec.initial_channel``.

        Parameters
        ----------
        spec : MissionSpec
            Mission parameters; ``spec.initial_channel`` is the starting channel and
            ``spec.tau_switch`` the retune blindness used for the look-ahead.
        seed : int
            Episode seed, forwarded to :meth:`common.protocol.BaseScheduler.reset`.  Unused beyond
            that: the policy is deterministic given the truth.

        Raises
        ------
        RuntimeError
            If :meth:`set_truth` has not been called.
        """
        super().reset(spec, seed)
        if self._truth is None:
            raise RuntimeError("Clairvoyant needs set_truth(env.get_truth()) before reset()")
        bursts = segment_bursts(self._truth)
        self._bursts = [(b.t_start, b.t_end, b.channel) for b in bursts]
        self._caught = [False] * len(self._bursts)
        self._head = 0
        self._t = 0
        self._current = spec.initial_channel
        self._last_a = -1
        self._tau = spec.tau_switch
        self._S = self._truth.S

    def _mark_caught(self, channel: int, t: int) -> None:
        """Flag every uncaught burst on ``channel`` that covers slot ``t`` as caught.

        Parameters
        ----------
        channel : int
            Channel the receiver was listening to.
        t : int
            Slot at which the confirming observation was made.
        """
        B, caught = self._bursts, self._caught
        i = self._head
        while i < len(B) and B[i][0] <= t:
            if not caught[i] and B[i][2] == channel and B[i][1] >= t:
                caught[i] = True
            i += 1

    def select_action(self, obs: int, info: dict) -> int:
        """Pick the next channel from the truth, first crediting the previous slot's detection.

        A non-blind ``obs == 1`` at slot ``t - 1`` on a truly occupied channel marks the matching
        burst(s) as caught.  Caught and expired bursts are then dropped from the head of the burst
        list and the greedy rule described in the class docstring chooses the channel.

        Parameters
        ----------
        obs : int
            Previous observation ``O_{t-1}`` (0 at the first call).
        info : dict
            Sanitised info dict from the previous step; only ``info["blind"]`` is read.

        Returns
        -------
        int
            Channel index ``a_t`` in ``[0, N)``.
        """
        t = self._t
        if t > 0 and obs == 1 and not info.get("blind", False) and self._S[self._last_a, t - 1]:
            self._mark_caught(self._last_a, t - 1)
        B, caught = self._bursts, self._caught
        # drop caught / expired bursts from the head
        while self._head < len(B) and (caught[self._head] or B[self._head][1] < t):
            self._head += 1
        best_time, best_chan = None, None
        stay = False
        arrive = t + self._tau
        i = self._head
        while i < len(B):
            ts, te, ch = B[i]
            if ts > arrive:
                if best_time is None:  # nothing catchable now: go early to the next future burst
                    best_time, best_chan = ts, ch
                break
            if not caught[i] and te >= t:
                if ch == self._current and ts <= t:
                    stay = True
                    break
                if te >= arrive:
                    cand = max(arrive, ts)
                    if best_time is None or cand < best_time:
                        best_time, best_chan = cand, ch
            i += 1
        if not stay and best_chan is not None:
            self._current = best_chan
        self._last_a = self._current
        self._t = t + 1
        return int(self._current)
