"""Burst segmentation and the per-dwell ledger (contract C8).

Everything here reads ``Truth`` and ``History`` only; nothing is imported from
``env/`` so the oracle stays a black box relative to the physics.

Two helpers are exposed:

* :func:`segment_bursts` turns the hidden per-emitter occupancy tensor into a
  sorted list of ``Burst`` records, one per maximal run of occupied slots.
* :func:`dwell_ledger` scores every executed dwell against the hidden truth and
  returns the boolean confusion columns that the rest of the oracle aggregates.
"""
from __future__ import annotations

import numpy as np

from common.types import Burst, History, Truth


def segment_bursts(truth: Truth) -> list[Burst]:
    """Maximal runs of 1s per (emitter, channel) in ``S_by_emitter``, sorted by ``t_start``.

    Every contiguous run of occupied slots for a single (emitter, channel) pair
    becomes one ``Burst``.  ``t_end`` is inclusive.  ``emitter_type`` is filled
    from ``truth.emitter_types``.

    Parameters
    ----------
    truth : Truth
        Hidden truth.  Only ``S_by_emitter`` (shape ``(M, N, T)``, int8) and
        ``emitter_types`` (length ``M``) are read.

    Returns
    -------
    list[Burst]
        One record per maximal run, sorted by ``(t_start, emitter, channel)``.
        Empty when no emitter ever transmits.

    Notes
    -----
    Runs are located by diffing a zero-padded copy of each channel row, so a
    burst that starts at slot 0 or ends at slot ``T - 1`` is closed correctly.
    Only channels with at least one occupied slot are scanned.
    """
    S_by = truth.S_by_emitter
    M, N, T = S_by.shape
    out: list[Burst] = []
    pad = np.zeros(T + 2, dtype=np.int8)
    for m in range(M):
        etype = truth.emitter_types[m]
        Sm = S_by[m]
        for n in np.nonzero(Sm.any(axis=1))[0]:
            pad[1:T + 1] = Sm[n]
            d = np.diff(pad)
            starts = np.nonzero(d == 1)[0]
            ends = np.nonzero(d == -1)[0] - 1
            for s, e in zip(starts.tolist(), ends.tolist()):
                out.append(Burst(int(m), etype, int(n), int(s), int(e)))
    out.sort(key=lambda b: (b.t_start, b.emitter, b.channel))
    return out


def dwell_ledger(truth: Truth, hist: History, upto_t: int | None = None) -> dict[str, np.ndarray]:
    """Per-dwell confusion ledger over the first ``t`` executed slots.

    Compares what the scheduler observed on each dwelt channel with the hidden
    truth at that slot and classifies every non-blind dwell into exactly one of
    tp / fp / fn / tn.

    Parameters
    ----------
    truth : Truth
        Hidden truth; only ``S`` (shape ``(N, T)``) is read.
    hist : History
        Episode history written by the env.  ``actions``, ``obs``, ``blind`` and
        ``switched`` are read along with ``hist.t``.
    upto_t : int or None, optional
        Number of leading slots to include.  ``None`` (default) uses every
        executed slot (``hist.t``); a larger value is clipped to ``hist.t`` and
        a negative value yields an empty ledger.

    Returns
    -------
    dict[str, np.ndarray]
        Bool arrays of length ``t``: ``tp``, ``fp``, ``fn``, ``tn``, ``blind``,
        ``switched`` plus ``s_at`` (truth on the dwelt cell) and ``obs``, and
        ``actions`` (int64 channel indices).  ``fa`` is an alias of ``fp`` (the
        same array object).  The scalar ``t`` (number of slots covered) is also
        stored under key ``"t"``.

    Notes
    -----
    tp/fp/fn/tn are defined on non-blind dwells only: a blind dwell is excluded
    from all four, so ``tp.sum() + fp.sum() + fn.sum() + tn.sum() + blind.sum()``
    equals ``t``.
    """
    t = hist.t if upto_t is None else min(int(upto_t), hist.t)
    t = max(0, t)
    a = hist.actions[:t].astype(np.int64)
    o = hist.obs[:t].astype(bool)
    blind = hist.blind[:t].astype(bool)
    switched = hist.switched[:t].astype(bool)
    idx = np.arange(t)
    s_at = truth.S[a, idx].astype(bool) if t > 0 else np.zeros(0, dtype=bool)
    nb = ~blind
    tp = nb & s_at & o
    fp = nb & ~s_at & o
    fn = nb & s_at & ~o
    tn = nb & ~s_at & ~o
    return {
        "tp": tp, "fp": fp, "fa": fp, "fn": fn, "tn": tn,
        "blind": blind, "switched": switched,
        "s_at": s_at, "obs": o, "actions": a, "t": t,
    }
