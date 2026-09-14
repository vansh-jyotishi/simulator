"""Burst segmentation and the per-dwell ledger (contract C8).

Everything here reads ``Truth`` and ``History`` only; nothing is imported from
``env/`` so the oracle stays a black box relative to the physics.
"""
from __future__ import annotations

import numpy as np

from common.types import Burst, History, Truth


def segment_bursts(truth: Truth) -> list[Burst]:
    """Maximal runs of 1s per (emitter, channel) in ``S_by_emitter``, sorted by ``t_start``.

    ``t_end`` is inclusive.  ``emitter_type`` is filled from ``truth.emitter_types``.
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

    Returns bool arrays of length ``t``: ``tp fp fn tn blind switched`` plus
    ``s_at`` (truth on the dwelt cell), ``obs`` and ``actions``.  ``fa`` is an
    alias of ``fp``.  tp/fp/fn/tn are defined on non-blind dwells only.
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
