"""Every metric the judges will check (contract C8, theory guide Part 6).

``compute_metrics(truth, hist, predictions=(), upto_t=None, delta_guard=1)``
returns a :class:`MetricsResult`; ``to_dict()`` is flat.  Every zero
denominator yields ``NaN``, never an exception.

Definitions
-----------
* ``s_at = S[a_t, t]``; tp/fp/fn/tn on non-blind dwells only.
* ``sensor_pd = tp/(tp+fn)`` (blind dwells excluded);
  ``effective_pd = tp / #(s_at == 1)`` (blind dwells on an occupied cell count as misses);
  ``sensor_pfa = fp/(fp+tn)``.
* A burst is *caught* iff there is a ``t`` in ``[t_start, t_end]`` with ``a_t == channel``
  and ``tp[t]`` (a blind dwell or a false alarm never catches).
  ``poi = n_caught / n_bursts`` where ``n_bursts`` counts bursts that have started
  before ``t`` (at ``t == T`` that is every burst); by type via ``Burst.emitter_type``,
  ``NaN`` for absent types.
* ``poi_time = #{(m,t): S_by_emitter[m,a_t,t] == 1 and tp[t]} / S_by_emitter[:, :, :t].sum()``.
* ``tti = t_first_tp - t_start`` over caught bursts (mean and median).
* ``intercept_rate_per_dwell = n_caught / t``; ``per_s = n_caught / (t * dwell_s)``.
* ``discovery_ratio = #emitters with >= 1 tp / M``.
* Predictions: only ``status == "issued"`` are scored; correct iff any
  ``S[c, t_pred-delta_guard .. t_pred+delta_guard] == 1``; error is
  ``|t_pred - nearest burst start on c|`` averaged over predictions whose channel
  has >= 1 burst; ``t_pred`` outside ``[0, T)`` is incorrect and excluded from the
  error; dropped predictions are counted only in ``n_dropped_predictions``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable

import numpy as np

from common.types import EMITTER_TYPES, Burst, History, Prediction, Truth
from oracle.truth_tracker import dwell_ledger, segment_bursts

NAN = float("nan")

METRIC_KEYS = (
    "sensor_pd", "effective_pd", "sensor_pfa", "n_tp", "n_fp", "n_fn", "n_tn", "n_blind",
    "poi", "poi_periodic", "poi_agile", "poi_scanning", "poi_time", "n_bursts", "n_caught",
    "mean_tti", "median_tti", "intercept_rate_per_dwell", "intercept_rate_per_s", "discovery_ratio",
    "n_predictions", "n_dropped_predictions", "correct_pred_pct", "avg_pred_time_error",
    "total_reward", "mean_reward", "blind_fraction", "switch_rate", "n_switches",
    "us_per_decision_mean", "us_per_decision_p99",
)


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else NAN


@dataclass
class MetricsResult:
    # detection
    sensor_pd: float = NAN
    effective_pd: float = NAN
    sensor_pfa: float = NAN
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    n_tn: int = 0
    n_blind: int = 0
    # intercept
    poi: float = NAN
    poi_periodic: float = NAN
    poi_agile: float = NAN
    poi_scanning: float = NAN
    poi_time: float = NAN
    n_bursts: int = 0
    n_caught: int = 0
    mean_tti: float = NAN
    median_tti: float = NAN
    intercept_rate_per_dwell: float = NAN
    intercept_rate_per_s: float = NAN
    discovery_ratio: float = NAN
    # predictions
    n_predictions: int = 0
    n_dropped_predictions: int = 0
    correct_pred_pct: float = NAN
    avg_pred_time_error: float = NAN
    # reward / behaviour
    total_reward: float = 0.0
    mean_reward: float = NAN
    blind_fraction: float = NAN
    switch_rate: float = NAN
    n_switches: int = 0
    # filled by eval
    us_per_decision_mean: float = NAN
    us_per_decision_p99: float = NAN
    # extras (defaulted fields may be added freely)
    n_dwells: int = 0
    n_emitters: int = 0
    n_bursts_total: int = 0
    caught_bursts: tuple[int, ...] = field(default_factory=tuple, repr=False)  # indices into segment_bursts()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("caught_bursts", None)
        return d

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls) if f.name != "caught_bursts")


def _score_predictions(truth: Truth, bursts: list[Burst], predictions: Iterable[Prediction],
                       delta_guard: int) -> tuple[int, int, float, float]:
    S = truth.S
    N, T = S.shape
    starts_by_channel: dict[int, np.ndarray] = {}
    for b in bursts:
        starts_by_channel.setdefault(b.channel, []).append(b.t_start)  # type: ignore[arg-type]
    starts_by_channel = {c: np.asarray(sorted(v)) for c, v in starts_by_channel.items()}
    n_issued = n_dropped = n_correct = 0
    errors: list[float] = []
    for p in predictions:
        status = getattr(p, "status", "issued")
        if status == "dropped":
            n_dropped += 1
            continue
        if status != "issued":
            continue
        n_issued += 1
        c, tp_ = int(p.channel), int(p.t_pred)
        if not (0 <= c < N) or not (0 <= tp_ < T):
            continue  # incorrect, excluded from the error average
        lo, hi = max(0, tp_ - delta_guard), min(T - 1, tp_ + delta_guard)
        if S[c, lo:hi + 1].any():
            n_correct += 1
        st = starts_by_channel.get(c)
        if st is not None and len(st):
            errors.append(float(np.min(np.abs(st - tp_))))
    pct = 100.0 * n_correct / n_issued if n_issued else NAN
    err = float(np.mean(errors)) if errors else NAN
    return n_issued, n_dropped, pct, err


def compute_metrics(truth: Truth, hist: History, predictions: Iterable[Prediction] = (),
                    upto_t: int | None = None, delta_guard: int = 1, dwell_s: float = 1e-3) -> MetricsResult:
    """Compute every metric over the first ``t`` executed slots (``t = hist.t`` or ``upto_t``)."""
    led = dwell_ledger(truth, hist, upto_t)
    t = led["t"]
    tp, fp, fn, tn = led["tp"], led["fp"], led["fn"], led["tn"]
    blind, switched, s_at, a = led["blind"], led["switched"], led["s_at"], led["actions"]
    n_tp, n_fp, n_fn, n_tn = int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum())
    n_blind, n_switches = int(blind.sum()), int(switched.sum())
    S_by = truth.S_by_emitter
    M, N, T = S_by.shape

    res = MetricsResult(
        n_tp=n_tp, n_fp=n_fp, n_fn=n_fn, n_tn=n_tn, n_blind=n_blind, n_switches=n_switches,
        n_dwells=t, n_emitters=M,
        sensor_pd=_ratio(n_tp, n_tp + n_fn),
        effective_pd=_ratio(n_tp, int(s_at.sum())),
        sensor_pfa=_ratio(n_fp, n_fp + n_tn),
        blind_fraction=_ratio(n_blind, t),
        switch_rate=_ratio(n_switches, t),
    )
    rewards = hist.rewards[:t].astype(np.float64)
    res.total_reward = float(rewards.sum())
    res.mean_reward = _ratio(res.total_reward, t)

    # --- bursts -----------------------------------------------------------
    bursts = segment_bursts(truth)
    res.n_bursts_total = len(bursts)
    # tp_by_channel[n, t] = a_t == n and tp[t]
    tp_idx = np.nonzero(tp)[0]
    tp_by_channel = np.zeros((N, t), dtype=bool)
    if len(tp_idx):
        tp_by_channel[a[tp_idx], tp_idx] = True
    caught: list[int] = []
    ttis: list[int] = []
    n_by_type = {k: 0 for k in EMITTER_TYPES}
    c_by_type = {k: 0 for k in EMITTER_TYPES}
    n_started = 0
    for i, b in enumerate(bursts):
        if b.t_start >= t:
            continue
        n_started += 1
        n_by_type[b.emitter_type] = n_by_type.get(b.emitter_type, 0) + 1
        seg = tp_by_channel[b.channel, b.t_start:min(b.t_end, t - 1) + 1]
        if seg.any():
            caught.append(i)
            ttis.append(int(np.argmax(seg)))
            c_by_type[b.emitter_type] = c_by_type.get(b.emitter_type, 0) + 1
    res.n_bursts = n_started
    res.n_caught = len(caught)
    res.caught_bursts = tuple(caught)
    res.poi = _ratio(res.n_caught, res.n_bursts)
    res.poi_periodic = _ratio(c_by_type["periodic"], n_by_type["periodic"])
    res.poi_agile = _ratio(c_by_type["agile"], n_by_type["agile"])
    res.poi_scanning = _ratio(c_by_type["scanning"], n_by_type["scanning"])
    if ttis:
        res.mean_tti = float(np.mean(ttis))
        res.median_tti = float(np.median(ttis))
    res.intercept_rate_per_dwell = _ratio(res.n_caught, t)
    res.intercept_rate_per_s = _ratio(res.n_caught, t * dwell_s) if t and dwell_s > 0 else NAN

    # --- time-coverage POI and discovery ------------------------------------
    if t > 0 and M > 0:
        hit_cells = S_by[:, a[tp_idx], tp_idx] != 0 if len(tp_idx) else np.zeros((M, 0), dtype=bool)  # (M, n_tp)
        res.poi_time = _ratio(int(hit_cells.sum()), int(S_by[:, :, :t].sum()))
        res.discovery_ratio = _ratio(int(hit_cells.any(axis=1).sum()), M)
    elif M > 0:
        res.poi_time = _ratio(0, int(S_by[:, :, :t].sum()))
        res.discovery_ratio = 0.0

    # --- predictions ----------------------------------------------------------
    n_issued, n_dropped, pct, err = _score_predictions(truth, bursts, predictions, int(delta_guard))
    res.n_predictions = n_issued
    res.n_dropped_predictions = n_dropped
    res.correct_pred_pct = pct
    res.avg_pred_time_error = err
    return res


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)
