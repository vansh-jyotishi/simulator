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

Notes
-----
Public API: :func:`compute_metrics`, :class:`MetricsResult`, :func:`is_nan` and the
``METRIC_KEYS`` tuple (the judged subset of :meth:`MetricsResult.keys`).
``us_per_decision_mean`` / ``us_per_decision_p99`` are never set here; the evaluation
harness fills them after timing the agent.
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
    """Return ``num / den`` as a float, or ``NaN`` when ``den`` is zero (never raises)."""
    return float(num) / float(den) if den else NAN


@dataclass
class MetricsResult:
    """Flat container for every metric produced by :func:`compute_metrics`.

    Ratios default to ``NaN`` and counts to ``0`` so a partially filled result is
    still well-formed.  Defaulted fields may be added freely; only ``caught_bursts``
    is excluded from :meth:`to_dict` and :meth:`keys`.

    Attributes
    ----------
    sensor_pd : float
        ``tp / (tp + fn)`` over non-blind dwells.
    effective_pd : float
        ``tp / #(s_at == 1)``; blind dwells on an occupied cell count as misses.
    sensor_pfa : float
        ``fp / (fp + tn)`` over non-blind dwells.
    n_tp, n_fp, n_fn, n_tn : int
        Dwell-level confusion counts; blind dwells are excluded from all four.
    n_blind : int
        Number of blind dwells (the receiver was retuning and observed nothing).
    poi : float
        Probability of intercept, ``n_caught / n_bursts``.
    poi_periodic, poi_agile, poi_scanning : float
        POI restricted to bursts of that ``Burst.emitter_type``; ``NaN`` for absent types.
    poi_time : float
        Time-coverage POI: fraction of occupied ``(emitter, channel, t)`` cells within
        ``[0, t)`` whose dwell was a true positive.
    n_bursts : int
        Number of bursts that have started before ``t`` (every burst once ``t == T``).
    n_caught : int
        Number of those bursts caught by at least one true positive dwell.
    mean_tti, median_tti : float
        Mean / median time-to-intercept ``t_first_tp - t_start`` over caught bursts.
    intercept_rate_per_dwell : float
        ``n_caught / t``.
    intercept_rate_per_s : float
        ``n_caught / (t * dwell_s)``.
    discovery_ratio : float
        Fraction of the ``M`` emitters with at least one true positive dwell.
    n_predictions : int
        Number of predictions with ``status == "issued"`` (the only ones scored).
    n_dropped_predictions : int
        Number of predictions with ``status == "dropped"``; counted but never scored.
    correct_pred_pct : float
        Percentage of issued predictions with an occupied cell within ``delta_guard``
        of ``t_pred`` on the predicted channel.
    avg_pred_time_error : float
        Mean ``|t_pred - nearest burst start on c|`` over in-range predictions whose
        channel has at least one burst.
    total_reward : float
        Sum of ``hist.rewards[:t]``.
    mean_reward : float
        ``total_reward / t``.
    blind_fraction : float
        ``n_blind / t``.
    switch_rate : float
        ``n_switches / t``.
    n_switches : int
        Number of dwells flagged ``switched`` (channel changed from the previous dwell).
    us_per_decision_mean, us_per_decision_p99 : float
        Agent decision latency in microseconds; left ``NaN`` here and filled by the
        evaluation harness.
    n_dwells : int
        Number of executed slots scored, i.e. ``t``.
    n_emitters : int
        ``M``, the number of emitters in the truth.
    n_bursts_total : int
        Total number of bursts in the truth regardless of ``t``.
    caught_bursts : tuple of int
        Indices into :func:`oracle.truth_tracker.segment_bursts` of the caught bursts;
        omitted from :meth:`to_dict` and from ``repr``.
    """
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
        """Return the metrics as a flat ``dict`` keyed by field name.

        Returns
        -------
        dict
            One entry per dataclass field except ``caught_bursts``; values are plain
            ``float`` / ``int`` scalars (``NaN`` for undefined ratios).
        """
        d = asdict(self)
        d.pop("caught_bursts", None)
        return d

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Return the field names in the order :meth:`to_dict` emits them.

        Returns
        -------
        tuple of str
            Every dataclass field name except ``caught_bursts``, in declaration order.
        """
        return tuple(f.name for f in fields(cls) if f.name != "caught_bursts")


def _score_predictions(truth: Truth, bursts: list[Burst], predictions: Iterable[Prediction],
                       delta_guard: int) -> tuple[int, int, float, float]:
    """Score issued predictions against the truth occupancy grid.

    Parameters
    ----------
    truth : Truth
        Ground truth; ``truth.S`` of shape ``(N, T)`` is consulted.
    bursts : list of Burst
        Output of :func:`oracle.truth_tracker.segment_bursts`, used for burst starts per channel.
    predictions : iterable of Prediction
        Predictions to score; ``status`` defaults to ``"issued"`` when the object lacks it.
    delta_guard : int
        Half-width of the window ``[t_pred - delta_guard, t_pred + delta_guard]``.

    Returns
    -------
    tuple of (int, int, float, float)
        ``(n_issued, n_dropped, correct_pct, mean_abs_time_error)``; the last two are
        ``NaN`` when no prediction was issued / no error could be measured.

    Notes
    -----
    A prediction with ``t_pred`` outside ``[0, T)`` or an invalid channel is counted
    as issued and incorrect but excluded from the time-error average.
    """
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
    """Compute every metric over the first ``t`` executed slots (``t = hist.t`` or ``upto_t``).

    Parameters
    ----------
    truth : Truth
        Ground truth with ``S`` of shape ``(N, T)`` and ``S_by_emitter`` of shape ``(M, N, T)``.
    hist : History
        Episode history written by the env; ``actions``, ``obs``, ``rewards``, ``blind``,
        ``switched`` and ``hist.t`` are read.
    predictions : iterable of Prediction, optional
        Predictions to score; only those with ``status == "issued"`` count.  Default is empty.
    upto_t : int or None, optional
        Score only the first ``upto_t`` slots; ``None`` (default) uses ``hist.t``.
    delta_guard : int, optional
        Half-width of the window around ``t_pred`` within which an occupied cell makes a
        prediction correct.  Default ``1``.
    dwell_s : float, optional
        Dwell duration in seconds, used only for ``intercept_rate_per_s``.  Default ``1e-3``.

    Returns
    -------
    MetricsResult
        Every metric for the first ``t`` slots; zero denominators give ``NaN`` rather
        than raising.

    Notes
    -----
    See the module docstring for the exact definition of each metric.  The
    ``us_per_decision_*`` fields are left ``NaN`` for the evaluation harness to fill.
    """
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
    """Return ``True`` iff ``x`` is a ``float`` holding ``NaN``.

    Parameters
    ----------
    x : float
        Value to test; non-``float`` inputs (e.g. ``int`` counts) are never ``NaN``.

    Returns
    -------
    bool
        ``True`` when ``x`` is a ``float`` and ``math.isnan(x)``, else ``False``.
    """
    return isinstance(x, float) and math.isnan(x)
