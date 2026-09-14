"""Oracle tests: hand-built fixtures with known numbers (S6-S8, S15) and property tests on the real env (S16)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from common.types import Burst, History, Prediction, Truth
from env.receiver_model import PassThroughReceiver
from oracle.metrics_engine import METRIC_KEYS, MetricsResult, compute_metrics
from oracle.truth_tracker import dwell_ledger, segment_bursts
from tests.fakes import (PRED_80PCT, PRED_OFF_BY_2, PRED_PERFECT, PRED_THEORY_ERRORS, TINY_BURSTS, TINY_EXPECTED,
                         TINY_T, Scripted, StayPut, SwitchEvery, play, tiny_history, tiny_truth)

MVP = "scenarios/mvp_periodic.json"


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ---------------------------------------------------------------- fixtures / hand numbers


def test_metric_keys_present():
    d = MetricsResult().to_dict()
    for k in METRIC_KEYS:
        assert k in d


def test_tiny_bursts_equal_hand_list():
    bursts = segment_bursts(tiny_truth())
    assert [(b.emitter, b.emitter_type, b.channel, b.t_start, b.t_end) for b in bursts] == TINY_BURSTS


def test_run_split_by_one_zero_gives_two_bursts():
    S_by = np.zeros((1, 2, 6), dtype=np.int8)
    S_by[0, 1] = [1, 1, 0, 1, 0, 0]
    tr = Truth(S=S_by[0].copy(), S_by_emitter=S_by, SNR=np.zeros((2, 6), np.float32), E=np.zeros((2, 6), np.int16),
               U=np.zeros((2, 6), np.float32), emitter_names=("x",), emitter_types=("periodic",))
    assert segment_bursts(tr) == [Burst(0, "periodic", 1, 0, 1), Burst(0, "periodic", 1, 3, 3)]


def test_dwell_ledger_counts():
    led = dwell_ledger(tiny_truth(), tiny_history())
    assert led["t"] == TINY_T
    assert (led["tp"].sum(), led["fp"].sum(), led["fn"].sum(), led["tn"].sum(), led["blind"].sum()) == (4, 1, 1, 19, 5)
    assert np.array_equal(led["fa"], led["fp"])
    assert not np.any(led["tp"] & led["blind"])


def test_tiny_metrics_reproduce_hand_numbers():
    d = compute_metrics(tiny_truth(), tiny_history()).to_dict()
    for k, v in TINY_EXPECTED.items():
        assert abs(d[k] - v) < 1e-9, (k, d[k], v)
    assert math.isnan(d["poi_scanning"])
    assert d["n_predictions"] == 0 and math.isnan(d["correct_pred_pct"]) and math.isnan(d["avg_pred_time_error"])


@pytest.mark.parametrize("preds, n_issued, n_dropped, pct, err", [
    (PRED_PERFECT, 3, 0, 100.0, 0.0),
    (PRED_OFF_BY_2, 3, 1, 0.0, 2.0),
    (PRED_THEORY_ERRORS, 5, 0, 60.0, 3.0),
    (PRED_80PCT, 5, 1, 80.0, 1.6),
])
def test_prediction_scoring(preds, n_issued, n_dropped, pct, err):
    m = compute_metrics(tiny_truth(), tiny_history(), preds, delta_guard=1)
    assert m.n_predictions == n_issued and m.n_dropped_predictions == n_dropped
    assert abs(m.correct_pred_pct - pct) < 1e-9 and abs(m.avg_pred_time_error - err) < 1e-9


def test_prediction_delta_guard_widens():
    m = compute_metrics(tiny_truth(), tiny_history(), PRED_OFF_BY_2, delta_guard=2)
    assert m.correct_pred_pct == 100.0


def test_prediction_out_of_range_incorrect_and_excluded():
    preds = [Prediction(1, -1, 0, "issued"), Prediction(1, TINY_T, 0, "issued"), Prediction(9, 2, 0, "issued"),
             Prediction(0, 5, 0, "issued")]  # channel 0 has no bursts: incorrect, no error contribution
    m = compute_metrics(tiny_truth(), tiny_history(), preds)
    assert m.n_predictions == 4 and m.correct_pred_pct == 0.0 and math.isnan(m.avg_pred_time_error)


def test_zero_denominators_are_nan_not_exceptions():
    m = compute_metrics(tiny_truth(), tiny_history(0)).to_dict()
    for k in ("sensor_pd", "effective_pd", "sensor_pfa", "poi", "poi_time", "mean_tti", "blind_fraction",
              "switch_rate", "mean_reward", "intercept_rate_per_dwell"):
        assert math.isnan(m[k]), k
    assert m["n_bursts"] == 0 and m["n_caught"] == 0 and m["total_reward"] == 0.0
    # all-blind history
    h = tiny_history()
    hb = History(actions=h.actions, obs=np.zeros_like(h.obs), blind=np.ones_like(h.blind), switched=h.switched,
                 new_detect=np.zeros_like(h.new_detect), rewards=h.rewards, t=h.t)
    mb = compute_metrics(tiny_truth(), hb)
    assert math.isnan(mb.sensor_pd) and math.isnan(mb.sensor_pfa) and mb.effective_pd == 0.0
    assert mb.poi == 0.0 and mb.blind_fraction == 1.0 and math.isnan(mb.mean_tti)


@pytest.mark.parametrize("t", [0, 1, 5, 12, 20, 29, 30])
def test_upto_t_equals_truncated_history(t):
    a = compute_metrics(tiny_truth(), tiny_history(), upto_t=t).to_dict()
    b = compute_metrics(tiny_truth(), tiny_history(t)).to_dict()
    for k in a:
        assert _same(a[k], b[k]), (k, a[k], b[k])
    assert a["n_dwells"] == t


def test_upto_t_beyond_history_clamps():
    a = compute_metrics(tiny_truth(), tiny_history(10), upto_t=25).to_dict()
    b = compute_metrics(tiny_truth(), tiny_history(10)).to_dict()
    assert all(_same(a[k], b[k]) for k in a)


# ---------------------------------------------------------------- real env


def test_full_episode_burst_lengths_sum_to_truth():
    env = play(MVP, 0, StayPut(7))
    tr = env.get_truth()
    bursts = segment_bursts(tr)
    assert sum(b.t_end - b.t_start + 1 for b in bursts) == int(tr.S_by_emitter.sum())
    assert all(b.emitter_type == "periodic" for b in bursts)


def _rand_scenario(T=2000):
    raw = json.load(open("scenarios/train_default.json"))
    raw["T"] = T
    raw["name"] = "prop"
    return raw


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_properties_on_real_env(seed):
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 50, size=2000)
    actions[::7] = 7  # make sure the strong radar gets visited
    for sched in (Scripted(actions), SwitchEvery(3, (7, 20, 21, 33, 41))):
        env = play(_rand_scenario(), seed, sched)
        m = compute_metrics(env.get_truth(), env.get_history(), dwell_s=env.mission_spec.dwell_s)
        for k in ("sensor_pd", "sensor_pfa", "effective_pd", "poi", "poi_time", "blind_fraction", "switch_rate",
                  "discovery_ratio"):
            v = getattr(m, k)
            assert math.isnan(v) or 0.0 <= v <= 1.0, (k, v)
        assert m.n_caught <= m.n_bursts
        assert m.n_tp + m.n_fp + m.n_fn + m.n_tn + m.n_blind == 2000
        if not math.isnan(m.sensor_pd):
            assert m.effective_pd <= m.sensor_pd + 1e-12
        assert m.n_bursts == len(segment_bursts(env.get_truth()))
        assert abs(m.total_reward - float(env.get_history().rewards.sum())) < 1e-3
        assert m.n_switches == int(env.get_history().switched.sum())


def test_stayput_passthrough_on_single_radar_gives_poi_one_pfa_zero():
    sc = {"name": "one", "N": 6, "T": 1500, "receiver": {"tau_switch": 1, "pfa": 1e-3},
          "emitters": [{"type": "periodic", "name": "r", "channel": 2, "onoff": [3, 2, 9], "snr_db": 10.0}]}
    env = play(sc, 0, StayPut(2), receiver=PassThroughReceiver())
    m = compute_metrics(env.get_truth(), env.get_history())
    assert m.poi == 1.0 and m.poi_periodic == 1.0 and m.sensor_pd == 1.0 and m.effective_pd == 1.0
    assert m.sensor_pfa == 0.0 and m.mean_tti == 0.0 and m.n_blind == 0 and m.discovery_ratio == 1.0
    assert m.poi_time == 1.0
    empty = play(sc, 0, StayPut(4), receiver=PassThroughReceiver())
    me = compute_metrics(empty.get_truth(), empty.get_history())
    assert me.poi == 0.0 and math.isnan(me.sensor_pd) and me.sensor_pfa == 0.0 and me.discovery_ratio == 0.0


def test_upto_t_on_real_env_matches_live_history():
    env = play(MVP, 1, SwitchEvery(4, (7, 33)))
    tr, h = env.get_truth(), env.get_history()
    for t in (0, 10, 500, 1999, 2000):
        a = compute_metrics(tr, h, upto_t=t).to_dict()
        ht = History(actions=np.where(np.arange(2000) < t, h.actions, -1), obs=np.where(np.arange(2000) < t, h.obs, 0),
                     blind=h.blind & (np.arange(2000) < t), switched=h.switched & (np.arange(2000) < t),
                     new_detect=h.new_detect & (np.arange(2000) < t),
                     rewards=np.where(np.arange(2000) < t, h.rewards, 0).astype(np.float32), t=t)
        b = compute_metrics(tr, ht).to_dict()
        assert all(_same(a[k], b[k]) for k in a)
