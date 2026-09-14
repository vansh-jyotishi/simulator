"""Baseline scheduler tests (S17, S21)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from baselines.clairvoyant import Clairvoyant
from baselines.round_robin import RoundRobin
from baselines.weighted_priority import WeightedPriority
from common.protocol import Scheduler
from common.types import MissionSpec
from env.receiver_model import PassThroughReceiver
from oracle.metrics_engine import compute_metrics
from tests.fakes import play

MVP = "scenarios/mvp_periodic.json"


def _spec(N=50, tau=1, weights=None):
    return MissionSpec(N=N, T=5000, dwell_s=1e-3, tau_switch=tau, pfa=1e-3, initial_channel=0, intel_weights=weights)


def _actions(sched, spec, n, obs_stream=None):
    sched.reset(spec, 0)
    out = []
    for t in range(n):
        o = 0 if obs_stream is None else int(obs_stream[t])
        out.append(sched.select_action(o, {"t": t - 1, "next_t": t, "blind": False}))
    return out


def test_protocol_shape():
    for cls in (RoundRobin, WeightedPriority, Clairvoyant):
        s = cls()  # no required constructor args
        assert isinstance(s, Scheduler)
        assert s.predictions() == []
    assert RoundRobin.needs_truth is False and WeightedPriority.needs_truth is False
    assert Clairvoyant.needs_truth is True


def test_round_robin_default_dwell_is_tau_plus_one():
    assert _actions(RoundRobin(), _spec(tau=1), 8) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert _actions(RoundRobin(), _spec(tau=0), 6) == [0, 1, 2, 3, 4, 5]
    assert _actions(RoundRobin(), _spec(tau=2), 7) == [0, 0, 0, 1, 1, 1, 2]
    assert _actions(RoundRobin(dwell_steps=1), _spec(tau=1), 4) == [0, 1, 2, 3]
    a = _actions(RoundRobin(), _spec(N=50, tau=1), 300)
    assert a[100] == 0 and a[99] == 49  # sweep period N*(tau+1) = 100


def test_weighted_priority_uniform_equals_round_robin_for_any_obs_stream():
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 2, 1000)
    rr = _actions(RoundRobin(), _spec(), 1000, obs)
    wp = _actions(WeightedPriority(), _spec(), 1000, obs)
    wp0 = _actions(WeightedPriority(), _spec(), 1000, None)
    assert rr == wp == wp0


def test_weighted_priority_visit_counts_match_weights():
    N = 10
    w = (5.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0)
    sched = WeightedPriority(dwell_steps=1)
    acts = np.asarray(_actions(sched, _spec(N=N, weights=w), 10_000))
    frac = np.bincount(acts, minlength=N) / len(acts)
    target = np.asarray(w) / sum(w)
    assert np.all(np.abs(frac - target) < 0.02)
    assert set(acts.tolist()) == set(range(N))
    # each pick held for dwell_steps
    sched2 = WeightedPriority()
    acts2 = _actions(sched2, _spec(N=N, tau=1, weights=w), 40)
    assert all(acts2[2 * i] == acts2[2 * i + 1] for i in range(20))


def test_weighted_priority_every_channel_within_n_dwell_slots_uniform():
    N = 50
    acts = _actions(WeightedPriority(), _spec(N=N, tau=1), N * 2)
    assert set(acts) == set(range(N))


def test_weighted_priority_open_loop_no_rng():
    s = WeightedPriority()
    a = _actions(s, _spec(N=8, weights=(1, 2, 3, 4, 5, 6, 7, 8)), 500, np.ones(500))
    b = _actions(WeightedPriority(), _spec(N=8, weights=(1, 2, 3, 4, 5, 6, 7, 8)), 500, np.zeros(500))
    assert a == b


def test_clairvoyant_catches_every_burst_on_single_radar():
    sc = {"name": "one", "N": 6, "T": 1500, "receiver": {"tau_switch": 1, "pfa": 1e-3},
          "emitters": [{"type": "periodic", "name": "r", "channel": 2, "onoff": [3, 2, 9], "snr_db": 10.0}]}
    env = play(sc, 0, Clairvoyant(), receiver=PassThroughReceiver())
    m = compute_metrics(env.get_truth(), env.get_history())
    assert m.poi == 1.0 and m.n_switches == 1


def test_clairvoyant_beats_round_robin_on_mvp():
    for seed in (0, 1, 2):
        env_c = play(MVP, seed, Clairvoyant())
        env_r = play(MVP, seed, RoundRobin())
        env_w = play(MVP, seed, WeightedPriority())
        mc = compute_metrics(env_c.get_truth(), env_c.get_history())
        mr = compute_metrics(env_r.get_truth(), env_r.get_history())
        mw = compute_metrics(env_w.get_truth(), env_w.get_history())
        assert mc.poi >= mr.poi and mc.poi >= mw.poi
        assert mc.poi > 0.5


def test_clairvoyant_requires_truth():
    with pytest.raises(RuntimeError):
        Clairvoyant().reset(_spec(), 0)


def test_blind_zone_round_robin_zero_weighted_positive():
    raw = json.load(open("scenarios/blind_zone.json"))
    env_r = play(raw, 0, RoundRobin())
    env_w = play(raw, 0, WeightedPriority())
    mr = compute_metrics(env_r.get_truth(), env_r.get_history())
    mw = compute_metrics(env_w.get_truth(), env_w.get_history())
    assert mr.poi == 0.0 and mr.n_bursts > 0
    assert mw.poi > 0.0
