"""Scheduler protocol conformance (S22).  Run it before asking for a merge:

    python -m pytest tests/test_protocol_conformance.py -q --scheduler schedulers.whittle_rmab:WhittleScheduler

Default: both baselines.  Checks 300 steps on the real env with
``scenarios/mvp_periodic.json``: actions are ints in ``[0, N)``, the same
seed gives the same actions, ``predictions()`` is well-formed,
``needs_truth`` is False, and mean / p99 microseconds per decision are printed.
"""
from __future__ import annotations

import importlib
from time import perf_counter_ns

import numpy as np

from common.protocol import Scheduler, sanitize_info
from common.types import INFO_KEYS, Prediction
from env.spectrum_env import make_env

MVP = "scenarios/mvp_periodic.json"
STEPS = 300


def _load(spec: str):
    mod, cls = spec.split(":", 1)
    return getattr(importlib.import_module(mod), cls)


def _drive(cls, seed: int):
    env = make_env(MVP, seed)
    obs, info = env.reset()
    sched = cls()  # constructor must take no required arguments
    assert isinstance(sched, Scheduler)
    assert getattr(sched, "needs_truth", False) is False, "candidates must not declare needs_truth=True"
    sched.reset(env.mission_spec, seed)
    N = env.mission_spec.N
    actions, us = [], []
    for _ in range(STEPS):
        sinfo = sanitize_info(info)
        assert set(sinfo) <= set(INFO_KEYS)
        t0 = perf_counter_ns()
        a = sched.select_action(obs, sinfo)
        us.append((perf_counter_ns() - t0) / 1000.0)
        assert isinstance(a, (int, np.integer)) and not isinstance(a, bool), f"action must be an int, got {type(a)}"
        assert 0 <= int(a) < N, f"action {a} outside [0, {N})"
        actions.append(int(a))
        obs, r, term, trunc, info = env.step(int(a))
        if trunc:
            break
    preds = sched.predictions()
    assert isinstance(preds, list)
    for p in preds:
        assert isinstance(p, Prediction) and p.status in ("issued", "dropped")
        assert isinstance(p.channel, int) and isinstance(p.t_pred, int) and isinstance(p.t_made, int)
    return actions, np.asarray(us)


def test_protocol_conformance(scheduler_spec):
    cls = _load(scheduler_spec)
    a1, us = _drive(cls, 0)
    a2, _ = _drive(cls, 0)
    assert a1 == a2, "same seed must give the same actions"
    print(f"\n  {scheduler_spec}: {len(a1)} steps, us/decision mean={us.mean():.2f} p99={np.percentile(us, 99):.2f}")
