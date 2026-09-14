"""Episode runner and scheduler registry (contract C9 plumbing).

``run_episode`` is the only place a scheduler meets the env: it applies
``sanitize_info`` before every ``select_action`` call, times each decision
with ``perf_counter_ns`` and computes the metrics from the oracle.  It imports
nothing from ``env/`` except ``make_env``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

from common.protocol import sanitize_info
from common.types import MissionSpec
from env.spectrum_env import make_env
from oracle.metrics_engine import MetricsResult, compute_metrics

from baselines.clairvoyant import Clairvoyant
from baselines.round_robin import RoundRobin
from baselines.weighted_priority import WeightedPriority

# Built-in baselines.  Only classes listed here may carry needs_truth=True.
REGISTRY: dict[str, type] = {
    "round_robin": RoundRobin,
    "weighted_priority": WeightedPriority,
    "clairvoyant": Clairvoyant,
}
BUILTIN_TRUTH_CLASSES = (Clairvoyant,)


def discover_registry() -> dict[str, type]:
    """Built-ins plus ``schedulers/__init__.py::REGISTRY`` (ImportError ignored)."""
    reg = dict(REGISTRY)
    try:
        mod = importlib.import_module("schedulers")
        extra = getattr(mod, "REGISTRY", {})
        for k, v in dict(extra).items():
            if k in reg:
                raise ValueError(f"schedulers.REGISTRY key {k!r} collides with a built-in baseline")
            reg[k] = v
    except ImportError:
        pass
    return reg


def load_extra(spec: str) -> tuple[str, type]:
    """Parse ``alias=pkg.mod:Class`` and import it.  ``needs_truth=True`` classes are refused."""
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"--extra must look like alias=pkg.mod:Class, got {spec!r}")
    alias, target = spec.split("=", 1)
    mod_name, cls_name = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    if getattr(cls, "needs_truth", False):
        raise ValueError(f"--extra {alias}: {target} declares needs_truth=True and is refused "
                         "(truth access is reserved for built-in baselines)")
    return alias.strip(), cls


@dataclass
class EpisodeResult:
    label: str
    scenario: str
    seed: int
    episode: int
    metrics: MetricsResult
    spec: MissionSpec
    n_steps: int
    env: Any = field(default=None, repr=False)  # kept only when keep_env=True
    predictions: list = field(default_factory=list, repr=False)

    def row(self) -> dict:
        d = {"scheduler": self.label, "scenario": self.scenario, "seed": self.seed, "episode": self.episode}
        d.update(self.metrics.to_dict())
        return d


def run_episode(scenario, seed: int, scheduler, *, env_factory: Callable = make_env, receiver=None,
                label: str | None = None, episode: int = 0, keep_env: bool = False,
                delta_guard: int = 1, scheduler_seed: int | None = None) -> EpisodeResult:
    """Run one full episode of ``scheduler`` on ``(scenario, seed)`` and score it.

    ``scheduler_seed`` defaults to ``seed``; the CLI offsets it per episode so
    repeated episodes re-roll the scheduler's randomness while the env truth
    and noise stay paired.
    """
    env = env_factory(scenario, seed, receiver) if receiver is not None else env_factory(scenario, seed)
    obs, info = env.reset()
    spec = env.mission_spec
    if getattr(scheduler, "needs_truth", False):
        if not isinstance(scheduler, BUILTIN_TRUTH_CLASSES):
            raise TypeError(f"{type(scheduler).__name__} declares needs_truth=True but is not a built-in baseline")
        scheduler.set_truth(env.get_truth())
    scheduler.reset(spec, seed if scheduler_seed is None else int(scheduler_seed))
    T = spec.T
    timings = np.empty(T, dtype=np.int64)
    t = 0
    while True:
        sinfo = sanitize_info(info)
        t0 = perf_counter_ns()
        a = scheduler.select_action(obs, sinfo)
        timings[t] = perf_counter_ns() - t0
        obs, r, term, trunc, info = env.step(a)
        t += 1
        if trunc:
            break
    preds = list(scheduler.predictions() or [])
    m = compute_metrics(env.get_truth(), env.get_history(), preds, delta_guard=delta_guard, dwell_s=spec.dwell_s)
    us = timings[:t] / 1000.0
    m.us_per_decision_mean = float(us.mean())
    m.us_per_decision_p99 = float(np.percentile(us, 99))
    name = getattr(env.cfg, "name", str(scenario))
    return EpisodeResult(label=label or type(scheduler).__name__, scenario=name, seed=int(seed), episode=episode,
                         metrics=m, spec=spec, n_steps=t, env=env if keep_env else None, predictions=preds)
