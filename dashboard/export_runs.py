"""Export real missions from the simulator as compact JSON for the dashboard.

    python -m dashboard.export_runs

Writes ``dashboard/runs.json``: the ground-truth occupancy runs plus, for every
built-in strategy, the receiver's action track, a per-slot outcome code and the
final whole-mission metrics.  The dashboard page replays that file, so every
mark it draws comes from a real run of this simulator.
"""
from __future__ import annotations

import json
import os

import numpy as np

from baselines.clairvoyant import Clairvoyant
from baselines.round_robin import RoundRobin
from baselines.weighted_priority import WeightedPriority
from common.protocol import sanitize_info
from env.spectrum_env import make_env
from oracle.metrics_engine import compute_metrics
from oracle.truth_tracker import segment_bursts

SCENARIO = "scenarios/train_default.json"
SEED = 0
T_SHOW = 600  # slots exported for the animation

STRATS = [
    ("round_robin", "Round-robin sweep", RoundRobin),
    ("weighted_priority", "Priority-weighted sweep", WeightedPriority),
    ("clairvoyant", "Clairvoyant (cheats)", Clairvoyant),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def _play(cls):
    """Run one full episode of ``cls`` and return the finished env."""
    env = make_env(SCENARIO, SEED)
    obs, info = env.reset()
    s = cls()
    if getattr(s, "needs_truth", False):
        s.set_truth(env.get_truth())
    s.reset(env.mission_spec, SEED)
    while True:
        obs, r, term, trunc, info = env.step(s.select_action(obs, sanitize_info(info)))
        if trunc:
            break
    return env


def _f(x):
    """Round a metric, mapping NaN to None so JSON stays valid."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 5)


def main() -> int:
    out = {"scenario": SCENARIO, "seed": SEED, "T_show": T_SHOW, "runs": {}}
    first = True

    for key, label, cls in STRATS:
        env = _play(cls)
        truth, hist = env.get_truth(), env.get_history()
        spec = env.mission_spec
        m = compute_metrics(truth, hist, dwell_s=spec.dwell_s)

        if first:
            first = False
            out.update({
                "N": int(spec.N), "T": int(spec.T),
                "dwell_s": float(spec.dwell_s), "tau_switch": int(spec.tau_switch),
                "emitters": [{"name": n, "type": t}
                             for n, t in zip(truth.emitter_names, truth.emitter_types)],
                "n_bursts_total": len(segment_bursts(truth)),
            })
            runs_rle = []
            S = truth.S[:, :T_SHOW]
            for n in range(S.shape[0]):
                pad = np.concatenate(([0], S[n], [0]))
                d = np.diff(pad)
                for st, en in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                    runs_rle.append([int(n), int(st), int(en - st), int(truth.E[n, st])])
            out["truth_runs"] = runs_rle

        a = hist.actions[:T_SHOW].astype(int)
        o = hist.obs[:T_SHOW].astype(int)
        bl = hist.blind[:T_SHOW].astype(int)
        s_at = truth.S[a, np.arange(T_SHOW)].astype(int)
        # 0 blind, 1 true positive, 2 miss, 3 false alarm, 4 empty
        code = np.where(bl == 1, 0,
               np.where((s_at == 1) & (o == 1), 1,
               np.where((s_at == 1) & (o == 0), 2,
               np.where((s_at == 0) & (o == 1), 3, 4))))

        out["runs"][key] = {
            "label": label,
            "actions": a.tolist(),
            "code": code.tolist(),
            "reward": np.round(hist.rewards[:T_SHOW].astype(float), 2).tolist(),
            "final": {
                "poi": _f(m.poi), "sensor_pd": _f(m.sensor_pd), "effective_pd": _f(m.effective_pd),
                "sensor_pfa": _f(m.sensor_pfa), "mean_tti": _f(m.mean_tti),
                "blind_fraction": _f(m.blind_fraction), "discovery_ratio": _f(m.discovery_ratio),
                "intercept_rate_per_s": _f(m.intercept_rate_per_s),
                "n_switches": int(m.n_switches), "n_caught": int(m.n_caught),
                "n_bursts": int(m.n_bursts), "total_reward": round(float(m.total_reward), 1),
            },
        }

    path = os.path.join(HERE, "runs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")
    print(f"  {out['N']} channels, {out['T']} slots, {T_SHOW} animated, "
          f"{out['n_bursts_total']} transmissions")
    for k, v in out["runs"].items():
        fin = v["final"]
        print(f"  {k:<20} POI {fin['poi']:.4f}  caught {fin['n_caught']}/{fin['n_bursts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
