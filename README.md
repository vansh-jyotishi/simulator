# smart-scan-ew — SIH 26055 RF Environment Simulator

Closed-loop scan-strategy testbed for intercepting unknown emitters with a **single-channel receiver**. The simulator holds a hidden truth matrix `S` (N channels × T slots), returns noisy Bernoulli observations under retune blindness, and prints one side-by-side comparison table over identical and held-out seeds. Schedulers plug in through the frozen contract in [INTERFACE.md](INTERFACE.md) and never see the truth.

**Say the limitations first:** open-loop, non-adversarial emitters; single-channel receiver; single-pulse Albersheim Bernoulli detection with a documented low-SNR blend; truth pre-rendered at `reset()`; sensor Pd excludes blind dwells while effective Pd counts them as misses (both printed); a false alarm earns `R_hit` and a blind dwell pays `C_empty`; the scanning radar supports only `mode="lighthouse"` (the optional `band_sweep` mode from the build doc is not implemented).

## Setup (Python 3.12, one shared venv)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # numba first so numpy is pinned compatibly
.venv\Scripts\python -m pytest -q                   # 88 unit/contract tests, ~5 s
.venv\Scripts\python -m pytest tests/verify_sim.py -q -s   # 13 black-box physics cases, prints configured vs recovered
```

Run every command from the repo root: nothing is pip-installed, `pytest.ini` puts the root on `sys.path`, and the snippets below assume the same (or set `PYTHONPATH` to the repo root). Python 3.14 is not supported (no Numba wheels). New packages go through the simulator owner.

**Repo location and backup.** The working repo lives in this OneDrive folder by the owner's decision (the build plan preferred `C:\dev`). The `origin` remote is a bare backup repository outside OneDrive at `C:\dev\smart-scan-ew.git`; `main`, `contract-v1` and `v0.1-sim` are pushed there after every commit. To hand off through GitHub, create the repository and run `git remote add github <url>` then `git push github main --tags`; teammates code against the `contract-v1` tag. Checkpoint work for CK1 to CK3 landed in the commits `98bffbf` and `0b392ad` rather than one commit per checkpoint; each results table records the code SHA in its header.

## Minimal loop (10 lines)

```python
from env.spectrum_env import make_env
from common.protocol import sanitize_info
from oracle.metrics_engine import compute_metrics
from baselines.round_robin import RoundRobin

env = make_env("scenarios/train_default.json", seed=0)
obs, info = env.reset(seed=0)
sched = RoundRobin(); sched.reset(env.mission_spec, seed=0)
while True:
    a = sched.select_action(obs, sanitize_info(info))
    obs, r, term, trunc, info = env.step(a)
    if trunc: break
m = compute_metrics(env.get_truth(), env.get_history(), sched.predictions(), dwell_s=env.mission_spec.dwell_s)
print(m.poi, m.sensor_pd, m.effective_pd, m.sensor_pfa, m.mean_tti)
```

## Writing a scheduler (scheduler team)

```python
import numpy as np
from common.protocol import BaseScheduler
from common.types import Prediction

class MyScheduler(BaseScheduler):
    needs_truth = False                      # never True for a candidate

    def reset(self, spec, seed):
        super().reset(spec, seed)            # self.spec, self.seed, self.rng = default_rng([seed, 20_000])
        self.d = spec.tau_switch + 1         # cold start: dwell tau+1 slots per channel, never 1
        self.t = 0

    def select_action(self, obs, info):      # obs = O_{t-1}; info keys = INFO_KEYS; never truth
        if info["blind"]:                    # drop blind observations from your transition counts
            pass
        a = (self.t // self.d) % self.spec.N
        self.t += 1
        return int(a)                        # int in [0, N)

    def predictions(self):                   # optional: ambush commitments, scored with delta_guard=1
        return [Prediction(channel=7, t_pred=120, t_made=100, status="issued")]
```

Rules (INTERFACE.md C7): constructor takes no required arguments; never import `env/` internals or call `get_truth()`; do not condition runtime decisions on `info["reward"]` (its `I_new` term is truth-derived); derive randomness from `self.rng`. **Cold-start warning:** the ML doc §7 one-dwell sweep `a_t = t` is 100 % blind at `tau_switch = 1`.

Register in `schedulers/__init__.py` as `REGISTRY = {"whittle": WhittleScheduler}` or pass `--extra whittle=schedulers.whittle_rmab:WhittleScheduler`. Before asking for a merge:

```powershell
.venv\Scripts\python -m pytest tests/test_protocol_conformance.py -q -s --scheduler schedulers.whittle_rmab:WhittleScheduler
```

It checks 300 steps on the real env, int actions in `[0, N)`, determinism per seed, well-formed predictions, `needs_truth == False`, and prints mean / p99 µs per decision (budget 15 µs). Train on `train_default.json` seeds 0–99 only; seeds 100–199 and `heldout_*` files are never used for tuning.

## The comparison table

```powershell
.venv\Scripts\python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json `
    --seeds 0-9 --heldout-seeds 100-109 --schedulers round_robin weighted_priority clairvoyant `
    --extra whittle=schedulers.whittle_rmab:WhittleScheduler --scheduler-kwargs '{"whittle": {"beta": 0.95}}' `
    --out results/comparison.csv --md results/comparison.md [--save-logs] [--episodes 3]
.venv\Scripts\python -m eval.run_comparison --snr-sweep --seeds 0-2 --schedulers round_robin   # sensor Pd vs SNR
```

Rows are `(scheduler, scenario, seed_split)`, cells are mean ± std over seeds; the header records git SHA, scenario hashes, seed lists and library versions. Columns: POI (burst), POI_time (time coverage), sensor Pd, effective Pd, Pfa, mean/median TTI, intercepts/s, correct-pred %, prediction error, total reward, blind %, switches, µs/decision mean and p99. `clairvoyant` reads the truth and is the CHEATING upper bound; `--extra` classes with `needs_truth=True` are refused. Same `(scenario, seed)` gives bit-identical truth and noise for every scheduler (common random numbers), so the comparison is paired. `--episodes k` re-rolls only the scheduler's seed (`seed + e*1_000_000`); the env stays fixed.

Committed reference numbers: [results/baselines_v0.1.md](results/baselines_v0.1.md) (before any RL) and [results/comparison.md](results/comparison.md) (all scenarios incl. clairvoyant). Re-running the header command reproduces the CSV bit-for-bit.

## Dashboard hooks (dashboard team)

- `env.get_truth().S` is the (N, T) waterfall; `.E` gives the emitter id per cell for colouring (-1 = empty); `.SNR` in dB.
- `env.get_history()` holds live views of `actions`, `obs`, `blind`, `switched`, `new_detect`, `rewards` and `t`.
- `compute_metrics(truth, hist, upto_t=t)` gives running numbers at any slot.
- `env.render()` is an ASCII fallback (`#` truth, `@` true positive, `x` blind, `?` false alarm, `o` empty visit, `m` miss).
- `--save-logs` writes `results/logs/<scenario>__<scheduler>__seed<k>_ep<e>.npz` with `S, S_by_emitter, E, SNR, actions, obs, blind, switched, new_detect, rewards` for replay without a scheduler.
- Channels and slots are 0-based in code; add +1 for display.

## Scenarios

| file | purpose |
|---|---|
| `mvp_periodic.json` | N=50, T=2000, fixed Pd 0.85 / Pfa 0.05, two periodic radars (PRI 17 and 23) — the smoke scenario |
| `train_default.json` | the reference mix: two radars, one hopping jammer, one lighthouse search radar, Albersheim noise, Pfa 1e-3 |
| `heldout_mix.json` | different mix (two jammers, PRIs 29 and 61, a 3 dB radar, T_rot 170) — never used for tuning |
| `heldout_stale_intel.json` | train mix with `intel_weights` boosting quiet channels: weighted_priority scores below round_robin |
| `blind_zone.json` | N=10, PRI 20 == round-robin sweep period: round_robin POI is exactly 0, weighted_priority > 0 |
| `snr_sweep_{-5,0,5,10,15}.json` | one always-on radar at X dB, tau_switch 0: sensor Pd 0.03 / 0.10 / 0.21 / 0.81 / 1.00 |

Seeding: emitter `i` uses `default_rng([seed, i])`, noise `U` uses `default_rng([seed, 10_000])`, schedulers use `default_rng([seed, 20_000])`. `reset(seed=None)` reuses the `make_env` seed and never auto-advances it.

## Metric decisions

- **Sensor Pd** = tp / (tp + fn) on non-blind dwells; **effective Pd** = tp / all dwells on occupied cells (blind dwells are misses). Both are printed.
- **POI** = caught bursts / all bursts, where a burst is a maximal run of 1s per (emitter, channel) and is caught only by a true positive (never by a blind dwell or a false alarm). **POI_time** is the time-coverage form. Per-type POI is NaN for absent types.
- **TTI** = first true positive minus burst start, over caught bursts.
- **Predictions** are scored only for `status == "issued"`; correct iff the channel is truly active within ±`delta_guard` (default 1) of `t_pred`; error is the distance to the nearest burst start on that channel.
- Every zero denominator gives NaN, never an exception.

## Performance (N=50, T=5000, this laptop)

`reset()` with a fresh seed ≈ 30–45 ms (truth render), well under 1 ms when the seed is unchanged; `step()` ≈ 7–10 µs mean over 5000 steps; round_robin ≈ 0.7 µs per decision. The baselines are not tuned for the 15 µs candidate budget: weighted_priority averages ≈ 4 µs but its p99 over a short 300-step run can exceed 15 µs because of first-call overhead.

## Notes on the borrowed ideas and known corrections

The `onoff=[start, on, off]` duty cycle, weighted hopping via `rng.choice`, the per-cell emitter-id matrix and the JSON type registry are ideas from rfrl-gym (MIT); the code is written fresh. rfrl-gym's `agile_freq.py` reacts to the player's last action, so it is *not* open-loop as the build doc §4.1 claims; our jammer is open-loop by design. Because a hop excludes the current channel, long-run occupancy equals `hop_probs` only when they are uniform; `AgileJammer.stationary_distribution()` gives the exact occupancy. Albersheim's closed form floors near Pd 0.07 at low SNR, so below the Pd = 0.1 point it is blended to Pfa with a 10 dB/decade law (INTERFACE.md C3).

## Layout

`common/` contract types and scheduler protocol · `env/` scenario loader, receiver, emitters, Gymnasium env · `oracle/` bursts, ledger, metrics · `baselines/` round_robin, weighted_priority, clairvoyant · `eval/` runner and comparison CLI · `tests/` unit tests, `verify_sim.py`, protocol conformance · `scenarios/` JSON · `results/` (gitignored except `baselines_v0.1.*` and `comparison.md`) · `schedulers/`, `estimators/`, `dashboard/` belong to the other teams.
