# SIH 26055 RF Environment Simulator — Solo Build Plan (Day 0 + 5 working days)

## Context

SIH PS 26055 asks for a closed-loop scan strategy that intercepts unknown emitters with a single-channel receiver. The research phase is done: the three docs in `C:\Users\Vansh\OneDrive\Desktop\SIH\` (RF simulator build doc, ML scheduler build doc, theory guide PDF) fully specify what to build. One person now builds the **RF Environment Simulator** end to end. The simulator is the evidence base for the whole project: it holds the hidden truth matrix S, returns noisy observations under retune blindness, and prints the single side-by-side comparison table that the docs call the most convincing artifact for judges. Consumers are the scheduler teammates (Whittle index + Lomb-Scargle, needs Numba) and the dashboard teammate (Streamlit). They plug in through a frozen contract and never edit simulator folders.

## What changed from the two-person plan

- **One sequential track.** The env is built first (Day 1) because every other piece needs it. Oracle, baselines and eval follow (Days 2–4). Extras and hand-off are Day 5.
- **FakeEnv is dropped.** It only existed so Person B could work before the real env was ready. Everything in `oracle/`, `baselines/` and `eval/` is now tested directly against `env.spectrum_env.make_env`. Scripted schedulers and hand-built tiny fixtures stay, because they prove metrics against known numbers rather than against the env's physics.
- **No branches, no sync merges, no pairing slots.** Work on `main`. The five integration checkpoints remain as self-checks with exact commands.
- **Honest sizing.** The two-person plan had about 52 h of work across two people in 3 days. Solo, with FakeEnv and pairing removed, it is about 45 h: Day 0 (1.5 h) plus five days of 8.5–9.5 h. Everything mandatory is finished by the end of Day 4 (CK3). If Day 5 is lost, the cut list at the bottom applies.

## Scope and assumptions

- **In scope:** `env/`, `oracle/`, `baselines/`, `eval/`, `tests/`, `scenarios/`, `common/`. **Not ours:** `schedulers/`, `estimators/`, `dashboard/` (empty `__init__.py` + one-line README stub only).
- **Toolchain:** this machine has only Python 3.14 and no numpy. Numba has no 3.14 wheels and the scheduler team needs it. Day 0 installs Python 3.12 (`winget install -e --id Python.Python.3.12`), one venv, numba installed first so pip pins a compatible numpy.
- **Repo location:** `C:\dev\smart-scan-ew` (outside OneDrive, which locks `.git` and `.venv`). Copy the three docs into `docs/`.
- **rfrl-gym decision:** borrow ideas, write fresh, do not clone. Verified reasons: unconditional PyQt6 import, agent sees all channels, no Pd/Pfa, no blindness, `reset(seed)` ignored, `eval()`-built entities, and `agile_freq.py` reacts to the player's last action (it is NOT open-loop; the build doc §4.1 claim is wrong). Borrow with an MIT attribution comment: the `onoff=[X,Y,Z]` cycle as closed form `active = t >= start and ((t-start) % (Y+Z)) < Y`, weighted hop via `rng.choice(channels, p=hop_probs)`, per-cell emitter-id matrix, JSON scenario via a type registry.
- **Design decisions kept:** emitters are open-loop, so S is rendered in full at `reset()` (identical output to per-step writes, far simpler). Single-channel receiver. Bernoulli Pd/Pfa via Albersheim (n_pulses=1) with a documented low-SNR blend. Channels and slots are 0-based in code (dashboard adds +1 for display).
- **Reference numbers:** N=50, T=5000, dwell_s=1e-3, tau_switch=1, Pfa=1e-3, R_new=+10, R_hit=+2, C_empty=-0.5, C_tune=-1.5 (ML doc §0, §9, §14). Train seeds 0–99, held-out seeds 100–199, held-out scenario files start with `heldout_`.
- **Stated limitations (say them first to judges):** open-loop non-adversarial emitters; single-channel receiver; single-pulse Albersheim Bernoulli approximation; S pre-rendered; sensor Pd excludes blind dwells while effective Pd includes them (both printed).
- **Key trap found in review:** with tau_switch=1, a baseline that switches every dwell is blind on every dwell (POI=0, Pd=NaN). Both baselines therefore dwell `tau_switch+1` slots per channel by default, and the hand-off warns the scheduler team that the ML doc §7 cold-start sweep `a_t = t` has the same problem.

## Repo layout and build order

```
C:\dev\smart-scan-ew\
  INTERFACE.md  common\types.py  common\protocol.py                      [Day 0, frozen after]
  requirements.txt  pyproject.toml  pytest.ini  .gitignore  README.md  docs\   [Day 0]
  env\scenario.py  env\receiver_model.py  env\emitters.py  env\spectrum_env.py   [Day 1, extended Day 2–3]
  scenarios\*.json  (mvp_periodic.json on Day 0, the rest Day 3 and Day 5)      [Day 0 / 3 / 5]
  tests\test_env_api.py  tests\test_emitters.py                                  [Day 1 / 3]
  tests\fakes.py  tests\test_oracle.py                                           [Day 2 / 4]
  oracle\truth_tracker.py  oracle\metrics_engine.py                              [Day 2, extended Day 3]
  baselines\round_robin.py  eval\runner.py  eval\run_comparison.py               [Day 2, extended Day 4–5]
  baselines\weighted_priority.py  tests\test_baselines.py  tests\verify_sim.py   [Day 4]
  baselines\clairvoyant.py  tests\test_protocol_conformance.py                   [Day 5]
  results\  (gitignored except results\baselines_v0.1.* and results\comparison.md) [Day 4 / 5]
  schedulers\  estimators\  dashboard\  (README stub + empty __init__.py)           [not ours]
```

Import discipline (kept even solo, because the scheduler team must obey the same rule): `oracle/`, `baselines/` and `eval/` import nothing from `env/` except `make_env`; `env/` never imports `oracle/`, `baselines/` or `eval/`. Contract beats code: if the harness and the env disagree, INTERFACE.md decides, and the fix goes to whichever side drifted.

## Day 0 (about 1.5 h)

1. **(10 min) Repo.** Install Python 3.12. `mkdir C:\dev\smart-scan-ew; cd C:\dev\smart-scan-ew; git init -b main; git config core.autocrlf true`. `.gitattributes` = `* text=auto`. `.gitignore` = `.venv/ results/ __pycache__/ .pytest_cache/ *.csv *.npz` plus `!results/baselines_v0.1.*` and `!results/comparison.md`. Create GitHub remote, push (the remote is the backup and the hand-off copy).
2. **(15 min) Venv.** `py -3.12 -m venv .venv; .venv\Scripts\activate; pip install "numba>=0.59"` then `pip install "gymnasium>=0.29" "scipy>=1.12" "pandas>=2.2" "pytest>=8"`. `pip freeze > requirements.txt`. `pytest.ini` = `[pytest]\npythonpath = .\ntestpaths = tests`. `pyproject.toml` with `requires-python = ">=3.11,<3.13"`. Done when `python -c "import sys,numba,numpy,gymnasium,scipy,pandas; assert sys.version_info[:2]==(3,12); print('ok')"` prints ok.
3. **(10 min) Skeleton.** Folders per layout above, empty `__init__.py` in every package (never edited again), README stubs in the three not-ours folders, docs copied into `docs/`.
4. **(50 min) Contract.** Write `common/types.py`, `common/protocol.py`, `INTERFACE.md` from the contract below, plus `scenarios/mvp_periodic.json` (N=50, T=2000, receiver `noise_model:"fixed"`, pd_fixed 0.85, pfa 0.05, tau_switch 1, two periodic radars: ch 7 onoff [0,3,14] and ch 33 onoff [4,5,18]. PRIs 17 and 23 are coprime with the round-robin sweep period N*(tau_switch+1)=100 so the first table is non-degenerate. All `randomize` flags false). Done when `python -c "import common.types, common.protocol"` succeeds. Commit `feat: interface contract v1`, tag `contract-v1`, push. Tell the scheduler and dashboard teammates the tag exists; they code against it from now on.
5. **(5 min) Git rules.** Work on `main`. Commit at least every 2 h, push on every commit. The contract is frozen after this tag because other teammates consume it: adding a field with a default is allowed; renaming or removing is not before tag `v0.1-sim`. Any contract change edits `INTERFACE.md`, `common/` and every implementer in the same commit, and is announced in the team chat with the exact diff.

## Interface contract (INTERFACE.md, frozen after Day 0)

### C1. Shared types — `common/types.py`
```python
EMITTER_TYPES = ("periodic", "agile", "scanning")          # closed set; loader accepts no aliases
R_NEW, R_HIT, C_EMPTY, C_TUNE = 10.0, 2.0, 0.5, 1.5         # magnitudes; signs applied in the formula
INFO_KEYS = ("t", "next_t", "action", "switched", "blind", "blind_remaining", "reward", "reward_terms")  # exact set, never truth

@dataclass(frozen=True)
class MissionSpec:        # what an operator legitimately knows; never contains truth
    N: int; T: int; dwell_s: float; tau_switch: int; pfa: float; initial_channel: int
    intel_weights: tuple[float, ...] | None                 # len N, >=0, sum>0, or None

@dataclass(frozen=True, eq=False)
class Truth:              # ORACLE/DASHBOARD ONLY, reachable only via env.get_truth()
    S: np.ndarray             # (N,T) int8, OR over S_by_emitter
    S_by_emitter: np.ndarray  # (M,N,T) int8
    SNR: np.ndarray           # (N,T) float32 dB, -inf where S==0, max over emitters on overlap
    E: np.ndarray             # (N,T) int16 emitter id with max SNR (ties: lowest id), -1 where S==0
    U: np.ndarray             # (N,T) float32 Uniform(0,1) pre-drawn noise = common random numbers
    emitter_names: tuple[str, ...]; emitter_types: tuple[str, ...]

@dataclass(frozen=True, eq=False)
class History:            # written by env, read by oracle; arrays length T, slots >= t unfilled (-1/0)
    actions: np.ndarray; obs: np.ndarray; blind: np.ndarray; switched: np.ndarray; new_detect: np.ndarray; rewards: np.ndarray
    t: int                # slots executed so far

class Prediction(NamedTuple): channel: int; t_pred: int; t_made: int; status: str   # "issued" | "dropped"
@dataclass(frozen=True)
class Burst: emitter: int; emitter_type: str; channel: int; t_start: int; t_end: int   # inclusive
```

### C2. Environment API — `env/spectrum_env.py`
```python
def make_env(scenario: str | Path | dict, seed: int, receiver: ReceiverModel | None = None) -> SpectrumEnv
class SpectrumEnv(gymnasium.Env):
    action_space = Discrete(N); observation_space = Discrete(2); mission_spec: MissionSpec
    def reset(self, *, seed=None, options=None) -> tuple[int, dict]    # (0, info0); seed None reuses make_env seed
    def step(self, action: int) -> tuple[int, float, bool, bool, dict]  # (O_t as Python int, R_t, terminated=False, truncated=(t==T), info)
    def get_truth(self) -> Truth; def get_history(self) -> History; def render(self) -> str  # ASCII waterfall
```
`info0 = {"t": -1, "next_t": 0, "action": initial_channel, "switched": False, "blind": False, "blind_remaining": 0, "reward": 0.0, "reward_terms": {"new":0,"hit":0,"empty":0,"tune":0}}`. Per step, `t` = index of the slot just executed. `reward_terms` are signed floats with `sum(reward_terms.values()) == reward`. Step after truncation raises `RuntimeError`.

Exact `step(a)` order for slot t:
1. `O_t, switched, blind, blind_remaining = receiver.observe(a, S[a,t], SNR[a,t], U[a,t])`. The receiver is the single owner of prev-channel and blind state (prev at reset = `initial_channel`, so no tune cost at t=0 if the scheduler starts there). Blind rule: on switch `blind_left = tau_switch`; `blind = blind_left > 0`; if blind, `blind_left -= 1`; `blind_remaining` reported post-decrement; a switch during blindness restarts the counter; tau_switch=0 never blinds. Blind → `O_t = 0`; else `O_t = int(U[a,t] < p)` with `p = pd(SNR[a,t])` if `S[a,t]==1` else `pfa`.
2. `I_new = 1` iff `O_t==1 and S[a,t]==1` and some emitter active in that cell was never truly detected before; all active ones are then marked seen.
3. `R_t = R_NEW*I_new + R_HIT*O_t - C_EMPTY*(1-O_t) - C_TUNE*switched`. Worked cases: switch + first true hit = 10.5; stay + known hit = 2.0; switch + empty = -2.0; stay + empty = -0.5. (A false alarm earns R_hit and a blind dwell pays C_empty: stated limitation.)
4. Write `History[t]`; `t += 1`; `truncated = (t == T)`; build info with exactly `INFO_KEYS`.

### C3. Receiver — `env/receiver_model.py`
```python
def albersheim_snr_db(pd, pfa, n_pulses=1) -> float
    # A = ln(0.62/pfa); B = ln(pd/(1-pd)); K = 6.2 + 4.54/sqrt(n+0.44)
    # snr = -5*log10(n) + K*log10(A + 0.12*A*B + 1.7*B)      golden: (0.9, 1e-6) -> 13.1 +/- 0.15 dB
def albersheim_pd(snr_db, pfa, n_pulses=1, blend=True) -> float
    # X = 10**((snr + 5*log10(n))/K); B = (X-A)/(0.12*A+1.7); pd = 1/(1+exp(-B)); clamp to [pfa, 1]
    # closed form floors at ~0.069 (pfa=1e-3) as snr -> -inf, so below snr_lo = albersheim_snr_db(0.1, pfa):
    #   pd = pfa + (0.1-pfa) * 10**((snr-snr_lo)/10)   (continuous, monotone, -> pfa)
    # reference @ pfa=1e-3: -5 dB .032, 0 dB .10, 5 dB .21, 10 dB .81, 13 dB .996, 15 dB > .999
def sinc2_gain(delta_deg, beamwidth_deg) -> float
    # x = 2.783*delta/bw; gain = (sin x / x)^2; gain(0)=1, gain(+-bw/2)=0.5 (3-dB beamwidth); first null at 1.129*bw
class Observation(NamedTuple): obs: int; switched: bool; blind: bool; blind_remaining: int
class ReceiverModel:
    def __init__(self, tau_switch: int, pfa: float, noise_model: Literal["fixed","albersheim"]="albersheim", pd_fixed: float=0.85, n_pulses: int=1)
    def reset(self, initial_channel: int) -> None
    def observe(self, action: int, truth_bit: int, snr_db: float, u: float) -> Observation   # pure; env owns U
    def pd(self, snr_db) -> float
class PassThroughReceiver(ReceiverModel):   # tau_switch=0, O_t == truth_bit, ignores u (debug / verify)
```

### C4. Emitters — `env/emitters.py`. All open-loop. `render(N, T) -> (S_m int8 (N,T), SNR_m float32 (N,T))`, vectorised.
- `PeriodicRadar(channel, onoff=[start_delay, on, off], snr_db)`: `S_m[channel,t] = 1 iff t >= start and ((t-start) % (on+off)) < on`; PRI = on+off. `randomize.periodic_phase` adds `rng.integers(0, PRI)` to start.
- `AgileJammer(channels, hop_period, hop_probs|None, p_stay, onoff, snr_db)`: every `hop_period` slots keep the channel with prob `p_stay`, else draw from `channels` with `hop_probs` (uniform if null) excluding the current one; emits only while onoff-active; exactly one channel per active slot. Optional `transition_matrix` (row-stochastic) is a Day 5 stretch.
- `ScanningRadar(mode="lighthouse", channel, T_rot, beamwidth_deg, phase0_deg, snr_peak_db, presence_gain_db=-10.0)`: `theta(t) = (360*t/T_rot + phase0) mod 360`, receiver at azimuth 0, `delta = wrap(theta(t))`. `SNR_m = snr_peak_db + 10*log10(sinc2_gain(delta, bw))`; `S_m = 1 iff |delta| < 1.129*bw (mainlobe only) and SNR_m - snr_peak_db >= presence_gain_db` (first sidelobe is -13.3 dB, never registers). Gives exactly one burst per rotation of length `round(T_rot * 1.667*bw / 360) +- 1` at -10 dB. `mode="band_sweep"` (beam sweeps across bands lo..hi, build doc §4.2) is optional Day 5.
- Overlap: `S = OR`, `SNR = max`, `E = argmax SNR`. Burst = maximal run of 1s per (emitter, channel) in `S_by_emitter`.

### C5. Scenario JSON — `scenarios/*.json`, `env/scenario.py`: `load_scenario(path|dict) -> ScenarioConfig`, `to_mission_spec(cfg)`
```json
{"name":"train_default","N":50,"T":5000,"dwell_s":0.001,
 "receiver":{"tau_switch":1,"pfa":1e-3,"noise_model":"albersheim","pd_fixed":0.85,"n_pulses":1,"initial_channel":0},
 "reward":{"R_new":10.0,"R_hit":2.0,"C_empty":0.5,"C_tune":1.5}, "intel_weights":null,
 "randomize":{"periodic_phase":true,"agile_start_channel":true,"scanning_phase":true},
 "emitters":[
  {"type":"periodic","name":"radar_A","channel":7,"onoff":[0,3,14],"snr_db":12.0},
  {"type":"periodic","name":"radar_B","channel":33,"onoff":[4,5,42],"snr_db":6.0},
  {"type":"agile","name":"jammer_1","channels":[20,21,22,23,24,25],"hop_probs":null,"hop_period":8,"p_stay":0.2,"onoff":[5,40,10],"snr_db":9.0},
  {"type":"scanning","name":"search_radar","mode":"lighthouse","channel":41,"T_rot":130,"beamwidth_deg":20.0,"phase0_deg":0.0,"snr_peak_db":15.0,"presence_gain_db":-10.0}]}
```
Validator raises `ValueError` naming the field for: unknown key; type not in EMITTER_TYPES; channel >= N; onoff ints >= 0 with on > 0; PRI >= T or T_rot >= T; hop_period < 1; agile `channels` not a subset of [0,N) or len < 2; hop_probs length/sum; intel_weights not null and (len != N or any < 0 or sum == 0); two fixed-channel emitters on the same channel (jammer hop sets may overlap radar channels). Relative paths resolve against cwd then repo root. Seed is never stored in JSON.

### C6. Seeding and common random numbers
`make_env(scenario, seed)`: emitter i uses `np.random.default_rng([seed, i])`; `U = default_rng([seed, 10_000]).random((N,T), dtype=float32)`; schedulers derive randomness from `default_rng([seed, 20_000])` inside `reset()`. Same (scenario, seed) → bit-identical S, E, SNR, U for every scheduler, so the comparison is paired.

### C7. Scheduler protocol — `common/protocol.py` (implemented by baselines and the scheduler team, consumed by eval)
```python
def sanitize_info(info: dict) -> dict            # whitelist INFO_KEYS; runner applies it before anything reaches a scheduler
class Scheduler(Protocol):
    needs_truth: bool                             # False for every candidate; True only for baselines.clairvoyant
    def reset(self, spec: MissionSpec, seed: int) -> None
    def select_action(self, obs: int, info: dict) -> int   # obs = O_{t-1} (0 at first call), info = sanitize_info(previous info); return a_t in [0,N)
    def predictions(self) -> list[Prediction]     # ambush commitments made during the episode; [] if none
class BaseScheduler:                              # stores spec, seed, rng=default_rng([seed,20_000]); predictions() -> []
```
Rules: constructor takes no required arguments (config via `reset` or `--scheduler-kwargs`). `info["reward"]` is training telemetry (I_new is truth-derived): runtime candidates must not condition decisions on it; offline benchmarks may. Never import `env/` internals or call `get_truth()`. **Cold-start warning:** the ML doc §7 one-dwell sweep `a_t = t` is 100% blind at tau_switch=1; bootstrap must dwell `spec.tau_switch + 1` slots per channel and drop blind observations (`info["blind"]`) from tau=1 transition counts.

### C8. Oracle — `oracle/`. `MetricsResult` lives in `oracle/metrics_engine.py` (defaulted fields may be added freely); core keys frozen here
```python
def segment_bursts(truth) -> list[Burst]
def dwell_ledger(truth, hist, upto_t=None) -> dict[str, np.ndarray]   # tp fa fn tn blind switched (bool, len t)
def compute_metrics(truth, hist, predictions=(), upto_t=None, delta_guard=1) -> MetricsResult   # .to_dict() flat
METRIC_KEYS = sensor_pd effective_pd sensor_pfa n_tp n_fp n_fn n_tn n_blind  poi poi_periodic poi_agile poi_scanning poi_time n_bursts n_caught
              mean_tti median_tti intercept_rate_per_dwell intercept_rate_per_s discovery_ratio  n_predictions n_dropped_predictions correct_pred_pct avg_pred_time_error
              total_reward mean_reward blind_fraction switch_rate n_switches  us_per_decision_mean us_per_decision_p99 (filled by eval)
```
Definitions (theory guide Part 6): `s_at = S[a_t,t]`; tp/fp/fn/tn on non-blind dwells only. `sensor_pd = tp/(tp+fn)`; `effective_pd = tp / #(s_at==1)` (blind counted as misses; the theory-guide 32/40 form); `sensor_pfa = fp/(fp+tn)`. Burst caught iff exists t in [t_start,t_end] with `a_t == channel` and tp[t] (a blind dwell or false alarm never catches); `poi = n_caught/n_bursts`, by type via `Burst.emitter_type`, NaN for absent types; `poi_time = #{(m,t): S_by_emitter[m,a_t,t]==1 and tp[t]} / S_by_emitter.sum()` (build doc §3.1 time-coverage form). `tti = t_first_tp - t_start` over caught bursts (mean and median). `intercept_rate_per_dwell = n_caught/t`, `per_s = n_caught/(t*dwell_s)`. `discovery_ratio = emitters with >= 1 tp / M`. Predictions: only `status=="issued"` scored; correct iff any `S[c, t_pred-delta_guard .. t_pred+delta_guard] == 1`; error = `|t_pred - nearest burst start on c|`, averaged over predictions whose channel has >= 1 burst; `t_pred` outside [0,T) → incorrect, excluded from error; dropped counted only in `n_dropped_predictions`. All zero denominators → NaN, never an exception.

### C9. Eval CLI — `eval/run_comparison.py`
```
python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json --seeds 0-9 --heldout-seeds 100-109
   --schedulers round_robin weighted_priority clairvoyant [--extra whittle=schedulers.whittle_rmab:WhittleScheduler]
   [--scheduler-kwargs '{"whittle": {"beta": 0.95}}'] [--episodes 1] [--save-logs] [--snr-sweep] [--out results/comparison.csv]
```
Every scheduler runs every scenario on both seed lists (asserted disjoint). Row = (scheduler, scenario, seed_split); `scenario_split` = heldout iff file name starts with `heldout_`. Row label = registry key or `--extra` alias. Header prints git SHA, sha256[:8] of each scenario file, seed lists, numpy/python versions. Columns: POI, poi_time, sensor Pd, effective Pd, sensor Pfa, mean/median TTI, intercepts/s, correct-pred %, pred err, total reward, blind %, switches, us/decision mean and p99 (mean ± std over seeds). Per-(scheduler, scenario, seed, episode) rows go to CSV. Registry auto-discovers `schedulers/__init__.py::REGISTRY` (ImportError ignored). `--extra` classes with `needs_truth=True` are refused.

## Day 1 (~8.5 h) — goal: `SpectrumEnv` passes `check_env` and the contract tests

| ID | File | What | Done when | h |
|---|---|---|---|---|
| S1 | `env/scenario.py` | Frozen dataclasses (ScenarioConfig, ReceiverCfg, RewardCfg, RandomizeCfg, per-type emitter cfgs), `load_scenario`, every validator rule in C5, `to_mission_spec`, repo-root path fallback | `load_scenario('scenarios/mvp_periodic.json').N == 50`; a typo key, channel 50, PRI >= T and intel_weights of length 49 each raise ValueError naming the field | 1.5 |
| S2 | `env/receiver_model.py` | `ReceiverModel` per C3 with full blind counter, `noise_model="fixed"` working, `albersheim_*` and `sinc2_gain` as `NotImplementedError` stubs, `PassThroughReceiver` | `PassThroughReceiver().observe(3,1,10.0,0.99) == (1,False,False,0)`; `ReceiverModel(2,1e-3)` after a switch is blind for exactly 2 calls with blind_remaining 1 then 0; fixed mode returns `int(u < 0.85)` on truth 1 | 1.0 |
| S3 | `env/emitters.py` | `Emitter` base with closed-form onoff + attribution comment, `PeriodicRadar.render`, `build_emitters(cfg, seed)` using `default_rng([seed, i])`, `render_truth(cfg, seed) -> Truth` (OR/max/argmax, U from `[seed, 10_000]`) | onoff [0,3,17], T=2000, randomize off: row sum == 300, autocorrelation peaks at lag 20, interior runs all 3 and gaps all 17; same seed → `np.array_equal`; Truth dtypes/shapes per C1 | 2.0 |
| S4 | `env/spectrum_env.py` | `SpectrumEnv` + `make_env` exactly per C2 (step order, info keys, reward, History, `truncated`, RuntimeError after end, `render`), no allocation in `step()` | `gymnasium.utils.env_checker.check_env(make_env('scenarios/mvp_periodic.json', 0))` passes; 2000 steps of action 7 with PassThroughReceiver give `obs == S[7]`; rewards 10.5 / 2.0 / -2.0 / -0.5 on scripted cases; info keys == INFO_KEYS; `info0["t"] == -1` | 3.0 |
| S5 | `tests/test_env_api.py` | Contract tests: info key set and types, `t`/`next_t` progression, dtypes/shapes of Truth/History, `mission_spec` is a MissionSpec, determinism, `reward == sum(reward_terms)`, step-after-end raises; inline lambda policies (`lambda t: 7`, switch-every-k) only, no imports from oracle/baselines/eval | `python -m pytest tests/test_env_api.py -q` green in < 20 s | 1.0 |

End of day: commit `feat: spectrum env v0`, push. Smoke check before stopping: a 10-line loop with `lambda t: 7` and `PassThroughReceiver` on mvp_periodic prints `obs.sum() == S[7].sum()`.

## Day 2 (~8.5 h) — goal: first real round-robin table on `mvp_periodic.json` by mid-afternoon (CK1), Albersheim in by end of day

| ID | File | What | Done when | h |
|---|---|---|---|---|
| S6 | `tests/fakes.py`, `tests/test_oracle.py` (fixtures) | Scripted schedulers `StayPut(channel)`, `Scripted(actions)`, `SwitchEvery(k, channels)`, `FakePredictor` (perfect and off-by-2 issued + one dropped); hand-built `tiny_truth`/`tiny_history` (N=4, T=20) with the theory-guide Part 6 ratios (Pd 4/5, Pfa 1/20, POI 2/4, errors [2,4,1,3,5] → 3.0, 4/5 predictions → 80%) | Fixtures collect; scripted schedulers pass the C7 type check | 1.0 |
| S7 | `oracle/truth_tracker.py` | `segment_bursts` (np.diff on padded rows of `S_by_emitter`, sorted by t_start, `emitter_type` filled), `dwell_ledger` with `upto_t` | tiny_truth bursts equal the hand list; a run split by one 0 gives two bursts; a full mvp_periodic episode through the real env: `sum(len) == S_by_emitter.sum()` | 2.0 |
| S8 | `oracle/metrics_engine.py` | `MetricsResult` + `compute_metrics` v0: sensor/effective Pd, sensor Pfa, confusion counts, burst POI, n_caught, total/mean reward, blind_fraction, switch_rate, `to_dict` | Worked-example numbers reproduce exactly; NaN (not exception) on zero denominators | 2.0 |
| S9 | `baselines/round_robin.py`, `eval/runner.py`, `eval/run_comparison.py` (v0) | `RoundRobin(dwell_steps=None)`: `a_t = ((t // d) % N)` with `d = spec.tau_switch + 1` by default (documented); `run_episode(env_factory, scenario, seed, scheduler)` with `sanitize_info`, `perf_counter_ns` per `select_action` (mean and p99), `REGISTRY`; CLI v0 (`--scenarios --seeds --schedulers --out`), pandas table with header block | `python -m eval.run_comparison --scenarios scenarios/mvp_periodic.json --seeds 0-2 --schedulers round_robin` prints a table in < 10 s | 1.5 |
| CK1 | — | First real table. Expect POI in (0,1), effective Pd < sensor Pd, blind % ≈ 50%. If a number is wrong, decide from INTERFACE.md which side drifted and fix that side only | Table prints as expected; commit `feat: first comparison table`, push | 0.5 |
| S10 | `env/receiver_model.py` | `albersheim_snr_db`, `albersheim_pd` with clamp and low-SNR blend, `sinc2_gain`, `noise_model="albersheim"` path; validity range in docstring | `abs(albersheim_snr_db(0.9,1e-6) - 13.1) < 0.15`; round-trip to 1e-9 for pd in {0.1,0.5,0.9}, pfa in {1e-3,1e-6}; `albersheim_pd(10,1e-3) == 0.81 ± 0.01`; `albersheim_pd(-5,1e-3) == 0.032 ± 0.005`; monotone on linspace(-15,20,71); `sinc2_gain(±10,20) == 0.5 ± 1e-3`, `sinc2_gain(0,20) == 1` | 1.5 |

## Day 3 (~9 h) — goal: all three emitters and the full metric set under the harness (CK2)

| ID | File | What | Done when | h |
|---|---|---|---|---|
| S11 | `env/emitters.py` | `AgileJammer` (hop sequence precomputed for ceil(T/hop_period) hops, `np.repeat` expansion, onoff mask, `hop_probs` + `p_stay`) | `S_m.sum(axis=0) in {0,1}` for every t; changes only at multiples of hop_period; p_stay=0.5 over 5000 hops → stay fraction 0.5 ± 0.03; occupancy within 0.03 of `hop_probs` when p_stay=0 | 2.0 |
| S12 | `env/emitters.py` | `ScanningRadar` lighthouse per C4, outer-product vectorisation, `np.errstate` for gain 0; band_sweep only if time on Day 5 | Lighthouse T_rot=130, bw=20, -10 dB: burst starts differ by exactly 130, run length == 12 ± 1, per-rotation peak == 15.0 ± 0.05 dB, no cell outside the mainlobe | 2.5 |
| S13 | `tests/test_emitters.py`, `tests/test_env_api.py` | Unit tests for S10–S12 plus `test_common_random_numbers` (two different action sequences, same seed → identical `Truth.U` and `S`); `test_blind_equals_switches_times_tau` for tau in {0,1,2,3}; seed 0 vs 1 differ with randomize on | `python -m pytest tests/test_env_api.py tests/test_emitters.py -q` green in < 40 s | 1.5 |
| S14 | `scenarios/train_default.json`, `scenarios/heldout_mix.json` | train_default = C5 JSON verbatim; heldout_mix = different mix (2 jammers, periodic PRIs 29 and 61, lighthouse on another channel with T_rot 170, one 3 dB radar); PRIs/T_rot coprime-ish with 100 | Both load; `python -m eval.run_comparison --scenarios scenarios/heldout_mix.json --seeds 100-101 --schedulers round_robin` runs | 0.5 |
| S15 | `oracle/metrics_engine.py` | Metrics v1: poi_by_type, poi_time, mean/median TTI, intercept rates (per dwell and per s), discovery_ratio, prediction scoring with `delta_guard` and dropped status, `upto_t` | tiny fixtures: TTI and rate hand values; FakePredictor perfect → 100 % / 0.0, off-by-2 → 0 % / 2.0 at delta_guard=1; baselines → NaN | 2.0 |
| CK2 | — | train_default table with Albersheim noise; sanity assert on POI and Pfa | CK2 command below passes; commit, push | 0.5 |

## Day 4 (~8.5 h) — goal: mandatory scope complete (CK3): both baselines, full CLI, verify_sim, `baselines_v0.1` committed

| ID | File | What | Done when | h |
|---|---|---|---|---|
| S16 | `tests/test_oracle.py` | Property tests parametrized over random seeds of the real env with `randomize` on (in-memory dict scenarios, N=50, T=2000; Hypothesis optional): 0 <= pd, pfa, poi <= 1; n_caught <= n_bursts; effective_pd <= sensor_pd; `StayPut` on a burst channel with `PassThroughReceiver` gives POI 1.0, Pfa 0; all-blind history → NaN safe; `upto_t` equals metrics of a truncated history | `python -m pytest tests/test_oracle.py -q` green in < 30 s | 1.5 |
| S17 | `baselines/weighted_priority.py`, `tests/test_baselines.py` | `WeightedPriority(dwell_steps=None)`: smooth weighted round-robin (current += w; pick argmax; current[pick] -= sum(w)) over `spec.intel_weights`, uniform (== RoundRobin) when None, each pick held for dwell_steps; open-loop, no rng | Visit counts over 10 000 slots within 2 % of weights; identical actions for any obs stream; every channel visited within N*dwell slots | 1.5 |
| S18 | `eval/run_comparison.py`, `eval/runner.py` | Full CLI per C9: `--heldout-seeds` with disjointness assert, split columns, `--extra alias=pkg.mod:Class` via importlib, `--scheduler-kwargs`, `schedulers.REGISTRY` discovery, `needs_truth` gate, CSV with per-seed rows, non-zero exit naming (scheduler, scenario, seed) on failure | The C9 example command with `--extra rr2=baselines.round_robin:RoundRobin` prints rows for all three labels; `--schedulers nope` fails with the list of names | 2.5 |
| S19 | `tests/verify_sim.py` | Black-box analytic suite written from INTERFACE.md against `env.spectrum_env.make_env` with the env source closed (in-memory dict scenarios; a disagreement is a contract question, resolved in INTERFACE.md first, then in code): cases listed under Verification; each prints `configured vs recovered` under `pytest -s` | `python -m pytest tests/verify_sim.py -q -s` green, runtime < 60 s | 2.0 |
| CK3 | `results/baselines_v0.1.{md,csv}` | Full green run; write baselines_v0.1 (train_default + heldout_mix, seeds 0-9 / 100-109, round_robin + weighted_priority) with git SHA, scenario hashes, seed lists in header; commit (build doc §6 step 2: numbers saved before any RL) | Re-running the command reproduces the numbers bit-for-bit; commit `feat: baselines v0.1`, push | 1.0 |

## Day 5 (~9.5 h) — goal: extras, hand-off docs, tag `v0.1-sim`

| ID | File | What | Done when | h |
|---|---|---|---|---|
| S20 | `scenarios/heldout_stale_intel.json`, `scenarios/snr_sweep_{-5,0,5,10,15}.json`, `scenarios/blind_zone.json` | stale_intel = train mix with `intel_weights` boosting channels that are quiet here; snr_sweep_X = one always-on periodic radar (onoff [0,T-1,1]) at X dB, tau_switch 0; blind_zone = N=10, one radar onoff [1,1,19] (PRI 20 == N*(tau+1)) so a dwell-2 round-robin always arrives one slot late | All load; `--snr-sweep` prints sensor Pd .03/.10/.21/.81/>.999 ± .02; blind_zone round_robin POI == 0 and weighted_priority (weight on that channel) POI > 0 | 1.5 |
| S21 | `baselines/clairvoyant.py` | `needs_truth = True`, labelled CHEATING / upper bound; each step picks the channel of the earliest-starting uncaught burst active at t+tau_switch; eval passes `env.get_truth()` only to built-in classes with `needs_truth` | clairvoyant POI >= every other row on every seed | 1.0 |
| S22 | `tests/test_protocol_conformance.py` | `pytest --scheduler pkg.mod:Class` option (default both baselines): 300 steps on the real env with `scenarios/mvp_periodic.json`, actions int in [0,N), same seed → same actions, `predictions()` well-formed, `needs_truth` False, mean/p99 µs printed | `python -m pytest tests/test_protocol_conformance.py -q --scheduler baselines.round_robin:RoundRobin` passes | 1.0 |
| S23 | `eval/run_comparison.py` | `results/comparison.md` pipe table, `--snr-sweep` (runs `scenarios/snr_sweep_*.json`, prints sensor Pd vs SNR), `--save-logs` npz per (scenario, scheduler, seed) with S, S_by_emitter, actions, obs, blind, rewards; `--episodes k` | npz reloads with `np.array_equal(S, env.get_truth().S)`; sweep table monotone in SNR | 2.0 |
| CK4 | — | stale_intel, blind_zone, snr sweep, clairvoyant, conformance, `--extra` proven | CK4 command below passes; commit, push | 0.5 |
| S24 | `env/spectrum_env.py`, `env/emitters.py` | Performance pass with cProfile at N=50, T=5000 | `reset() < 50 ms`, mean `step() < 20 µs` over 5000 steps (numbers in README) | 1.0 |
| S25 | `README.md`, `INTERFACE.md` (final), numpy-style docstrings in `env/*.py`, `oracle/*.py`, `baselines/*.py` | Hand-off docs: 10-line minimal loop; 15-line BaseScheduler example; conformance command; `--extra` row; dashboard live view (`get_truth`, `get_history`, `compute_metrics(upto_t)`); seeding; splits; metric decisions; limitations paragraph incl. Albersheim blend, agile_freq correction, `reset(seed=None)` never auto-advances; cold-start warning | A scheduler teammate gets a dummy scheduler into the table using only README + INTERFACE.md; a dashboard teammate runs the README snippet from a clean clone in < 5 min | 2.0 |
| CK5 | `results/comparison.md` | Full run incl. stale_intel, blind_zone, clairvoyant; sanity: measured Pfa ≈ 1e-3, blind % matches dwell, weighted_priority < round_robin on heldout_stale_intel, RR POI 0 on blind_zone; clean-clone check; tag `v0.1-sim` | `pytest -q` green at the tag; `results/comparison.md` committed | 0.5 |
| S26 | `env/emitters.py` | Stretch, only if ahead of schedule: `transition_matrix` for AgileJammer | 2-state [[0.9,0.1],[0.1,0.9]] → empirical stay 0.9 ± 0.03; or the limitation is documented if skipped | 1.0 |

## Stubs that keep you moving

- `PassThroughReceiver` (O_t == truth) and `noise_model:"fixed"` (theory-guide 0.85/0.05) make the env runnable on Day 1 before Albersheim exists; Albersheim switches in by JSON key on Day 2 without touching env or harness.
- Inline lambda policies (`lambda t: 7`, switch-every-k) inside `tests/test_env_api.py`, so env tests never depend on `baselines/`.
- `tests/fakes.py` scripted schedulers so verify_sim and metric tests never depend on round-robin luck or on the scheduler team; hand-built `tiny_truth` fixtures prove metrics against known numbers, not against the env's physics.
- `common/types.py` + `common/protocol.py` are the only cross-imports between `env/` and the rest. `BaseScheduler` gives the scheduler team a working default from Day 0.
- **FakeEnv is intentionally absent.** With one builder the real env exists before the oracle is started, so a second implementation of C2 would be 2 h of duplicated contract surface with its own drift risk. If fast CI without the physics is wanted later, it can be added after `v0.1-sim`.

## Integration checkpoints

| When | What must work | Exact command |
|---|---|---|
| Day 0 end (CK0) | 3.12 venv with numba, contract imports, skeleton, tag `contract-v1` pushed | `python -c "import sys,numba,numpy,gymnasium,scipy,pandas,common.types,common.protocol; assert sys.version_info[:2]==(3,12); print('ok')" && git tag --list contract-v1` |
| Day 2 mid-afternoon (CK1) | Real MVP env + runner + round_robin print a non-degenerate table; S5 and S6–S8 tests green | `python -m pytest tests/test_env_api.py tests/test_oracle.py -q && python -m eval.run_comparison --scenarios scenarios/mvp_periodic.json --seeds 0-2 --schedulers round_robin` |
| Day 3 end (CK2) | All three emitters + Albersheim under the harness on train_default; full metric set | `python -m eval.run_comparison --scenarios scenarios/train_default.json --seeds 0-2 --schedulers round_robin --out results/ck2.csv && python -c "import pandas as pd; d=pd.read_csv('results/ck2.csv'); assert 0<d.poi.mean()<1 and 5e-4<d.sensor_pfa.mean()<2e-3"` |
| Day 4 end (CK3) | Whole suite green incl. verify_sim; both baselines on train + heldout_mix with held-out seeds; baselines_v0.1 committed | `python -m pytest -q && python -m pytest tests/verify_sim.py -q -s && python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json --seeds 0-9 --heldout-seeds 100-109 --schedulers round_robin weighted_priority --out results/baselines_v0.1.csv` |
| Day 5 mid-afternoon (CK4) | stale_intel, blind_zone, snr sweep, clairvoyant, conformance, `--extra` proven | `python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json scenarios/heldout_stale_intel.json scenarios/blind_zone.json --seeds 0-4 --heldout-seeds 100-104 --schedulers round_robin weighted_priority clairvoyant --extra rr2=baselines.round_robin:RoundRobin --save-logs && python -m eval.run_comparison --snr-sweep --seeds 0-2 --schedulers round_robin && python -m pytest tests/test_protocol_conformance.py -q` |
| Day 5 end (CK5, tag) | Clean-clone reproducibility and hand-off | `git clone <remote> C:\dev\ssew-clean && cd C:\dev\ssew-clean && py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt && .venv\Scripts\python -m pytest -q && .venv\Scripts\python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json --seeds 0-9 --heldout-seeds 100-109 --schedulers round_robin weighted_priority clairvoyant --out results/comparison.csv && git tag v0.1-sim` |

## Verification

`tests/verify_sim.py` (black-box through `make_env` + scripted schedulers, in-memory dict scenarios, `configured vs recovered` printed; written from INTERFACE.md, not from the env source):
1. **PRI recovered:** single periodic radar, PassThroughReceiver, `StayPut(channel)`: autocorrelation of obs peaks at lag Y+Z, interior run lengths == Y, gaps == Z, burst count == floor((T-start)/(Y+Z)) ± 1.
2. **Rotation recovered:** single lighthouse radar: burst starts spaced exactly T_rot; run length == analytic width ± 1.
3. **Hop recovered:** jammer with p_stay 0.5: channel changes only at multiples of hop_period, stay fraction 0.5 ± 0.03, occupancy vs hop_probs ± 0.03, one channel per active slot.
4. **Overlap:** jammer hop set containing a radar channel: `S == OR(S_by_emitter)`, `SNR == max`, `E == argmax`, both emitters get bursts.
5. **Noise rates:** `StayPut` on an always-on radar at 10 dB, tau_switch 0, T=40 000: measured sensor Pd within 3 binomial sigma of `albersheim_pd(10,1e-3)`; `StayPut` on an empty channel: measured Pfa within 3 sigma of 1e-3; fixed mode reproduces 0.85/0.05.
6. **Blindness:** `SwitchEvery(k)` for tau in {0,1,2,3}: `n_blind == n_switches*tau`, obs 0 on every blind dwell, no tp while blind, `blind_remaining` post-decrement sequence.
7. **Common random numbers and determinism:** two different action sequences, same seed → identical `S` and `U`; same actions → identical History; seed 0 vs 1 differ; train and held-out seed sets disjoint.
8. **No truth leak:** env info key set == `INFO_KEYS`; `sanitize_info` of an info dict with an injected `"S"` key drops it; observation is a Python int in {0,1}; `check_env` passes.
9. **Hand example through the real env:** `Scripted` actions on a single-radar scenario with PassThroughReceiver reproduce hand-computed POI, TTI, effective vs sensor Pd.
10. **Low-SNR curve and blind zone:** `--snr-sweep` Pd monotone, Pd(-5) 0.03 ± 0.02, Pd(15) > 0.999; blind_zone: round_robin POI == 0 while weighted_priority > 0 (theory guide §1.3 talking point).

End-to-end: the CK5 command from a clean clone. Expect a table with train/held-out cells, clairvoyant on top, weighted_priority below round_robin on heldout_stale_intel, sensor Pfa ≈ 1e-3, blind % consistent with dwell length.

## If Day 5 is lost

Everything mandatory exists at CK3 (end of Day 4): all emitters, Albersheim + blend, blindness, both baselines, full metric set, verify_sim, held-out seeds and held-out mix, baselines_v0.1 committed, `--extra`, latency column. Cut in this order: transition_matrix (S26), band_sweep, `--episodes`, `--save-logs`, markdown output, perf pass (S24), clairvoyant (S21), conformance test (S22; hand the scheduler team the README minimal loop on `mvp_periodic.json` instead), stale_intel/blind_zone/snr_sweep scenarios (S20; quote the Pd-vs-SNR reference values from S10 instead). Never cut: POI, TTI, sensor/effective Pd, Pfa, blindness test, held-out cells, the table, the README minimal loop.

If a day slips earlier than Day 5, hold the order and let the whole schedule shift by that day; do not reorder to reach a checkpoint sooner, because every later step depends on the one before it.

## Hand-off to scheduler and dashboard teammates (at `v0.1-sim`, end of Day 5)

- **Minimal loop (README):** `env = make_env('scenarios/train_default.json', seed=0); obs, info = env.reset(seed=0); sched.reset(env.mission_spec, seed=0); while True: a = sched.select_action(obs, sanitize_info(info)); obs, r, term, trunc, info = env.step(a); if trunc: break; m = compute_metrics(env.get_truth(), env.get_history(), sched.predictions())`.
- **Scheduler team:** subclass `common.protocol.BaseScheduler`, implement `reset(spec, seed)` and `select_action(obs, info) -> int` (obs = O_{t-1}, info keys = INFO_KEYS, never truth); optional `predictions()` fills correct-pred % and avg pred error (ML doc §14 target <= 1 dwell; delta_guard=1); derive randomness from `default_rng([seed, 20_000])`; dwell `spec.tau_switch+1` slots per channel in the cold-start sweep and drop `info["blind"]` observations from tau=1 counts; do not condition runtime decisions on `info["reward"]`. Register in `schedulers/__init__.py::REGISTRY` or use `--extra whittle=schedulers.whittle_rmab:WhittleScheduler`; run the conformance test before asking for a merge; µs/decision mean and p99 are reported for the 15 µs budget; Pd vs SNR via `--snr-sweep`. Train on train_default seeds 0–99 only; seeds 100–199 and `heldout_*` files are never used for tuning.
- **Dashboard team:** `env.get_truth().S` (N,T) waterfall, `.E` for per-emitter colouring, `get_history()` for track/obs/blind/new_detect, `compute_metrics(truth, hist, upto_t=t)` for running numbers, `env.render()` text fallback, `--save-logs` npz for replay without a scheduler; channels are 0-based (add +1 for display).
- **Shared venv:** Python 3.12, `requirements.txt` frozen; new packages go through the simulator owner.

## Risks

| Risk | Mitigation |
|---|---|
| Only Python 3.14 installed; Numba unsupported; numpy absent | Day 0 steps 1–2: 3.12 venv, numba first, CK0 asserts version; `requires-python <3.13` |
| Repo under OneDrive corrupts `.git`/`.venv` | Repo at `C:\dev\smart-scan-ew`; GitHub remote is the backup |
| Single builder, nothing runs in parallel; env is the long pole | Env first; a real table exists by Day 2 afternoon; mandatory scope closes at CK3 (Day 4); Day 5 is entirely cuttable |
| A day is lost to illness or another subject | Order is fixed, schedule shifts; cut list above; the scheduler team codes against `contract-v1` and the README loop from Day 0 and never waits on Day 5 items |
| No second pair of eyes on the contract | `INTERFACE.md` is written before any code; verify_sim is written from INTERFACE.md with the env source closed; hand-built tiny fixtures and theory-guide worked numbers are unit tests; contract beats code |
| 1-dwell sweeps are 100 % blind at tau_switch=1 (degenerate baseline rows; ML doc §7 cold start) | `dwell_steps = tau_switch+1` default in both baselines; blind % always printed; cold-start warning in the hand-off |
| Harmonic blind zone makes a reference row read 0 by accident | Reference PRIs/T_rot coprime-ish with 100; the effect is isolated in `blind_zone.json` as a deliberate demo |
| Scanning radar sidelobes create spurious bursts / wrong beamwidth convention | Mainlobe mask + relative `presence_gain_db`; 3-dB convention `x = 2.783*delta/bw`; lighthouse is the reference mode |
| Albersheim transcription error or floor at Pd ≈ 0.07 at low SNR | 13.1 dB golden, round-trip, reference Pd values, monotone test, documented blend below the Pd=0.1 edge, `--snr-sweep` artifact |
| Truth leaks into a scheduler | Env info == INFO_KEYS by construction, `sanitize_info` whitelist + leak test, `needs_truth` gate |
| Metric definition disputes (blind in Pd, burst vs time POI, prediction scoring) | Both Pd forms and both POI forms printed; C8 fixes every rule; theory-guide worked numbers are unit tests; confusion counts let a judge recompute |
| Days 3 and 5 are ~9–9.5 h | Accepted for the push; S24 and S26 are the pressure valves, and S23 sub-features (`--episodes`, `--save-logs`) can drop individually |

## Critical files for implementation

- `INTERFACE.md` with `common/types.py` and `common/protocol.py` — the frozen contract everything else codes against
- `env/spectrum_env.py` — Gymnasium env: step order, blind/noise/reward, History, truth access
- `env/emitters.py` — PeriodicRadar, AgileJammer, ScanningRadar, truth rendering with common random numbers
- `oracle/metrics_engine.py` — every metric definition the judges will check
- `eval/run_comparison.py` — the single comparison table on identical and held-out seeds
