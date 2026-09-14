# INTERFACE.md — Simulator contract v1 (frozen at tag `contract-v1`)

Contract beats code. If the harness and the env disagree, this file decides, and the fix goes to whichever side drifted. Adding a field with a default is allowed; renaming or removing is not before tag `v0.1-sim`. Any change edits this file, `common/` and every implementer in the same commit.

Import discipline: `oracle/`, `baselines/` and `eval/` import nothing from `env/` except `make_env` (and `PassThroughReceiver` / `ReceiverModel` for test receivers). `env/` never imports `oracle/`, `baselines/` or `eval/`. Channels and slots are 0-based in code (the dashboard adds +1 for display).

Reference numbers: N=50, T=5000, dwell_s=1e-3, tau_switch=1, Pfa=1e-3, R_new=+10, R_hit=+2, C_empty=-0.5, C_tune=-1.5. Train seeds 0–99, held-out seeds 100–199, held-out scenario files start with `heldout_`.

## C1. Shared types — `common/types.py`

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

## C2. Environment API — `env/spectrum_env.py`

```python
def make_env(scenario: str | Path | dict, seed: int, receiver: ReceiverModel | None = None) -> SpectrumEnv
class SpectrumEnv(gymnasium.Env):
    action_space = Discrete(N); observation_space = Discrete(2); mission_spec: MissionSpec
    def reset(self, *, seed=None, options=None) -> tuple[int, dict]    # (0, info0); seed None reuses make_env seed
    def step(self, action: int) -> tuple[int, float, bool, bool, dict]  # (O_t as Python int, R_t, terminated=False, truncated=(t==T), info)
    def get_truth(self) -> Truth; def get_history(self) -> History; def render(self) -> str  # ASCII waterfall
```

`info0 = {"t": -1, "next_t": 0, "action": initial_channel, "switched": False, "blind": False, "blind_remaining": 0, "reward": 0.0, "reward_terms": {"new":0,"hit":0,"empty":0,"tune":0}}`. Per step, `t` = index of the slot just executed. `reward_terms` are signed floats with `sum(reward_terms.values()) == reward`. Step after truncation (or before `reset()`) raises `RuntimeError`; an action outside `[0, N)` raises `ValueError`. `reset(seed=None)` reuses the `make_env` seed and never auto-advances it; `reset(seed=k)` makes `k` the env seed from then on. A `receiver` passed to `make_env` overrides the scenario's receiver block, and `mission_spec.tau_switch` / `.pfa` report the receiver actually in use. `Truth` arrays are read-only; `History` arrays are live views.

Exact `step(a)` order for slot t:
1. `O_t, switched, blind, blind_remaining = receiver.observe(a, S[a,t], SNR[a,t], U[a,t])`. The receiver is the single owner of prev-channel and blind state (prev at reset = `initial_channel`, so no tune cost at t=0 if the scheduler starts there). Blind rule: on switch `blind_left = tau_switch`; `blind = blind_left > 0`; if blind, `blind_left -= 1`; `blind_remaining` reported post-decrement; a switch during blindness restarts the counter; tau_switch=0 never blinds. Blind → `O_t = 0`; else `O_t = int(U[a,t] < p)` with `p = pd(SNR[a,t])` if `S[a,t]==1` else `pfa`.
2. `I_new = 1` iff `O_t==1 and S[a,t]==1` and some emitter active in that cell was never truly detected before; all active ones are then marked seen.
3. `R_t = R_NEW*I_new + R_HIT*O_t - C_EMPTY*(1-O_t) - C_TUNE*switched`. Worked cases: switch + first true hit = 10.5; stay + known hit = 2.0; switch + empty = -2.0; stay + empty = -0.5. (A false alarm earns R_hit and a blind dwell pays C_empty: stated limitation.)
4. Write `History[t]`; `t += 1`; `truncated = (t == T)`; build info with exactly `INFO_KEYS`.

## C3. Receiver — `env/receiver_model.py`

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

## C4. Emitters — `env/emitters.py`

All open-loop. `render(N, T) -> (S_m int8 (N,T), SNR_m float32 (N,T))`, vectorised.

- `PeriodicRadar(channel, onoff=[start_delay, on, off], snr_db)`: `S_m[channel,t] = 1 iff t >= start and ((t-start) % (on+off)) < on`; PRI = on+off. `randomize.periodic_phase` adds `rng.integers(0, PRI)` to start.
- `AgileJammer(channels, hop_period, hop_probs|None, p_stay, onoff, snr_db)`: every `hop_period` slots keep the channel with prob `p_stay`, else draw from `channels` with `hop_probs` (uniform if null) excluding the current one; emits only while onoff-active; exactly one channel per active slot. Because the draw excludes the current channel, long-run occupancy equals `hop_probs` only when they are uniform; `AgileJammer.stationary_distribution()` returns the exact occupancy. `hop_probs` must sum to 1. Optional `transition_matrix` (row-stochastic over `channels`) replaces the `p_stay`/`hop_probs` rule. `randomize.agile_start_channel` draws the first channel from `hop_probs`; otherwise it is `channels[0]`.
- `ScanningRadar(mode="lighthouse", channel, T_rot, beamwidth_deg, phase0_deg, snr_peak_db, presence_gain_db=-10.0)`: `theta(t) = (360*t/T_rot + phase0) mod 360`, receiver at azimuth 0, `delta = wrap(theta(t))`. `SNR_m = snr_peak_db + 10*log10(sinc2_gain(delta, bw))`; `S_m = 1 iff |delta| < 1.129*bw (mainlobe only) and SNR_m - snr_peak_db >= presence_gain_db` (first sidelobe is -13.3 dB, never registers). Gives exactly one burst per rotation of length `round(T_rot * 1.667*bw / 360) +- 1` at -10 dB. `mode="band_sweep"` is optional.
- Overlap: `S = OR`, `SNR = max`, `E = argmax SNR`. Burst = maximal run of 1s per (emitter, channel) in `S_by_emitter`.

## C5. Scenario JSON — `scenarios/*.json`, `env/scenario.py`

`load_scenario(path|dict) -> ScenarioConfig`, `to_mission_spec(cfg) -> MissionSpec`.

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

## C6. Seeding and common random numbers

`make_env(scenario, seed)`: emitter i uses `np.random.default_rng([seed, i])`; `U = default_rng([seed, 10_000]).random((N,T), dtype=float32)`; schedulers derive randomness from `default_rng([seed, 20_000])` inside `reset()`. Same (scenario, seed) → bit-identical S, E, SNR, U for every scheduler, so the comparison is paired.

## C7. Scheduler protocol — `common/protocol.py`

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

## C8. Oracle — `oracle/`

`MetricsResult` lives in `oracle/metrics_engine.py` (defaulted fields may be added freely); core keys frozen here.

```python
def segment_bursts(truth) -> list[Burst]
def dwell_ledger(truth, hist, upto_t=None) -> dict[str, np.ndarray]   # tp fa fn tn blind switched (bool, len t)
def compute_metrics(truth, hist, predictions=(), upto_t=None, delta_guard=1, dwell_s=1e-3) -> MetricsResult   # .to_dict() flat; dwell_s only feeds intercept_rate_per_s
METRIC_KEYS = sensor_pd effective_pd sensor_pfa n_tp n_fp n_fn n_tn n_blind  poi poi_periodic poi_agile poi_scanning poi_time n_bursts n_caught
              mean_tti median_tti intercept_rate_per_dwell intercept_rate_per_s discovery_ratio  n_predictions n_dropped_predictions correct_pred_pct avg_pred_time_error
              total_reward mean_reward blind_fraction switch_rate n_switches  us_per_decision_mean us_per_decision_p99 (filled by eval)
```

Definitions (theory guide Part 6): `s_at = S[a_t,t]`; tp/fp/fn/tn on non-blind dwells only. `sensor_pd = tp/(tp+fn)`; `effective_pd = tp / #(s_at==1)` (blind counted as misses); `sensor_pfa = fp/(fp+tn)`. Burst caught iff exists t in [t_start,t_end] with `a_t == channel` and tp[t] (a blind dwell or false alarm never catches); `poi = n_caught/n_bursts` where `n_bursts` counts bursts with `t_start < t` (every burst once `t == T`; `n_bursts_total` is the whole-episode count), by type via `Burst.emitter_type`, NaN for absent types; `poi_time = #{(m,t): S_by_emitter[m,a_t,t]==1 and tp[t]} / S_by_emitter[:, :, :t].sum()` (denominator over the slots executed so far; equals `S_by_emitter.sum()` at `t == T`). `tti = t_first_tp - t_start` over caught bursts (mean and median). `intercept_rate_per_dwell = n_caught/t`, `per_s = n_caught/(t*dwell_s)`. `discovery_ratio = emitters with >= 1 tp / M`. Predictions: only `status=="issued"` scored; correct iff any `S[c, t_pred-delta_guard .. t_pred+delta_guard] == 1`; error = `|t_pred - nearest burst start on c|`, averaged over predictions whose channel has >= 1 burst; `t_pred` outside [0,T) → incorrect, excluded from error; dropped counted only in `n_dropped_predictions`. All zero denominators → NaN, never an exception.

## C9. Eval CLI — `eval/run_comparison.py`

```
python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json --seeds 0-9 --heldout-seeds 100-109
   --schedulers round_robin weighted_priority clairvoyant [--extra whittle=schedulers.whittle_rmab:WhittleScheduler]
   [--scheduler-kwargs '{"whittle": {"beta": 0.95}}'] [--episodes 1] [--save-logs] [--snr-sweep] [--out results/comparison.csv]
```

Every scheduler runs every scenario on both seed lists (asserted disjoint). Row = (scheduler, scenario, seed_split); `scenario_split` = heldout iff file name starts with `heldout_`. Row label = registry key or `--extra` alias. Header prints git SHA, sha256[:8] of each scenario file, seed lists, numpy/python versions. Columns: POI, poi_time, sensor Pd, effective Pd, sensor Pfa, mean/median TTI, intercepts/s, correct-pred %, pred err, total reward, blind %, switches, us/decision mean and p99 (mean ± std over seeds). Per-(scheduler, scenario, seed, episode) rows go to CSV (`--out`); `--md` writes the pipe table. Registry auto-discovers `schedulers/__init__.py::REGISTRY` (ImportError ignored). `--extra` classes with `needs_truth=True` are refused; the runner passes `env.get_truth()` only to built-in classes with `needs_truth` (via `set_truth`). `--episodes k` keeps the env seed and offsets the scheduler seed by `e * 1_000_000` for episode `e`. `--save-logs` writes `results/logs/<scenario>__<label>__seed<k>_ep<e>.npz` with `S, S_by_emitter, E, SNR, actions, obs, blind, switched, new_detect, rewards`. A failing episode exits non-zero naming (scheduler, scenario, seed).

## Contract amendments since `contract-v1` (all additive, no renames or removals)

Every entry adds a field, a default, or a clarification; nothing that existed at `contract-v1` changed meaning. Implementers shipped in the commit named; the wording here caught up in `0b392ad` and later, so from `v0.1-sim` onwards any further amendment lands in the same commit as its implementer.

| Clause | Amendment | Shipped in |
|---|---|---|
| C2 | `step()` before `reset()` raises `RuntimeError`; action outside `[0, N)` raises `ValueError`; `reset(seed=k)` makes `k` the env seed; a `receiver` passed to `make_env` overrides the scenario receiver and `mission_spec` reports it; `Truth` arrays are read-only, `History` arrays are live views | `8d1f380` |
| C3 | `PassThroughReceiver(pfa=1e-3)` accepts an optional `pfa`; `albersheim_pd` / `sinc2_gain` accept arrays; a receiver that was never `reset()` treats its first dwell as no switch | `8d1f380` |
| C4 | Long-run jammer occupancy equals `hop_probs` only when uniform (exclude-current rule); `AgileJammer.stationary_distribution()`; `hop_probs` must sum to 1; `transition_matrix` supported; `randomize.agile_start_channel` semantics | `8d1f380` |
| C5 | Validator also requires unique emitter names, distinct agile channels, `p_stay` in [0,1], `presence_gain_db <= 0`, `beamwidth_deg` in (0,360); `name`, `dwell_s`, `reward`, `randomize`, `intel_weights` and the optional receiver keys have defaults; `to_mission_spec(cfg, tau_switch=None, pfa=None)` | `8d1f380` |
| C8 | `compute_metrics(..., dwell_s=1e-3)`; `n_bursts` counts bursts with `t_start < t` plus `n_bursts_total`; `poi_time` denominator over slots `< t`; `dwell_ledger` also returns `fp`, `s_at`, `obs`, `actions`, `t`; `MetricsResult` extras `n_dwells`, `n_emitters`, `n_bursts_total`, `caught_bursts` | `98bffbf` |
| C9 | `--md`, `--delta-guard`, `--quiet`; `--episodes` offsets the scheduler seed by `e * 1_000_000`; `--save-logs` file naming and extra npz keys; `Clairvoyant.set_truth()` is how the runner injects truth into built-in cheaters | `98bffbf` |

Import discipline as practised: `oracle/`, `baselines/` and `eval/` import only `make_env` from `env/` (tests may additionally import `PassThroughReceiver`, `ReceiverModel` and `albersheim_pd` as instruments); `baselines/clairvoyant.py` imports `oracle.truth_tracker.segment_bursts`, which is oracle code, not env code.

## Stated limitations (say them first)

Open-loop non-adversarial emitters; single-channel receiver; single-pulse Albersheim Bernoulli approximation with a documented low-SNR blend; S pre-rendered at `reset()`; sensor Pd excludes blind dwells while effective Pd includes them (both printed); a false alarm earns R_hit and a blind dwell pays C_empty.
