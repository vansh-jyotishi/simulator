# The ML Scheduler — Complete Build Document
## SIH Problem Statement 26055 — Smart Scan Strategy for Electronic Warfare

### The single source of truth for building the scheduler: what it is, what it does, how it does it, and every design decision, fix, and open question resolved along the way.

---

## 0. Reference Scenario (concrete numbers used throughout this doc)

Every formula, data structure, and latency estimate below is grounded against these fixed reference values so nothing stays symbolic and unverifiable:

| Parameter | Value | Reason |
|---|---|---|
| N (number of channels) | 50 | Realistic surveillance band count for a demo-scale ES receiver |
| τ_max (max tracked dwell age) | 1000 | Long enough that belief has fully relaxed to stationary distribution well before this point |
| β (discount factor) | 0.95 | Standard choice for infinite-horizon discounted bandit problems; empirically tunable |
| ε (indexability clamp epsilon) | 0.01 | Minimum enforced gap between P11 and P01 estimates |
| K (LUT recompute interval) | every 10 new observations per channel | Balances freshness of index values against recompute cost |
| Base dwell period (Δt) | 100 µs – 1 ms | Receiver hardware dwell window per channel check |
| Hard decision latency budget | < 15 µs | Must not exceed one dwell period's worth of compute overhead |

---

## 1. What The Scheduler Is

**In one sentence:** the scheduler is a function that runs once per dwell slot and outputs exactly one integer — the channel index the receiver hardware should tune to next.

It is the "brain" sitting directly above the receiver hardware interface, replacing the old open-loop fixed-sweep logic with a closed-loop decision system that uses its own accumulated observations to decide, every single time step, where to look next.

**What it is NOT:** it is not a single black-box neural network. It is a layered, mostly classical-math system with one narrow, well-justified machine-learning component (online Bayesian/MAP parameter estimation) at its core — not a deep learning model. This is a deliberate design choice, explained fully in Section 9.

---

## 2. What The Scheduler Does — The Full Decision Loop

At every discrete time slot `t`:

1. Receives the noisy observation `O_t ∈ {0, 1}` from the receiver hardware, reporting what happened on the channel it was tuned to during the previous slot.
2. Updates its internal belief and statistics about that channel.
3. Checks whether a high-confidence periodic signal is due on any channel right now.
4. If yes — forces a dwell on that channel (predictive ambush).
5. If no — computes an "urgency score" (Whittle Index) for every channel and picks the highest one.
6. Dispatches the chosen channel index to the receiver hardware as the next tuning command.
7. Logs the decision and outcome for metrics.

This repeats continuously for the entire mission duration.

---

## 3. Full System Architecture Diagram

```
+──────────────────────────────────────────────────────────────────────────────────────────+
|                                    RECEIVER HARDWARE                                     |
|                       Delivers: Noisy Observation O_t ∈ {0, 1}                           |
|                       Accepts:  Tuning Command a_t ∈ {1, ..., N}                         |
+───────────────────────────────┬──────────────────────────────────▲───────────────────────+
                                 │                                  │
                                 ▼                                  │
+──────────────────────────────────────────────────────────────────┴───────────────────────+
|                                 ML SCHEDULER SUBSYSTEM                                    |
|                                                                                            |
|  1. INGESTION & COLD-START BOOTSTRAP                                                      |
|     • If t <= N: Dispatch sequential round-robin action a_t = t.                         |
|     • If t > N: Route observation O_t to active channel tracker.                          |
|                               │                                                            |
|                               ▼                                                            |
|  2. PER-CHANNEL BELIEF & STATE TRACKER                                                    |
|     • Update dwell age: tau_n += 1 for all n != a_(t-1); tau_(a_(t-1)) = 0.                |
|     • Update online transition counters: N_{i->j}^(a_(t-1)) (Path B: tau=1 revisits only). |
|     • Compute MAP transition estimates: P01_hat^(n), P11_hat^(n).                          |
|     • Apply indexability clamp: if P11_hat < P01_hat, P11_hat = P01_hat + epsilon.          |
|     • Update belief: b_t(n) via multi-step power-law projection.                          |
|                               │                                                            |
|               ┌───────────────┴───────────────┐                                           |
|               ▼                               ▼                                           |
|  3. PERIODICITY ENGINE            4. WHITTLE RMAB CORE POLICY                             |
|     • Sparse hit buffer                       • Per-channel LUT, recomputed every K        |
|     • Lomb-Scargle transform                    new observations for that channel          |
|     • Period/phase lock (T_rot_hat, t0_hat)   • Lookup: W_n = LUT[n, tau_n, s_last]        |
|     • Reservation queue with tie-break         • Complexity: O(1) retrieval per channel     |
|       priority (peak spectral power)          • Computes candidate: a_whittle = argmax W_n |
|               │                                       │                                    |
|               └───────────────┬───────────────────────┘                                    |
|                               ▼                                                            |
|  5. RUNTIME PRIORITY ARBITER (VETO LAYER)                                                 |
|     • Check: Is any channel scheduled for a high-confidence ambush dwell at t?             |
|       - YES → Force a_t = n_periodic (Predictive Override). Whittle candidate discarded.   |
|       - NO  → Select a_t = a_whittle (Core RMAB Policy).                                   |
|                               │                                                            |
|                               ▼                                                            |
|  6. ACTION DISPATCHER & METRIC RECORDER (async, non-blocking)                              |
|     • Emit command a_t to receiver local oscillator.                                       |
|     • Forward decision context to metrics engine (Pd, Pfa, POI, delta_t_err) via async     |
|       push — does NOT block the next decision cycle.                                       |
+──────────────────────────────────────────────────────────────────────────────────────────+
```

**Critical-path note:** Stages 1–5 (ingestion through arbitration + dispatch) form the hard real-time critical path and must complete within the 15 µs budget. Stage 6's telemetry hand-off is fire-and-forget asynchronous and does not block the next decision cycle — it is shown in the diagram for completeness but excluded from the critical-path latency total.

---

## 4. The Arbitration Engine — Decision Tree

The scheduler resolves "which policy actually gets to choose" through a strict two-layer hierarchy — never competing models, always a clean fallthrough:

```
========================================================================================
                          RUNTIME ARBITRATION TREE (PER TIME SLOT)
========================================================================================

                 Has the Periodicity Estimator locked onto a confirmed
                 transmission window for channel n at time t?
                                  │
                 ┌────────────────┴────────────────┐
                 │ YES                              │ NO
                 ▼                                  ▼
       [PREDICTIVE OVERRIDE]              [CORE RMAB POLICY]
       Force dwell on channel n.          Compute Whittle Index W_n for all n in {1...N}.
       Whittle candidate discarded        Tune to channel a_t = argmax W_n.
       silently, no merge logic needed.
========================================================================================
```

**Collision case — two periodic reservations, same time slot, different channels:** resolved by priority tie-break. The reservation with the higher Lomb-Scargle peak spectral power (i.e., higher-confidence lock) wins. The losing reservation is not silently dropped — it is logged as a known, accepted miss for honest POI accounting, and re-queued for its next predicted cycle.

```python
def resolve_reservation_collision(t, new_channel, new_priority, reservation_schedule, priority_of):
    existing = reservation_schedule[t]
    if existing == -1:
        reservation_schedule[t] = new_channel
        priority_of[t] = new_priority
    elif new_priority > priority_of[t]:
        log_known_miss(existing, t)  # honest metrics accounting
        reservation_schedule[t] = new_channel
        priority_of[t] = new_priority
    else:
        log_known_miss(new_channel, t)
```

---

## 5. Core Policy — Whittle Index (Restless Multi-Armed Bandit)

### 5.1 Why This Framework

Enemy emitters turn on, off, or hop channels regardless of whether the receiver is currently watching them. This makes the spectrum a **Restless Multi-Armed Bandit (RMAB)** — unlike a classic bandit, unwatched "arms" (channels) keep evolving on their own. Combined with the fact that the receiver only ever observes the one channel it's tuned to, this is also a **Partially Observable Markov Decision Process (POMDP)**.

### 5.2 The Channel Model

Each channel `n` is modeled as an independent, discrete-time, two-state Markov chain (Gilbert-Elliott channel model):

- State space: `S = {0, 1}` — 0 = idle/silent, 1 = active transmission.
- Transition matrix:

```
P(n) = | P00(n)  P01(n) |   =   | 1-P01(n)   P01(n)  |
       | P10(n)  P11(n) |       | 1-P11(n)   P11(n)  |
```

Under positively correlated dynamics (`P11(n) >= P01(n)` — an active channel tends to stay active, a quiet one tends to stay quiet), Liu & Zhao (2010) prove this two-state restless bandit is **Whittle-indexable**, meaning a well-defined, provably near-optimal index policy exists.

**Literature anchor:** Liu, K., & Zhao, Q. (2010). "Indexability of Restless Bandit Problems and Optimality of Whittle Index for Dynamic Multichannel Access." *IEEE Transactions on Information Theory.* — Before finalizing this section in any judge-facing document, verify the closed-form expression below term-by-term against the original paper; this is a paraphrased derivation and should be double-checked against source, not trusted from memory alone.

### 5.3 Belief State Update

When channel `n` was last observed `τ_n` steps ago in state `s_last(n)`, its belief `b_t(n) = P(S_t(n) = 1)` relaxes toward the stationary distribution:

```
b_t(n) = p_stat(n) + (s_last(n) - p_stat(n)) * (P11(n) - P01(n))^tau_n

where p_stat(n) = P01(n) / (P01(n) + P10(n))
```

### 5.4 The Whittle Index Formula

```
W_n(b) = (b_t(n) - p_stat(n)) / (1 - beta*(P11(n) - P01(n))) + p_stat(n)
```

**Plain-English intuition:** the index answers "if I had to put a price on checking this channel right now instead of resting, what would that price be?" Channels ignored a long time accumulate rising uncertainty (index rises, forcing exploration). Channels known active keep a high index (continued exploitation). Channels confidently believed quiet get a low index.

### 5.5 Indexability Safeguard (Critical Fix)

The Liu & Zhao proof requires `P11 >= P01`. Online MAP estimation from sparse early data can produce estimates that violate this by chance, silently breaking the mathematical guarantee without crashing anything — the formula still returns a number, it just stops meaning what it's supposed to mean.

**Fix — enforced clamp, applied immediately after every MAP update:**

```python
epsilon = 0.01
if p11_hat < p01_hat:
    p11_hat = p01_hat + epsilon
```

This is stated explicitly as a deliberate design choice: enforced positive correlation guarantees the indexability precondition holds even under sparse, noisy early estimates.

---

## 6. Transition Probability Estimation (Online Learning)

Because no prior intelligence exists, `P01(n)` and `P11(n)` are unknown at deployment and must be estimated online from accumulating observations.

### 6.1 Method: MAP Estimation with Laplace Smoothing (Path B — Consecutive-Revisit Counting)

**Design decision, explicit:** transition counters are updated only when two consecutive dwells on the *same* channel are genuinely one time-step apart (`τ = 1` at time of revisit). Longer gaps between visits to the same channel are **not** used to update the transition probability estimates directly — they are only used for belief decay via the multi-step power-law projection formula in Section 5.3.

This is a deliberate simplification. The mathematically complete alternative (gap-aware Maximum Likelihood Estimation that correctly accounts for arbitrary gap lengths via the matrix-power relation) is real, solvable, and documented here as future work — but is not required for a working, defensible first build. Stating this limitation explicitly, rather than leaving it ambiguous, is what makes the design honest and defensible under questioning.

```
P01_hat(n) = (N_{0->1}(n) + 1) / (N_{0->0}(n) + N_{0->1}(n) + 2)
P11_hat(n) = (N_{1->1}(n) + 1) / (N_{1->0}(n) + N_{1->1}(n) + 2)
```

The `+1` / `+2` terms are Laplace (non-informative Dirichlet) smoothing, preventing zero-frequency artifacts during early mission intervals when counts are still small.

### 6.2 Future Work (documented, not built for v1)

Gap-aware MLE: fit `P01`, `P11` via numerical likelihood maximization that correctly weights each observation by its actual elapsed gap `τ`, using the same `P^τ` matrix-power relation already derived in Section 5.3. This is a well-studied problem (estimation of embeddable discrete-time Markov chains from irregularly sampled data) and would replace the τ=1-only restriction with full data utilization. Left as a stated stretch goal.

---

## 7. Cold-Start Protocol

At mission start (`t = 0`), every channel's transition counts are unpopulated. The scheduler follows a deterministic bootstrap:

```
MISSION TIMELINE:
t = 0                 t = N                   t > N (Steady State)
  |─────────────────────|───────────────────────>
  [Phase 1: Bootstrap]   [Phase 2: Closed-Loop RMAB]
  * Round-Robin Sweep    * Whittle Index ranking
  * Seed transition      * Periodic ambush overrides
    registers            * Online Bayesian updates
```

**Prior initialization at t=0** (symmetric, non-informative):

```
N_{0->1}(n) = 1,  N_{0->0}(n) = 1,  N_{1->1}(n) = 1,  N_{1->0}(n) = 1
=> P01_hat(n) = 0.5,  P11_hat(n) = 0.5
```

Belief states initialize to maximum entropy: `b_0(n) = 0.5` for all `n`.

**Phase 1 ([1, N]):** deterministic sequential sweep, `a_t = t`. Guarantees every channel is sampled once, seeding `s_last(n)` and dwell-age registers.

**Phase 2 (t > N):** full control transfers to the Whittle RMAB policy. Unvisited channels experience rising `τ_n`, causing their uncertainty (and index) to rise — the mathematical structure of the index naturally forces broad exploration of cold channels without needing an external heuristic like epsilon-greedy.

---

## 8. Periodicity Estimator (Predictive Ambush Layer)

### 8.1 Purpose

Targets periodic/rotating emitters (e.g., rotating search radars) specifically — the case where a purely reactive bandit policy would always be one step behind a predictable pattern.

### 8.2 The Problem It Solves

Because the receiver hops across many bands, it only catches scattered, irregular glimpses of any single periodic radar. Standard Fast Fourier Transforms require evenly-spaced samples and break down on this irregular data.

### 8.3 The Tool: Lomb-Scargle Periodogram

Purpose-built for extracting repeating patterns from unevenly sampled time-series data. Implemented via `scipy.signal.lombscargle`.

```
P_LS(omega) = (1/2) * { [sum_k cos(omega*(t_k - tau))]^2 / sum_k cos^2(omega*(t_k - tau))
                       + [sum_k sin(omega*(t_k - tau))]^2 / sum_k sin^2(omega*(t_k - tau)) }

tan(2*omega*tau) = sum_k sin(2*omega*t_k) / sum_k cos(2*omega*t_k)
```

**Detection lock:** a periodic signal is flagged when peak spectral power exceeds a false-alarm threshold: `max_omega P_LS(omega) >= gamma_period`.

Estimated period: `T_rot_hat = 2*pi / omega_peak`. Phase `t0_hat` resolved via least-squares alignment across observed timestamps.

### 8.4 Execution Trigger

Runs only when a channel has accumulated `k >= 4` hits — not evaluated on every tick. Heavy transform work is dispatched to an **asynchronous background worker**, triggered only when a new hit (`O_t = 1`) is logged, publishing updated timing reservations to the main decision thread without blocking the microsecond-scale critical path.

### 8.5 Arbitration Veto

For any time slot `t`, if an active burst is predicted on channel `n*` such that:

```
| t - (t0_hat(n*) + m * T_rot_hat(n*)) | <= delta_guard
```

the Periodicity Estimator overrides the Whittle calculation and forces `a_t = n*`.

---

## 9. Reward Function (Training-Time Only — Not Part of Runtime Inference)

Note: the reward function shapes the *offline training / online parameter learning* process. The deployed Whittle Index policy itself is a closed-form calculation, not a trained neural policy — the reward framing below applies to how the system's learned components (transition estimates, and any offline-compared baselines) are evaluated and shaped.

```
R(s_t, a_t) = R_new * I_new(a_t) + R_hit * O_t - C_empty * (1 - O_t) - C_tune * I(a_t != a_(t-1))
```

| Term | Value | Purpose | Status |
|---|---|---|---|
| `+R_new` | +10.0 | Exploration bonus — first-ever detection of an uncharacterized emitter | Locked |
| `+R_hit` | +2.0 | Baseline positive reward for any valid detection | Locked |
| `-C_empty` | -0.5 | Search-efficiency cost, penalizes dwelling on empty static | Locked |
| `-C_tune` | -1.5 | Hardware switching penalty, models real tuning-latency cost | **Illustrative — pending empirical tuning against simulator, not yet validated by experiment** |

**`R_threat` term — explicitly cut from the core formula.** An earlier draft included a `+R_threat` bonus for re-intercepting emitters tagged as "high-threat." This is removed from the primary reward formula because: (a) this problem statement is about interception, not threat classification, and including it risked implying the system performs threat classification, which is out of scope; (b) if reintroduced later purely as simulator-internal reward shaping metadata (never exposed as an agent output or capability), it must be documented with the explicit caveat: *"Threat tagging, if used, exists only as simulator-internal reward-shaping metadata; the agent has no threat-classification output or capability."* For the current build, this term is simply absent.

---

## 10. Data Structures & Memory Layout

### 10.1 Per-Channel State — Structured Array (Primary Design, Pending Verification)

```python
import numpy as np

channel_dtype = np.dtype([
    ('tau', np.int32),            # Dwell age (steps since last check)
    ('s_last', np.int8),          # Last observed binary state (0 or 1)
    ('n_00', np.int32),           # Transition count: 0 -> 0
    ('n_01', np.int32),           # Transition count: 0 -> 1
    ('n_10', np.int32),           # Transition count: 1 -> 0
    ('n_11', np.int32),           # Transition count: 1 -> 1
    ('p01', np.float32),          # Online estimated P(0 -> 1)
    ('p11', np.float32),          # Online estimated P(1 -> 1)
])

channel_state = np.zeros(N, dtype=channel_dtype)  # N = 50 in reference scenario
```

**Mandatory pre-build verification step — run this before writing any further scheduler code:**

```python
import numba as nb
import numpy as np

test_dtype = np.dtype([('tau', np.int32), ('s_last', np.int8)])
arr = np.zeros(5, dtype=test_dtype)

@nb.njit
def test_access(a):
    return a[0].tau + a[0].s_last

test_access(arr)  # If this compiles clean under nopython mode, structured-array design is safe.
```

**If this fails:** fall back to flat parallel arrays instead of one structured array:

```python
tau = np.zeros(N, dtype=np.int32)
s_last = np.zeros(N, dtype=np.int8)
n_00 = np.zeros(N, dtype=np.int32)
n_01 = np.zeros(N, dtype=np.int32)
n_10 = np.zeros(N, dtype=np.int32)
n_11 = np.zeros(N, dtype=np.int32)
p01 = np.zeros(N, dtype=np.float32)
p11 = np.zeros(N, dtype=np.float32)
```

Uglier, but guaranteed Numba-`nopython`-compatible.

### 10.2 Whittle Index Lookup Table — Per-Channel, Dynamically Recomputed (Corrected Design)

**Earlier design flaw, now fixed:** a single static, offline-precomputed LUT is incompatible with online transition-probability learning — if `P01`/`P11` update continuously, a LUT built once at the start goes stale. The corrected design uses a **small, per-channel LUT, recomputed only when that channel's transition estimate changes meaningfully** (every `K = 10` new observations of that specific channel, not every global tick):

```python
tau_max = 1000  # reference scenario value

# LUT shape: [Channel Index, Elapsed Dwell Age, Last Known State (0 or 1)]
whittle_lut = np.zeros((N, tau_max, 2), dtype=np.float32)

def recompute_channel_lut(n, p01, p11, beta, p_stat):
    for tau in range(tau_max):
        for s_last in (0, 1):
            b = p_stat + (s_last - p_stat) * (p11 - p01) ** tau
            whittle_lut[n, tau, s_last] = (b - p_stat) / (1 - beta * (p11 - p01)) + p_stat

# O(1) retrieval during step execution, with boundary clamp for tau exceeding tau_max:
def get_whittle_score(channel_id, tau, s_last):
    clamped_tau = min(tau, tau_max - 1)
    return whittle_lut[channel_id, clamped_tau, s_last]
```

This keeps O(1) lookup at decision time while staying honestly consistent with the online-learning design — recompute cost is small (only `tau_max * 2` values per trigger) and infrequent (every 10 observations of that specific channel, not every global tick).

**Source of truth, stated explicitly:** `whittle_lut` is the only cached index storage. No duplicate `whittle_index` field is kept on the per-channel state struct — if a debug/logging copy of the last retrieved value is needed, it is clearly labeled as "last retrieved value, for logging only, never read by decision logic."

### 10.3 Periodicity Reservation Queue

```python
# Simulation / training version — fixed horizon, known mission length:
reservation_schedule = np.full(total_time_steps, -1, dtype=np.int32)
priority_of = np.zeros(total_time_steps, dtype=np.float32)

# To schedule an ambush:
reservation_schedule[predicted_burst_slot] = target_channel_idx
priority_of[predicted_burst_slot] = peak_spectral_power
```

**Deployment note, explicit:** this fixed-horizon array is valid for simulation/training only, where total mission length is known in advance. A production/deployed version would replace this with a bounded ring buffer sized to the maximum useful lookahead window (e.g., 500 slots) or a sparse dictionary keyed by relative offset from current time, avoiding unbounded or unknown-length preallocation.

---

## 11. Latency Budget & Real-Time Constraints

Target: complete the full decision cycle in under 15 µs per slot, to keep pace with fast-settling RF synthesizers.

```
SLOT DURATION: delta_t (100 us to 1 ms base dwell)
├── [0.0 - 2.5 us]  : Ingestion of O_t & updating active channel transition counts
├── [2.5 - 5.0 us]  : Dwell age increment (vectorized tau += 1, O(N) op) & reset for a_(t-1)
├── [5.0 - 8.5 us]  : Whittle index retrieval across N channels (LUT slicing / vectorized math)
├── [8.5 - 10.0 us] : Periodicity queue check & arbitration veto selection
├── [10.0 - 12.0 us]: Dispatch tuning command a_t to receiver interface
└── (async, non-blocking, excluded from critical path):
    Telemetry hand-off to logger (ZeroMQ / buffer push)

TARGET CRITICAL-PATH LATENCY: ~12.0 us (< 15 us hard budget)
STATUS: Target budget derived from operation-count reasoning — to be validated against
        actual profiled measurements once the Numba-compiled hot loop is implemented and
        benchmarked on target hardware. Not yet an empirically confirmed figure.
```

**Note on "vectorized tau += 1":** this is a single `O(N)` vectorized NumPy/Numba operation across all channels each tick — not N independent per-channel compute cycles. Stated explicitly to avoid any ambiguity about computational complexity.

---

## 12. Technical Stack

| Subsystem | Technology / Library | Version | Engineering Justification |
|---|---|---|---|
| Language | Python | 3.11+ | Core control logic, rapid prototyping, data-science toolchain integration |
| Numeric Engine | NumPy | 1.26+ | Vectorized operations for ground-truth arrays and belief-state arrays |
| JIT Accelerator | Numba | 0.59+ | JIT-compiles state tracking and Whittle index operations into machine code via `@njit`, targeting <15 µs decision speed. **Must use `nopython=True` / `@njit` explicitly — object-mode fallback silently defeats the latency guarantee and must never be allowed.** |
| Signal Processing | SciPy | 1.12+ | `scipy.signal.lombscargle` for unevenly-spaced spectral period estimation |
| Environment Standard | Gymnasium | 0.29+ | Standard `step()`/`reset()` API, environment-agent decoupling |
| Offline Benchmark Only | Stable-Baselines3 / PyTorch | 2.2+ / 2.2+ | Powers the offline DQN comparison baseline (`eval/run_comparison.py`) — **not part of the runtime decision loop**, see Section 13 |
| Data Pipelines | Pandas / PyArrow | 2.2+ / 15.0+ | Serializes simulation logs for downstream metrics analysis |
| Interactive UI | Streamlit & Plotly | 1.32+ / 5.19+ | Live operational waterfall plot — ground truth, scan tracks, metrics |
| Testing Harness | PyTest & Hypothesis | 8.0+ / 6.98+ | Property-based testing for Markov estimation convergence and mathematical sanity checks |

**Pre-build verification, mandatory:**

```bash
pip install "numpy>=1.26" "numba>=0.59"
python -c "import numba, numpy; print(numba.__version__, numpy.__version__)"
```

Confirm this resolves cleanly with no dependency conflicts before locking these version numbers into any final documentation as settled fact — Numba has historically lagged behind the newest NumPy releases.

---

## 13. Explicit Design Cuts (What We Deliberately Did NOT Build, and Why)

Stating cuts openly, rather than letting them silently vanish between document drafts, is itself part of the defensible design.

### 13.1 Deep Q-Networks (DQN / DRQN) — Removed from Runtime

- **Cold-start latency:** Deep RL agents need thousands of exploratory episodes to converge, making them ineffective during the initial phase of a mission.
- **Computational overhead:** Evaluating neural network weights on microsecond dwell boundaries creates processing bottlenecks unsuitable for the target latency budget.
- **Theoretical redundancy:** RMAB theory already provides a mathematically proven, near-optimal index policy without black-box convergence risk.
- **Status:** retained solely as an **offline comparative benchmark** (`eval/run_comparison.py`) to empirically demonstrate why the closed-form Whittle Index is preferable for real-time electronic combat use. Zero DQN logic exists in the live decision loop.

### 13.2 UCB1 and Thompson Sampling — Removed from Runtime

- Classical bandit algorithms assume stationary or frozen unpulled arms — a premise the restless RF spectrum violates by definition.
- **Status:** retained as baseline comparison scripts (`baselines/bandit_ucb.py`) to quantify the performance delta between stationary heuristics and the true restless-bandit policy. The deployed scheduler contains zero UCB or Thompson Sampling logic.

---

## 14. Rigorous, Testable Performance Targets

Qualitative claims are avoided in favor of falsifiable, measurable figures of merit:

| Metric | Target | Definition |
|---|---|---|
| Decision compute latency | < 15 µs per decision slot (target, pending empirical validation) | Measured wall-clock latency per call to `scheduler.select_action()` |
| Average intercept time error | <= 1.0 dwell period against periodic targets (design goal) | `(1/K) * sum \|t_pred - t_actual\|` over K predictions |
| Probability of Detection (Pd) | Reported across simulated SNR range [-5 dB, +15 dB] | True intercept events / total dwells on truly-active channels |
| Probability of False Alarm (Pfa) | Bounded at <= 10^-3 (Neyman-Pearson threshold) | False detections / total dwells on truly-silent channels |
| Interception Ratio (POI) | Target >= 85% on periodic and agile emitters | Detected transmission windows / total transmission windows |

---

## 15. Risk Assessment & Mitigation (Feasibility Slide Content)

| Risk | Mitigation |
|---|---|
| Numba fails to compile structured NumPy dtype in `nopython` mode | Pre-verified with isolated test snippet (Section 10.1) before scheduler build begins; flat parallel-array fallback ready if needed |
| Online transition estimates violate Whittle indexability precondition (P11 < P01) | Explicit clamp enforced immediately after every MAP update (Section 5.5) |
| Static LUT goes stale under continuous online learning | Redesigned as per-channel, periodically recomputed LUT (Section 10.2), not a single fixed offline tensor |
| Two periodic reservations collide on the same time slot | Deterministic tie-break by peak spectral power confidence, loser logged as accepted known-miss for honest metrics (Section 4) |
| Reservation queue sized to unknown real-mission length | Explicitly scoped as simulation-only; production version uses bounded ring buffer or sparse dict (Section 10.3) |
| 15 µs latency budget not actually achievable in practice | Numba JIT compilation targeted specifically for this; target explicitly labeled as design goal pending empirical benchmarking, not yet claimed as proven (Section 11) |
| Numba/NumPy version incompatibility | Verified via clean `pip install` resolution before locking version numbers into final documentation (Section 12) |
| Whittle closed-form formula transcription error | To be checked term-by-term against the original Liu & Zhao (2010) paper before final documentation lock (Section 5.2) |

---

## 16. Build Order / Implementation Checklist

Work through in this order — each step is a safety net if later steps run out of time:

1. [ ] Run Numba structured-dtype smoke test (Section 10.1). Confirm array design before writing anything else.
2. [ ] Verify NumPy/Numba version compatibility via clean pip install (Section 12).
3. [ ] Implement flat/structured per-channel state array and cold-start bootstrap (Sections 7, 10.1).
4. [ ] Implement MAP transition estimator with Laplace smoothing, Path B (τ=1-only) counting (Section 6.1).
5. [ ] Implement indexability clamp (Section 5.5).
6. [ ] Implement belief update formula (Section 5.3).
7. [ ] Verify Whittle Index closed-form formula against Liu & Zhao (2010) source paper directly (Section 5.2).
8. [ ] Implement per-channel dynamic LUT with recompute trigger every K=10 observations (Section 10.2).
9. [ ] Implement core Whittle policy: `argmax` over LUT retrieval across all N channels.
10. [ ] Implement Lomb-Scargle periodicity estimator, triggered async on new hits (Section 8).
11. [ ] Implement reservation queue with collision tie-break logic (Section 4).
12. [ ] Implement arbitration veto layer combining periodicity override + Whittle fallthrough (Section 4).
13. [ ] Wrap hot-path functions in `@njit(nopython=True)`, confirm no silent object-mode fallback.
14. [ ] Profile actual latency on target hardware; update Section 11's status from "target" to "measured."
15. [ ] Build offline DQN/UCB/Thompson comparison scripts for benchmarking only (Section 13) — lowest priority, only if time remains.
16. [ ] Run PyTest/Hypothesis property-based tests on Markov estimation convergence.
17. [ ] Wire up Streamlit/Plotly live waterfall dashboard for demo.

---

## 17. Glossary (Scheduler-Specific)

| Term | Meaning |
|---|---|
| Dwell | One discrete time slot where the receiver listens to a chosen channel |
| τ_n (tau) | Dwell age — time steps elapsed since channel n was last checked |
| s_last(n) | Last observed binary state (0 or 1) for channel n |
| Belief b_t(n) | Current estimated probability that channel n is active, given elapsed time and history |
| Whittle Index | Per-channel urgency score; scheduler always picks the channel with the highest index |
| Arbitration Veto Layer | The stage that decides whether a periodicity override or the core Whittle policy gets to choose the next action |
| MAP estimation | Maximum A Posteriori — the method used to estimate transition probabilities online, with Laplace smoothing to avoid zero-frequency issues early on |
| Indexability | The mathematical property (requiring P11 >= P01) that guarantees the Whittle Index policy is well-defined and near-optimal |
| LUT | Lookup Table — precomputed/cached Whittle Index values indexed by channel, dwell age, and last state, for O(1) retrieval |
| Reservation | A scheduled forced dwell on a specific channel at a specific future time slot, created by the Periodicity Estimator |
| Critical path | The sequence of computation stages that must complete before the next tuning command can be dispatched, bounded by the 15 µs latency budget |

---

*End of document. This file, together with the RF Environment Simulator build document, contains everything needed to implement the scheduler from scratch: the architecture, the math (with an explicit, stated simplification for transition estimation and a flagged item to verify against source literature), the data structures (with a pre-build verification step to de-risk the Numba dependency), the tech stack, every known open risk with its mitigation, and a concrete, ordered build checklist.*
