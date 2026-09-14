# RF Environment Simulator — Full Build Document
### Smart India Hackathon PS 26055 — Smart Scan Strategy for Electronic Warfare

---

## 1. What We Are Building

The RF Environment Simulator is **not** a physics-level electromagnetic waveform simulator. It does not generate raw I/Q samples at gigasamples-per-second, and it does not model continuous carrier waves. That approach is computationally prohibitive and would stall any ML training loop.

Instead, the simulator is an **abstracted spatiotemporal occupancy and power oracle**: a discrete-event software sandbox that maintains a hidden, uncorrupted ground-truth map of where hostile electromagnetic energy exists across frequency and time, while enforcing realistic hardware constraints on a receiver trying to intercept it.

Think of it as a game engine. The world (emitters) runs autonomously and is completely blind to what the receiver (the RL agent) chooses to look at. The receiver only ever gets a noisy, partial window into that world.

### 1.1 Why this design, and what it costs

This is the correct engineering trade-off for training an RL scheduler fast — but it is also a simplification, and the team should be able to state that simplification out loud:

- Emitters in this design are **open-loop** — they do not react to the receiver's behavior. Real adversarial EW involves emitters that adapt when they sense they're being intercepted. This simulator does not model that. This should be stated explicitly as a scoped limitation, not hidden.
- The core research value is in the **ground-truth fidelity, the noise model, and the hardware constraints** — not in the RL algorithm itself, which can be off-the-shelf (UCB, Thompson Sampling, Whittle Index). Judges who understand EW will look for exactly this fidelity.

---

## 2. The Three-Layer Architecture

```
+------------------------------------------------------------------------------------+
|                         RF ENVIRONMENT SIMULATOR (THE ORACLE)                      |
|                                                                                    |
|  [Layer 1: Emitter World Model]                                                    |
|  • Periodic Radar: emits pulses at deterministic intervals (PRI)                   |
|  • Agile Jammer/Comms: hops carrier bands via random/Markov rules                  |
|  • Rotating Search Radar: sinc²-shaped beam sweeping across azimuth                |
|                                         │                                          |
|                                         ▼ Generates True Spectrum State            |
|  [Layer 2: The Hidden Ground Truth Matrix]                                         |
|  • Matrix S[N][T] ∈ {0, 1} (or continuous power P[N][T] in dBm)                    |
|  • Completely hidden from the ML agent; logs everything happening in the ether    |
|                                         │                                          |
|                                         ▼ Filtered by Receiver Action (a_t)        |
|  [Layer 3: Hardware Bottlenecks & Sensor Channel]                                  |
|  • Retuning Latency: receiver blind for τ_switch if changing channels              |
|  • RF Detection Noise: Pd / Pfa conditional Bernoulli decisions                    |
|                                         │                                          |
+-----------------------------------------|------------------------------------------+
                                          ▼ Returns Noisy Observation (O_t)
                              [ML Scheduler / Agent]
```

### 2.1 Layer 1 — Emitter World Model

Independent generator objects, each advancing its own internal state every timestep, indifferent to what the receiver is doing:

| Emitter type | Behavior |
|---|---|
| **Periodic Radar** | Fires pulses at a fixed Pulse Repetition Interval (PRI) — on for `Y` steps, off for `Z` steps, repeating |
| **Agile Jammer / Frequency Hopper** | Jumps between channels according to a random or Markov-style hopping rule |
| **Rotating Search Radar** | Antenna beam sweeps across azimuth; power at any given band follows a sinc² envelope centered on the beam's current pointing angle |

### 2.2 Layer 2 — The Hidden Ground-Truth Tensor (S)

A discrete `N × T` matrix:
- `N = ⌈B_sys / B_inst⌉` — number of channels, derived from system bandwidth divided by receiver's instantaneous bandwidth
- `T` — number of discrete dwell-time steps in the episode
- `S(n, t) ∈ {0, 1}` — answers exactly one question: *is there a transmission in band `n` at time `t`?*

This matrix is written to by Layer 1 every timestep and is **never exposed directly to the agent**. It exists purely so the simulator (not the agent) can compute ground-truth performance metrics.

### 2.3 Layer 3 — Sensor Observation Channel (Receiver Model)

Takes the agent's chosen action `a_t ∈ {1, ..., N}` and:

1. **Checks retuning state.** If `a_t ≠ a_{t-1}`, the receiver is blind for this dwell (local-oscillator retuning penalty, `τ_switch = t_tune`). Return `O_t = 0` regardless of truth.
2. **If not blind**, look up the true value `S(a_t, t)`:
   - If `S(a_t, t) == 1`: draw `O_t ~ Bernoulli(P_d(SNR))`
   - If `S(a_t, t) == 0`: draw `O_t ~ Bernoulli(P_fa)`
3. Returns noisy observation `O_t ∈ {0, 1}` to the agent — never the raw truth.

---

## 3. Operational Flow (Gymnasium API)

```
                     +---------------------------------------+
                     |                reset()                |
                     |  • Clear spectrum grid S[N][T]        |
                     |  • Instantiate & randomize emitters   |
                     |  • Reset clock t = 0                  |
                     +-------------------+-------------------+
                                         │
                                         ▼
                      Action Selected by Agent: a_t (Band index)
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
|                                     step(a_t)                                   |
|                                                                                 |
| 1. Hardware State Evaluation:                                                   |
|    • Check if a_t == a_{t-1}.                                                   |
|    • If not, apply tuning penalty τ_switch = t_tune. Receiver BLIND (O_t = 0). |
|                                                                                 |
| 2. Advance Emitter States:                                                      |
|    • Every emitter updates its own internal clock (PRI counter, beam angle,     |
|      hop transition) independent of the receiver.                              |
|    • Write true presence (1 or 0) into Ground Truth Matrix S[·][t].            |
|                                                                                 |
| 3. Sensor Observation Channel:                                                  |
|    • If receiver NOT blind:                                                    |
|      - S[a_t][t] == 1 → draw O_t ~ Bernoulli(P_d(SNR))                         |
|      - S[a_t][t] == 0 → draw O_t ~ Bernoulli(P_fa)                             |
|                                                                                 |
| 4. Metric Logging (Truth Oracle):                                               |
|    • Compare O_t against S[a_t][t].                                            |
|    • Log True Positives, False Alarms, Missed Bursts, Latency Errors.         |
|    • Compute Reward R_t.                                                        |
|                                                                                 |
| 5. Clock Advance: t ← t + 1                                                     |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
                 Returns: (Observation O_t, Reward R_t, Terminated, Info)
```

### 3.1 Why the ground truth matters — The Verification Oracle

Because the simulator retains the uncorrupted matrix `S`, it can compute metrics that a **real physical receiver could never compute during an operational mission**:

- **Time-to-Intercept (TTI) error**: `Δt = t_detected − t_started`
- **Probability of Intercept (POI)**: fraction of an emitter's total active time that was actually observed

These two numbers are the headline metrics — everything else (Pd, Pfa, reward curve, correct-prediction %) exists to support them.

---

## 4. Open-Source Reuse Map — Verified, Not Assumed

Every repo below was actually cloned/searched and checked before being listed — this is important, because an earlier draft of this research included repo names that turned out not to exist. Only verified, real repositories are listed as reusable.

### 4.1 `vtnsi/rfrl-gym` (GitHub — Virginia Tech National Security Institute)

Real, active repository, built specifically for RF spectrum sensing / dynamic spectrum access / jamming RL research. Subclasses OpenAI/Gymnasium. This is the closest existing analog to our exact problem and is the primary reuse source.

**What we take, file by file:**

| File in `rfrl-gym` | What it contains | How we use it |
|---|---|---|
| `rfrl_gym/entities/entity.py` | Base `Entity` class with a built-in `onoff=[X,Y,Z]` cycle timer (`__cycle_onoff`) — on for `Y` steps, off for `Z` steps | **Reuse almost as-is.** This on/off cycle *is* a periodic-pulse (PRI) mechanism already. Forms the base class for all our emitters. |
| `rfrl_gym/entities/constant_freq.py` | Fixed-channel entity built on the onoff cycle | **Reuse, rename to `PeriodicRadar`.** This is functionally our periodic radar — a channel that goes on/off at fixed intervals. |
| `rfrl_gym/entities/stochastic_hop_freq.py` | Random channel selection each active tick, weighted by a probability vector | **Reuse as our Agile Jammer.** A working weighted-random hop; can be upgraded later to a full Markov transition matrix if time permits. |
| `rfrl_gym/entities/agile_freq.py` | Smarter hop logic that avoids channels already occupied by other entities | **Optional bonus**, use if we want an agile emitter that reacts to spectrum congestion (not to the receiver — still open-loop with respect to the agent). |
| `rfrl_gym/envs/rfrl_gym_abstract_env.py` | Full Gym `step()`/`reset()` skeleton, `true_history` ground-truth matrix, `action_history`, JSON-driven scenario loading | **Reuse the skeleton wholesale.** This is effectively Layer 2 + the Gym harness already built. Strip the PyQt rendering dependency — not needed for our purposes. |

**What is confirmed missing from this repo — must be written fresh:**

- No **Scanning/Rotating Radar** entity (sinc² beam sweep) exists anywhere in the codebase.
- No **Pd/Pfa Bernoulli noise model** exists. The only detector present (`rfrl_gym/detectors/energy_detector.py`) is a flat energy threshold (`>0.001 → 1`), not a probabilistic detection model.
- No **retuning/blind latency penalty** exists — action switching is free in this repo.

### 4.2 Pieces we write ourselves (deliberately, not because nothing exists — because what exists is either wrong-shaped or overkill)

| Piece | Why not reused | What we do instead |
|---|---|---|
| **Scanning Radar (sinc² beam)** | Only real prior art found (`WeatherGod/ScanRadSim`) is a 2010-era, Python 2.6, meteorology-domain tool with obscure unmaintained dependencies. Not worth integrating for what is fundamentally a ~10-line function. | Write `gain(θ) = (sin(x)/x)²` directly, where `x` is proportional to the angular offset between beam center and target band's mapped azimuth. |
| **Pd/Pfa detection model** | Only real prior art found (`radarsimx/radarsimpy`) is a full waveform-level radar simulator distributed as prebuilt binaries under a tiered/paid license — exactly the heavyweight physics simulation this project deliberately avoids. | Implement **Albersheim's equation** directly (closed-form relationship between required SNR, Pd, and Pfa) — a few lines of math, no dependency, textbook-standard and defensible if a judge asks "where does this formula come from." |
| **Retuning/blind latency logic** | Not present in any reviewed repo. | A few lines inside our own `step()`: compare `a_t` to `a_{t-1}`, force `O_t = 0` for `τ_switch` steps on a channel change. |

---

## 5. Repository / Directory Structure

```
smart-scan-ew/
├── env/
│   ├── emitters.py            # Base Entity (from rfrl-gym) + PeriodicRadar, AgileJammer
│   │                           # (reused/adapted from rfrl-gym) + ScanningRadar (written fresh)
│   ├── spectrum_env.py        # Gymnasium core: step(), reset(), true_history ground truth
│   │                           # (adapted from rfrl_gym_abstract_env.py, PyQt stripped out)
│   └── receiver_model.py      # Retuning latency logic + Albersheim Pd/Pfa noise model (fresh)
│
├── oracle/
│   ├── truth_tracker.py       # S[N][T] ground truth recorder and active-burst auditor
│   └── metrics_engine.py      # Computes POI, False Alarm rate, TTI error, avg intercept rate
│
├── baselines/
│   ├── round_robin.py         # Fixed sweep baseline — mandatory comparison point
│   └── weighted_priority.py   # Open-loop, priority-weighted baseline
│
├── schedulers/
│   ├── bandit_ucb.py
│   ├── bandit_thompson.py
│   ├── whittle_rmab.py        # Differentiator — most competing teams will skip this
│   └── dqn_agent.py           # Stretch goal only, Stable-Baselines3 wrapper
│
├── estimators/
│   └── periodicity_estimator.py   # Lomb-Scargle / autocorrelation — predicts next pulse
│                                    # arrival for a periodic/scanning emitter
│
├── eval/
│   └── run_comparison.py      # Runs every strategy on identical scenario seeds,
│                                # outputs a single comparison table — this table is
│                                # the single most convincing artifact for judges
│
├── dashboard/
│   └── app.py                  # Streamlit live spectrum waterfall + running metrics
│
└── tests/
    └── verify_sim.py           # Validates synthetic PRI / rotation periods against
                                  # the analytic math, before any RL touches the sim
```

---

## 6. Build Order (Do Not Reorder)

1. **Ground truth + metrics engine.** No shortcuts here — this is the entire evidentiary basis for every later claim. Build `emitters.py`, `spectrum_env.py`, `truth_tracker.py`.
2. **Open-loop baselines.** Round-robin, then weighted-priority. Run both, save the numbers before writing a single line of RL.
3. **Bandit schedulers.** UCB and Thompson Sampling — cheap, fast, reliably working same day.
4. **Whittle Index / Restless Bandit.** This is the differentiator most competing teams will skip — a working, explainable Whittle index scheduler is worth more to judges than an under-trained neural net.
5. **Periodicity estimator.** Lomb-Scargle or autocorrelation-based predictor for scanning/periodic emitters. Enables the "intercept time error" metric that most teams neglect entirely.
6. **Deep RL (DQN) — stretch goal only.** Only attempt if time remains after steps 1–5 are solid. An under-trained DQN wobbling live in front of judges is worse than not having one.

---

## 7. What Actually Wins — Presentation Layer

The simulator is not "just a delivery mechanism" for the RL model — it **is** the evidence. A model without an honest simulator and baseline comparison is a number, not a result. Judges specifically watch for:

1. **A single side-by-side comparison table** — round-robin vs weighted-priority vs UCB vs Thompson vs Whittle vs (DQN if present), same scenario seeds, all three headline metrics (POI, TTI, false-alarm rate). This one artifact carries more weight than algorithm sophistication.
2. **Held-out generalization test** — evaluate on a scenario seed/emitter mix different from training. State this explicitly out loud in the demo. Most teams skip it; it is a direct, low-cost differentiator.
3. **Explainability** — be ready to say precisely why Whittle Index is near-optimal for a restless bandit (cite Zhao & Krishnamachari), and why a closed-form periodic-interception model alone isn't sufficient (it needs known emitter parameters; the online estimator does not).
4. **Live spectrum waterfall dashboard** (Streamlit) — watching the agent visibly chase a hopping emitter in real time is what gets remembered after the reward-curve chart is forgotten.
5. **Stated limitations, up front.** Explicitly say the emitters are open-loop/non-adversarial and that the receiver model uses single-channel, Bernoulli-approximated Pd/Pfa. Stating a limitation as a deliberate scope decision reads as competence; having a judge discover it unprompted reads as an oversight.

---

## 8. Formal Notation Reference

- Spectrum divided into `N` bands, `b = 1..N`
- Time discretized into slots `t = 1..T`, each slot = one dwell period `Δt = t_dwell`
- Ground truth: `S[b][t] ∈ {0,1}`
- Action: `a_t ∈ {1..N}` — band chosen at time `t` (single-channel receiver; extend to k-channel as a stretch goal)
- Observation: `O_t = S[a_t][t]` passed through the noise/blind model in Layer 3
- Reward: `+1` for first-time/high-value intercept, smaller reward for repeat intercept of an already-characterized emitter, `0`/slightly negative for an empty look, heavier weight for new/threat-classified emitters

### 8.1 Why this is a Restless Multi-Armed Bandit, not a plain bandit

In a classical bandit, arms don't change unless pulled. Here, **every band's state evolves whether or not the receiver is watching it** — an emitter can turn on/off in a band that isn't currently being observed. This is precisely the structure of a Restless Multi-Armed Bandit (RMAB), well studied in dynamic spectrum access literature since ~2007–2010 (Zhao & Krishnamachari; Liu & Zhao, on Whittle index optimality for multichannel access).

---

## 9. Common Failure Modes to Avoid

1. **No baseline comparison** — the single most common judging failure. An RL model demoed alone, with no round-robin run on identical scenarios, proves nothing.
2. **Overfitting to one scenario/seed** — always evaluate on a held-out scenario set and say so explicitly.
3. **Scope creep into signal classification** — this PS is about scan scheduling to intercept, not modulation classification/fingerprinting. Do not build a classifier; it dilutes time on a different, deep problem.
4. **Skipping the "intercept time error" / periodicity metrics** because they require a genuine prediction module — this is exactly where most competing teams will not bother, making it a strong, low-effort differentiator.
5. **Pure deep-RL-or-bust** — an under-trained DQN in a live demo looks worse than a clean, explainable bandit or Whittle-index scheduler. Ship the simple thing that works; add complexity only if time remains.

---

## 10. One-Paragraph Summary (say this out loud in 30 seconds)

"Open-loop EW scanning sweeps the spectrum blind, on a fixed schedule set before the mission, so it wastes time revisiting known-quiet bands and gives no special attention to new or agile threats. We model this as a restless multi-armed bandit / partially observable scheduling problem: each frequency band's state changes whether or not we're watching it, so the receiver has to balance exploring for new emitters against exploiting bands known to be active. We built a simulated RF environment with ground-truth emitter behavior — periodic radar, rotating-beam radar, and frequency-agile jamming — layered with a realistic detection-noise and retuning-latency model, and compared round-robin and priority-weighted open-loop baselines against bandit-based and Whittle-index restless-bandit schedulers, plus a periodicity estimator that predicts when a scanning emitter will next be visible. Across probability of intercept, average intercept time, and interception ratio, our closed-loop scheduler beats open-loop scanning, with the largest gains against periodic and agile emitters — exactly the cases open-loop handles worst."

---

*End of build document. This file alone contains everything needed to construct the simulator: architecture, data flow, verified repo reuse map with exact files, directory structure, build order, and the presentation strategy for judging.*
