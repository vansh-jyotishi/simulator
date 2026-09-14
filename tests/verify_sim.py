"""Black-box analytic verification suite (S19).

Written from INTERFACE.md against ``env.spectrum_env.make_env`` with scripted
schedulers and in-memory dict scenarios.  Each case prints ``configured vs
recovered`` under ``pytest -s``.  A disagreement here is a contract question:
resolve it in INTERFACE.md first, then in code.

    python -m pytest tests/verify_sim.py -q -s
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from baselines.round_robin import RoundRobin
from baselines.weighted_priority import WeightedPriority
from common.protocol import sanitize_info
from common.types import INFO_KEYS
from env.receiver_model import PassThroughReceiver, albersheim_pd
from env.spectrum_env import make_env
from eval.run_comparison import parse_seeds
from oracle.metrics_engine import compute_metrics
from oracle.truth_tracker import segment_bursts
from tests.fakes import Scripted, StayPut, SwitchEvery, play


def _say(label, configured, recovered):
    print(f"\n  [{label}] configured: {configured}  |  recovered: {recovered}")


def _runs(row):
    pad = np.concatenate(([0], np.asarray(row).astype(np.int8), [0]))
    d = np.diff(pad)
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]  # starts, ends (exclusive)


def _base(N=8, T=3000, tau=1, pfa=1e-3, noise="albersheim", **extra):
    d = {"name": "verify", "N": N, "T": T, "dwell_s": 1e-3,
         "receiver": {"tau_switch": tau, "pfa": pfa, "noise_model": noise, "initial_channel": 0},
         "emitters": []}
    d.update(extra)
    return d


# 1 ---------------------------------------------------------------- PRI recovered
def test_1_pri_recovered():
    start, on, off = 5, 4, 13
    sc = _base(emitters=[{"type": "periodic", "name": "r", "channel": 3, "onoff": [start, on, off], "snr_db": 10.0}])
    env = play(sc, 0, StayPut(3), receiver=PassThroughReceiver())
    obs = env.get_history().obs.astype(float)
    c = obs - obs.mean()
    ac = np.correlate(c, c, "full")[len(obs) - 1:]
    lag = int(np.argmax(ac[1:100]) + 1)
    starts, ends = _runs(obs)
    runs = set((ends - starts)[:-1].tolist())
    gaps = set((starts[1:] - ends[:-1]).tolist())
    n_bursts = len(starts)
    expected = (3000 - start) // (on + off)
    _say("PRI", f"PRI={on + off} on={on} off={off} bursts~{expected}", f"lag={lag} runs={runs} gaps={gaps} bursts={n_bursts}")
    assert lag == on + off and runs == {on} and gaps == {off}
    assert abs(n_bursts - expected) <= 1


# 2 ---------------------------------------------------------------- rotation recovered
def test_2_rotation_recovered():
    T_rot, bw = 130, 20.0
    sc = _base(T=5000, emitters=[{"type": "scanning", "name": "s", "mode": "lighthouse", "channel": 5, "T_rot": T_rot,
                                  "beamwidth_deg": bw, "phase0_deg": 180.0, "snr_peak_db": 15.0,
                                  "presence_gain_db": -10.0}])
    env = play(sc, 0, StayPut(5), receiver=PassThroughReceiver())
    obs = env.get_history().obs
    starts, ends = _runs(obs)
    spacing = set(np.diff(starts).tolist())
    width = round(T_rot * 1.667 * bw / 360)
    if ends[-1] == len(obs):  # last burst truncated by the episode end
        starts, ends = starts[:-1], ends[:-1]
    lengths = set((ends - starts).tolist())
    _say("rotation", f"T_rot={T_rot} width~{width}", f"spacing={spacing} lengths={lengths}")
    assert spacing == {T_rot}
    assert all(abs(L - width) <= 1 for L in lengths)


# 3 ---------------------------------------------------------------- hop recovered
def test_3_hop_recovered():
    hp, p_stay = 4, 0.5
    probs = [0.4, 0.3, 0.2, 0.1]
    sc = _base(T=20000, emitters=[{"type": "agile", "name": "j", "channels": [1, 2, 3, 4], "hop_probs": probs,
                                   "hop_period": hp, "p_stay": p_stay, "onoff": [0, 19998, 1], "snr_db": 9.0}])
    env = make_env(sc, 0)
    env.reset()
    S = env.get_truth().S
    L = 19996  # analyse the always-on prefix (a whole number of hops)
    col = S.sum(axis=0)
    assert set(col[:L].tolist()) == {1}, "exactly one channel per active slot"
    chan = S.argmax(axis=0)[:L]
    changes = np.nonzero(np.diff(chan))[0] + 1
    hops = chan[::hp]
    stay = float(np.mean(hops[1:] == hops[:-1]))
    occ = np.bincount(chan, minlength=8)[1:5] / len(chan)
    # stationary occupancy of the exclude-current chain (INTERFACE C4)
    P = np.zeros((4, 4))
    pr = np.asarray(probs)
    for c in range(4):
        o = pr.copy(); o[c] = 0
        P[c] = (1 - p_stay) * o / o.sum(); P[c, c] = p_stay
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
    _say("hop", f"hop_period={hp} p_stay={p_stay} stationary={pi.round(3).tolist()}",
         f"changes%hp=0:{bool(np.all(changes % hp == 0))} stay={stay:.3f} occupancy={occ.round(3).tolist()}")
    assert np.all(changes % hp == 0)
    assert abs(stay - p_stay) < 0.03
    assert np.all(np.abs(occ - pi) < 0.03)
    # literal plan clause "occupancy vs hop_probs +- 0.03": holds exactly when hop_probs is uniform (INTERFACE C4)
    scu = _base(T=20000, emitters=[{"type": "agile", "name": "j", "channels": [1, 2, 3, 4], "hop_probs": None,
                                    "hop_period": hp, "p_stay": 0.0, "onoff": [0, 19998, 1], "snr_db": 9.0}])
    envu = make_env(scu, 0)
    envu.reset()
    chan_u = envu.get_truth().S.argmax(axis=0)[:L]
    occ_u = np.bincount(chan_u, minlength=8)[1:5] / len(chan_u)
    _say("hop uniform", "occupancy == hop_probs = 0.25 each (+-0.03)", occ_u.round(3).tolist())
    assert np.all(np.abs(occ_u - 0.25) < 0.03)


# 4 ---------------------------------------------------------------- overlap
def test_4_overlap():
    sc = _base(T=1200, emitters=[
        {"type": "periodic", "name": "r", "channel": 3, "onoff": [0, 4, 6], "snr_db": 12.0},
        {"type": "agile", "name": "j", "channels": [2, 3, 4], "hop_period": 3, "p_stay": 0.3, "onoff": [0, 1198, 1],
         "snr_db": 6.0}])
    env = make_env(sc, 0)
    env.reset()
    tr = env.get_truth()
    S_by = tr.S_by_emitter
    both = (S_by[0] == 1) & (S_by[1] == 1)
    bursts = segment_bursts(tr)
    n0 = sum(b.emitter == 0 for b in bursts); n1 = sum(b.emitter == 1 for b in bursts)
    _say("overlap", "S=OR SNR=max E=argmax", f"overlap cells={int(both.sum())} bursts e0={n0} e1={n1}")
    assert both.sum() > 0
    assert np.array_equal(tr.S, S_by.any(axis=0).astype(np.int8))
    assert np.all(tr.SNR[both] == 12.0) and np.all(tr.E[both] == 0)
    assert np.all(tr.E[(S_by[1] == 1) & ~both] == 1)
    assert n0 > 0 and n1 > 0


# 5 ---------------------------------------------------------------- noise rates
def test_5_noise_rates():
    T = 40000
    sc = _base(T=T, tau=0, emitters=[{"type": "periodic", "name": "r", "channel": 1, "onoff": [0, T - 2, 1],
                                      "snr_db": 10.0}])
    env = play(sc, 0, StayPut(1))
    m = compute_metrics(env.get_truth(), env.get_history())
    pd_ref = albersheim_pd(10.0, 1e-3)
    sig = math.sqrt(pd_ref * (1 - pd_ref) / (T - 1))
    _say("Pd", f"albersheim_pd(10 dB, 1e-3)={pd_ref:.4f}", f"{m.sensor_pd:.4f} (3 sigma = {3 * sig:.4f})")
    assert abs(m.sensor_pd - pd_ref) < 3 * sig
    env2 = play(sc, 0, StayPut(4))
    m2 = compute_metrics(env2.get_truth(), env2.get_history())
    sig2 = math.sqrt(1e-3 * (1 - 1e-3) / T)
    _say("Pfa", "1e-3", f"{m2.sensor_pfa:.5f} (3 sigma = {3 * sig2:.5f})")
    assert abs(m2.sensor_pfa - 1e-3) < 3 * sig2
    scf = dict(sc); scf["receiver"] = {"tau_switch": 0, "pfa": 0.05, "noise_model": "fixed", "pd_fixed": 0.85}
    e3 = play(scf, 0, StayPut(1)); e4 = play(scf, 0, StayPut(4))
    m3 = compute_metrics(e3.get_truth(), e3.get_history()); m4 = compute_metrics(e4.get_truth(), e4.get_history())
    _say("fixed", "0.85 / 0.05", f"{m3.sensor_pd:.4f} / {m4.sensor_pfa:.4f}")
    assert abs(m3.sensor_pd - 0.85) < 3 * math.sqrt(0.85 * 0.15 / T)
    assert abs(m4.sensor_pfa - 0.05) < 3 * math.sqrt(0.05 * 0.95 / T)


# 6 ---------------------------------------------------------------- blindness
@pytest.mark.parametrize("tau", [0, 1, 2, 3])
def test_6_blindness(tau):
    k = 6
    sc = _base(T=1200, tau=tau, emitters=[{"type": "periodic", "name": "r", "channel": 1, "onoff": [0, 1198, 1],
                                          "snr_db": 30.0}])
    env = make_env(sc, 0)
    obs, info = env.reset()
    sched = SwitchEvery(k, (1, 2))
    sched.reset(env.mission_spec, 0)
    infos = []
    while True:
        a = sched.select_action(obs, sanitize_info(info))
        obs, r, term, trunc, info = env.step(a)
        infos.append(info)
        if trunc:
            break
    h = env.get_history()
    n_sw, n_bl = int(h.switched.sum()), int(h.blind.sum())
    seq = [i["blind_remaining"] for i in infos[k:k + tau + 1]]
    m = compute_metrics(env.get_truth(), h)
    _say(f"blind tau={tau}", f"n_blind == n_switches*tau", f"switches={n_sw} blind={n_bl} seq_after_switch={seq}")
    assert n_bl == n_sw * tau
    assert np.all(h.obs[h.blind] == 0)
    assert seq == list(range(tau - 1, -1, -1)) + [0]
    assert m.n_blind == n_bl
    # no tp while blind: every blind dwell on channel 1 is occupied yet obs==0, so effective < sensor for tau>0
    if tau > 0:
        assert m.effective_pd < m.sensor_pd


# 7 ---------------------------------------------------------------- CRN / determinism
def test_7_common_random_numbers_and_determinism():
    sc = json.load(open("scenarios/train_default.json")); sc["T"] = 1500
    e1 = play(sc, 0, StayPut(7)); e2 = play(sc, 0, SwitchEvery(3, (7, 20, 41)))
    assert np.array_equal(e1.get_truth().S, e2.get_truth().S) and np.array_equal(e1.get_truth().U, e2.get_truth().U)
    e3 = play(sc, 0, StayPut(7))
    for f in ("actions", "obs", "blind", "switched", "new_detect", "rewards"):
        assert np.array_equal(getattr(e1.get_history(), f), getattr(e3.get_history(), f))
    e4 = play(sc, 1, StayPut(7))
    assert not np.array_equal(e1.get_truth().U, e4.get_truth().U)
    assert not np.array_equal(e1.get_truth().S, e4.get_truth().S)
    train, held = parse_seeds("0-99"), parse_seeds("100-199")
    _say("CRN", "same seed -> same S,U; seeds 0-99 vs 100-199 disjoint", f"disjoint={not set(train) & set(held)}")
    assert not (set(train) & set(held)) and len(train) == 100 and len(held) == 100


# 8 ---------------------------------------------------------------- no truth leak
def test_8_no_truth_leak():
    env = make_env("scenarios/mvp_periodic.json", 0)
    obs, info = env.reset()
    assert set(info) == set(INFO_KEYS)
    for a in (0, 7, 7, 33):
        obs, r, term, trunc, info = env.step(a)
        assert set(info) == set(INFO_KEYS)
        assert type(obs) is int and obs in (0, 1)
    leaked = dict(info); leaked["S"] = env.get_truth().S
    clean = sanitize_info(leaked)
    _say("leak", f"INFO_KEYS={list(INFO_KEYS)}", f"sanitized keys={sorted(clean)}")
    assert "S" not in clean and set(clean) == set(INFO_KEYS)
    check_env(make_env("scenarios/mvp_periodic.json", 0), skip_render_check=True)


# 9 ---------------------------------------------------------------- hand example through the real env
def test_9_hand_example():
    # radar on channel 1: onoff [2,3,7] -> bursts 2-4, 12-14, 22-24, 32-34 (PRI 10), T=40, N=4, tau=1, Pd=1, Pfa~0
    sc = _base(N=4, T=40, tau=1, pfa=1e-12, noise="fixed",
               emitters=[{"type": "periodic", "name": "r", "channel": 1, "onoff": [2, 3, 7], "snr_db": 10.0}])
    sc["receiver"]["pd_fixed"] = 1.0
    actions = [0, 0, 0] + [1] * 13 + [2] * 6 + [1] * 18  # switch to 1 at t=3, away at 16, back at 22
    env = play(sc, 0, Scripted(actions))
    m = compute_metrics(env.get_truth(), env.get_history())
    # hand: A caught t=4 (tti 2), B t=12 (0), C t=23 (1), D t=32 (0) -> POI 1, mean 0.75, median 0.5
    # occupied dwells: t3(blind) 4 | 12 13 14 | 22(blind) 23 24 | 32 33 34 -> tp 9, blind-occupied 2
    _say("hand", "POI=1 tti mean=0.75 med=0.5 Pd=1 eff=9/11 Pfa=0 switches=3 blind=3",
         f"POI={m.poi} tti={m.mean_tti}/{m.median_tti} Pd={m.sensor_pd} eff={m.effective_pd:.4f} Pfa={m.sensor_pfa} "
         f"switches={m.n_switches} blind={m.n_blind}")
    assert m.poi == 1.0 and m.n_bursts == 4
    assert m.mean_tti == 0.75 and m.median_tti == 0.5
    assert m.sensor_pd == 1.0 and abs(m.effective_pd - 9 / 11) < 1e-12 and m.sensor_pfa == 0.0
    assert m.n_switches == 3 and m.n_blind == 3 and m.n_tp == 9 and m.n_fn == 0


# 10 --------------------------------------------------------------- low-SNR curve and blind zone
def test_10_snr_curve_and_blind_zone():
    pds = []
    for x in (-5, 0, 5, 10, 15):
        env = play(f"scenarios/snr_sweep_{x}.json", 0, StayPut(1))
        pds.append(compute_metrics(env.get_truth(), env.get_history()).sensor_pd)
    _say("snr sweep", "Pd(-5)=.03 Pd(0)=.10 Pd(5)=.21 Pd(10)=.81 Pd(15)>.999 monotone", [round(p, 4) for p in pds])
    assert all(b >= a for a, b in zip(pds, pds[1:]))
    assert abs(pds[0] - 0.03) < 0.02 and abs(pds[1] - 0.10) < 0.02 and abs(pds[2] - 0.21) < 0.02
    assert abs(pds[3] - 0.81) < 0.02 and pds[4] > 0.999
    bz = json.load(open("scenarios/blind_zone.json"))
    er = play(bz, 0, RoundRobin()); ew = play(bz, 0, WeightedPriority())
    mr = compute_metrics(er.get_truth(), er.get_history()); mw = compute_metrics(ew.get_truth(), ew.get_history())
    _say("blind zone", "round_robin POI 0, weighted_priority > 0", f"rr={mr.poi} wp={mw.poi:.3f}")
    assert mr.poi == 0.0 and mw.poi > 0.0
