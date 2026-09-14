"""Contract tests for the environment API (INTERFACE.md C1, C2, C3, C6).

Only inline lambda policies are used here; nothing is imported from
``oracle/``, ``baselines/`` or ``eval/`` so the env tests never depend on them.
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from common.types import INFO_KEYS, History, MissionSpec, Truth
from env.receiver_model import PassThroughReceiver, ReceiverModel
from env.scenario import load_scenario
from env.spectrum_env import make_env

MVP = "scenarios/mvp_periodic.json"


def _run(env, policy, n=None):
    """Run ``policy(t, obs, info) -> action`` until truncation (or ``n`` steps)."""
    obs, info = env.reset()
    out = []
    t = 0
    while True:
        a = policy(t, obs, info)
        obs, r, term, trunc, info = env.step(a)
        out.append((a, obs, r, info))
        t += 1
        if trunc or (n is not None and t >= n):
            break
    return out


# ---------------------------------------------------------------- reset / info


def test_reset_returns_zero_and_info0():
    env = make_env(MVP, 0)
    obs, info = env.reset()
    assert obs == 0 and isinstance(obs, int)
    assert set(info) == set(INFO_KEYS)
    assert info["t"] == -1 and info["next_t"] == 0
    assert info["action"] == env.mission_spec.initial_channel
    assert info["switched"] is False and info["blind"] is False and info["blind_remaining"] == 0
    assert info["reward"] == 0.0
    assert set(info["reward_terms"]) == {"new", "hit", "empty", "tune"}


def test_info_keys_and_types_every_step():
    env = make_env(MVP, 0)
    for t, (a, obs, r, info) in enumerate(_run(env, lambda t, o, i: (t * 3) % 50, n=200)):
        assert set(info) == set(INFO_KEYS)
        assert isinstance(obs, int) and obs in (0, 1)
        assert isinstance(r, float)
        assert isinstance(info["t"], int) and isinstance(info["next_t"], int)
        assert isinstance(info["switched"], bool) and isinstance(info["blind"], bool)
        assert isinstance(info["blind_remaining"], int)
        assert isinstance(info["reward"], float)
        assert info["t"] == t and info["next_t"] == t + 1
        assert info["action"] == a
        assert abs(sum(info["reward_terms"].values()) - info["reward"]) < 1e-9
        assert info["reward"] == r


def test_mission_spec_is_mission_spec_without_truth():
    env = make_env(MVP, 0)
    spec = env.mission_spec
    assert isinstance(spec, MissionSpec)
    assert spec.N == 50 and spec.T == 2000 and spec.tau_switch == 1 and spec.pfa == 0.05
    assert spec.intel_weights is None
    assert not hasattr(spec, "S")


def test_truth_and_history_dtypes_shapes():
    env = make_env(MVP, 0)
    env.reset()
    tr = env.get_truth()
    assert isinstance(tr, Truth)
    N, T = 50, 2000
    assert tr.S.shape == (N, T) and tr.S.dtype == np.int8
    assert tr.S_by_emitter.shape == (2, N, T) and tr.S_by_emitter.dtype == np.int8
    assert tr.SNR.shape == (N, T) and tr.SNR.dtype == np.float32
    assert tr.E.shape == (N, T) and tr.E.dtype == np.int16
    assert tr.U.shape == (N, T) and tr.U.dtype == np.float32
    assert np.all((tr.U >= 0) & (tr.U < 1))
    assert np.all(np.isneginf(tr.SNR[tr.S == 0])) and np.all(tr.E[tr.S == 0] == -1)
    assert np.all(tr.E[tr.S == 1] >= 0)
    assert np.array_equal(tr.S, tr.S_by_emitter.any(axis=0).astype(np.int8))
    assert tr.emitter_names == ("radar_A", "radar_B") and tr.emitter_types == ("periodic", "periodic")
    h = env.get_history()
    assert isinstance(h, History)
    assert h.t == 0 and h.actions.shape == (T,) and np.all(h.actions == -1)
    assert h.obs.dtype == np.int8 and h.blind.dtype == bool and h.switched.dtype == bool
    assert h.new_detect.dtype == bool and h.rewards.dtype == np.float32
    env.step(7)
    h = env.get_history()
    assert h.t == 1 and h.actions[0] == 7 and h.actions[1] == -1


def test_truth_is_read_only():
    env = make_env(MVP, 0)
    env.reset()
    tr = env.get_truth()
    with pytest.raises(ValueError):
        tr.S[0, 0] = 1


# ---------------------------------------------------------------- step semantics


def test_passthrough_obs_equals_truth_row():
    env = make_env(MVP, 0, receiver=PassThroughReceiver())
    out = _run(env, lambda t, o, i: 7)
    obs = np.array([o for _, o, _, _ in out])
    assert np.array_equal(obs, env.get_truth().S[7])
    assert obs.sum() == env.get_truth().S[7].sum() > 0


def test_worked_reward_cases():
    env = make_env(MVP, 0, receiver=PassThroughReceiver())
    env.reset()
    S = env.get_truth().S
    assert S[7, 0] == 1 and S[7, 1] == 1 and S[5, 2] == 0 and S[5, 3] == 0
    _, r, _, _, info = env.step(7)  # switch + first true hit
    assert r == 10.5 and info["reward_terms"] == {"new": 10.0, "hit": 2.0, "empty": 0.0, "tune": -1.5}
    _, r, _, _, info = env.step(7)  # stay + known hit
    assert r == 2.0 and info["reward_terms"]["new"] == 0.0
    _, r, _, _, info = env.step(5)  # switch + empty
    assert r == -2.0
    _, r, _, _, info = env.step(5)  # stay + empty
    assert r == -0.5


def test_no_tune_cost_when_starting_on_initial_channel():
    env = make_env(MVP, 0, receiver=PassThroughReceiver())
    env.reset()
    _, r, _, _, info = env.step(env.mission_spec.initial_channel)
    assert info["switched"] is False and info["reward_terms"]["tune"] == 0.0


def test_truncated_at_T_and_step_after_end_raises():
    env = make_env(MVP, 0)
    out = _run(env, lambda t, o, i: 7)
    assert len(out) == 2000
    assert out[-1][3]["t"] == 1999 and out[-1][3]["next_t"] == 2000
    with pytest.raises(RuntimeError):
        env.step(7)
    env.reset()
    env.step(7)  # fine again after reset


def test_step_before_reset_raises():
    env = make_env(MVP, 0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_bad_action_raises():
    env = make_env(MVP, 0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(50)


@pytest.mark.parametrize("tau", [0, 1, 2, 3])
def test_blind_equals_switches_times_tau(tau):
    raw = json.load(open(MVP))
    raw["receiver"]["tau_switch"] = tau
    env = make_env(raw, 0)
    k = 5  # switch every k dwells between channels 7 and 33
    out = _run(env, lambda t, o, i: 7 if (t // k) % 2 == 0 else 33)
    h = env.get_history()
    n_switch = int(h.switched.sum())
    assert n_switch == 2000 // k  # initial_channel is 0, so t=0 is also a switch
    assert int(h.blind.sum()) == n_switch * tau
    assert np.all(h.obs[h.blind] == 0)
    # blind_remaining sequence after a switch is tau-1, tau-2, ..., 0
    br = [info["blind_remaining"] for _, _, _, info in out[k:k + tau + 1]]
    assert br == list(range(tau - 1, -1, -1)) + [0]
    assert env.mission_spec.tau_switch == tau


def test_switch_during_blindness_restarts_counter():
    raw = json.load(open(MVP))
    raw["receiver"]["tau_switch"] = 3
    env = make_env(raw, 0)
    env.reset()
    infos = [env.step(a)[4] for a in (1, 2, 2, 2, 2, 2)]
    assert [i["blind"] for i in infos] == [True, True, True, True, False, False]
    assert [i["blind_remaining"] for i in infos] == [2, 2, 1, 0, 0, 0]


def test_i_new_only_once_per_emitter():
    env = make_env(MVP, 0, receiver=PassThroughReceiver())
    _run(env, lambda t, o, i: 7)
    h = env.get_history()
    assert int(h.new_detect.sum()) == 1
    env = make_env(MVP, 0, receiver=PassThroughReceiver())
    _run(env, lambda t, o, i: 7 if t < 100 else 33)
    assert int(env.get_history().new_detect.sum()) == 2


# ---------------------------------------------------------------- determinism / CRN


def test_determinism_same_seed_same_history():
    pol = lambda t, o, i: (t // 2) % 50  # noqa: E731
    a = make_env(MVP, 3)
    b = make_env(MVP, 3)
    oa = _run(a, pol)
    ob = _run(b, pol)
    assert [x[1] for x in oa] == [x[1] for x in ob]
    assert np.array_equal(a.get_history().rewards, b.get_history().rewards)


def test_common_random_numbers_independent_of_actions():
    a = make_env(MVP, 5)
    b = make_env(MVP, 5)
    _run(a, lambda t, o, i: 7)
    _run(b, lambda t, o, i: (t * 7) % 50)
    assert np.array_equal(a.get_truth().S, b.get_truth().S)
    assert np.array_equal(a.get_truth().U, b.get_truth().U)
    assert np.array_equal(a.get_truth().E, b.get_truth().E)


def test_reset_seed_none_reuses_make_env_seed():
    env = make_env(MVP, 11)
    env.reset()
    u1 = env.get_truth().U.copy()
    env.reset()
    assert np.array_equal(env.get_truth().U, u1)
    env.reset(seed=12)
    assert not np.array_equal(env.get_truth().U, u1)


def test_seed_changes_noise_and_randomized_phase():
    raw = json.load(open(MVP))
    raw["randomize"]["periodic_phase"] = True
    a, b = make_env(raw, 0), make_env(raw, 1)
    a.reset()
    b.reset()
    assert not np.array_equal(a.get_truth().U, b.get_truth().U)
    assert not np.array_equal(a.get_truth().S, b.get_truth().S)
    # randomize off: S identical across seeds, U differs
    a0, b0 = make_env(MVP, 0), make_env(MVP, 1)
    a0.reset()
    b0.reset()
    assert np.array_equal(a0.get_truth().S, b0.get_truth().S)


# ---------------------------------------------------------------- scenario loader


def test_load_scenario_and_validator_messages():
    cfg = load_scenario(MVP)
    assert cfg.N == 50 and cfg.T == 2000 and len(cfg.emitters) == 2
    raw = json.load(open(MVP))
    cases = {
        "typo": lambda d: d.__setitem__("typo", 1),
        "channel": lambda d: d["emitters"][0].__setitem__("channel", 50),
        "onoff": lambda d: d["emitters"][0].__setitem__("onoff", [0, 1000, 1000]),
        "intel_weights": lambda d: d.__setitem__("intel_weights", [1.0] * 49),
        "type": lambda d: d["emitters"][0].__setitem__("type", "pulsed"),
        "emitters[1].channel": lambda d: d["emitters"][1].__setitem__("channel", 7),
        "noise_model": lambda d: d["receiver"].__setitem__("noise_model", "gauss"),
    }
    for field, mutate in cases.items():
        d = copy.deepcopy(raw)
        mutate(d)
        with pytest.raises(ValueError, match=re.escape(field)):
            load_scenario(d)


def test_scenario_path_resolves_against_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_scenario(MVP)
    assert cfg.name == "mvp_periodic"


# ---------------------------------------------------------------- receiver unit


def test_receiver_blind_counter_unit():
    r = ReceiverModel(2, 1e-3)
    r.reset(0)
    seq = [r.observe(1, 1, 30.0, 0.5) for _ in range(3)]
    assert [(o.blind, o.blind_remaining, o.obs) for o in seq] == [(True, 1, 0), (True, 0, 0), (False, 0, 1)]
    assert seq[0].switched and not seq[1].switched
    fixed = ReceiverModel(1, 0.05, "fixed", 0.85)
    fixed.reset(0)
    assert fixed.observe(0, 1, 0.0, 0.84).obs == 1 and fixed.observe(0, 1, 0.0, 0.86).obs == 0
    assert fixed.observe(0, 0, -np.inf, 0.04).obs == 1 and fixed.observe(0, 0, -np.inf, 0.06).obs == 0
    assert PassThroughReceiver().observe(3, 1, 10.0, 0.99) == (1, False, False, 0)


# ---------------------------------------------------------------- gymnasium


def test_gymnasium_check_env():
    check_env(make_env(MVP, 0), skip_render_check=True)


def test_render_returns_string():
    env = make_env(MVP, 0)
    env.reset()
    for _ in range(30):
        env.step(7)
    s = env.render(width=40)
    assert isinstance(s, str) and "#" in s and "@" in s
