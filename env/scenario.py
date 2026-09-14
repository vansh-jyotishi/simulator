"""Scenario loading and validation (contract C5).

A scenario is a JSON file (or an in-memory dict of the same shape) describing
the band (``N`` channels, ``T`` slots), the receiver, the reward magnitudes,
optional operator intel and the emitter list.  The seed is never stored here.

``load_scenario`` returns frozen dataclasses so nothing downstream can mutate
the configuration after the truth has been rendered.  Every validator error is
a ``ValueError`` naming the offending field.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.types import C_EMPTY, C_TUNE, EMITTER_TYPES, MissionSpec, R_HIT, R_NEW

REPO_ROOT = Path(__file__).resolve().parent.parent
NOISE_MODELS = ("fixed", "albersheim")
SCANNING_MODES = ("lighthouse",)

# ----------------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverCfg:
    tau_switch: int
    pfa: float
    noise_model: str = "albersheim"
    pd_fixed: float = 0.85
    n_pulses: int = 1
    initial_channel: int = 0


@dataclass(frozen=True)
class RewardCfg:
    R_new: float = R_NEW
    R_hit: float = R_HIT
    C_empty: float = C_EMPTY
    C_tune: float = C_TUNE


@dataclass(frozen=True)
class RandomizeCfg:
    periodic_phase: bool = False
    agile_start_channel: bool = False
    scanning_phase: bool = False


@dataclass(frozen=True)
class PeriodicCfg:
    name: str
    channel: int
    onoff: tuple[int, int, int]  # (start_delay, on, off)
    snr_db: float
    type: str = "periodic"

    @property
    def pri(self) -> int:
        return self.onoff[1] + self.onoff[2]


@dataclass(frozen=True)
class AgileCfg:
    name: str
    channels: tuple[int, ...]
    hop_period: int
    p_stay: float
    onoff: tuple[int, int, int]
    snr_db: float
    hop_probs: tuple[float, ...] | None = None
    transition_matrix: tuple[tuple[float, ...], ...] | None = None  # stretch (S26)
    type: str = "agile"


@dataclass(frozen=True)
class ScanningCfg:
    name: str
    channel: int
    T_rot: int
    beamwidth_deg: float
    snr_peak_db: float
    phase0_deg: float = 0.0
    presence_gain_db: float = -10.0
    mode: str = "lighthouse"
    type: str = "scanning"


EmitterCfg = PeriodicCfg | AgileCfg | ScanningCfg


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    N: int
    T: int
    dwell_s: float
    receiver: ReceiverCfg
    reward: RewardCfg
    randomize: RandomizeCfg
    emitters: tuple[EmitterCfg, ...]
    intel_weights: tuple[float, ...] | None = None
    path: str | None = None  # resolved source file, None for dict scenarios
    raw: dict = field(default_factory=dict, repr=False, compare=False)


# ----------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------


def _err(field_name: str, msg: str) -> ValueError:
    return ValueError(f"scenario field '{field_name}': {msg}")


def _check_keys(d: dict, allowed: set[str], required: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise _err(where, f"expected an object, got {type(d).__name__}")
    for k in d:
        if k not in allowed:
            raise _err(f"{where}.{k}" if where else k, "unknown key")
    for k in required:
        if k not in d:
            raise _err(f"{where}.{k}" if where else k, "missing required key")


def _int(d: dict, key: str, where: str, lo: int | None = None, hi: int | None = None) -> int:
    v = d[key]
    name = f"{where}.{key}" if where else key
    if isinstance(v, bool) or not isinstance(v, int):
        raise _err(name, f"expected an integer, got {v!r}")
    if lo is not None and v < lo:
        raise _err(name, f"must be >= {lo}, got {v}")
    if hi is not None and v >= hi:
        raise _err(name, f"must be < {hi}, got {v}")
    return v


def _float(d: dict, key: str, where: str, lo: float | None = None, hi: float | None = None,
           lo_open: bool = False, hi_open: bool = False) -> float:
    v = d[key]
    name = f"{where}.{key}" if where else key
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(name, f"expected a number, got {v!r}")
    v = float(v)
    if math.isnan(v):
        raise _err(name, "must not be NaN")
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise _err(name, f"must be {'>' if lo_open else '>='} {lo}, got {v}")
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise _err(name, f"must be {'<' if hi_open else '<='} {hi}, got {v}")
    return v


def _bool(d: dict, key: str, where: str) -> bool:
    v = d[key]
    if not isinstance(v, bool):
        raise _err(f"{where}.{key}", f"expected true/false, got {v!r}")
    return v


def _onoff(d: dict, where: str, T: int) -> tuple[int, int, int]:
    v = d["onoff"]
    name = f"{where}.onoff"
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise _err(name, "expected [start_delay, on, off]")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            raise _err(name, f"entries must be integers >= 0, got {v}")
    start, on, off = (int(x) for x in v)
    if on <= 0:
        raise _err(name, f"'on' must be > 0, got {on}")
    if on + off >= T:
        raise _err(name, f"PRI = on + off = {on + off} must be < T = {T}")
    return start, on, off


# ----------------------------------------------------------------------------
# Emitter parsers
# ----------------------------------------------------------------------------

_PERIODIC_KEYS = {"type", "name", "channel", "onoff", "snr_db"}
_AGILE_KEYS = {"type", "name", "channels", "hop_probs", "hop_period", "p_stay", "onoff", "snr_db",
               "transition_matrix"}
_SCANNING_KEYS = {"type", "name", "mode", "channel", "T_rot", "beamwidth_deg", "phase0_deg",
                  "snr_peak_db", "presence_gain_db"}


def _parse_periodic(d: dict, where: str, N: int, T: int) -> PeriodicCfg:
    _check_keys(d, _PERIODIC_KEYS, {"type", "name", "channel", "onoff", "snr_db"}, where)
    return PeriodicCfg(
        name=str(d["name"]),
        channel=_int(d, "channel", where, 0, N),
        onoff=_onoff(d, where, T),
        snr_db=_float(d, "snr_db", where),
    )


def _parse_agile(d: dict, where: str, N: int, T: int) -> AgileCfg:
    _check_keys(d, _AGILE_KEYS, {"type", "name", "channels", "hop_period", "p_stay", "onoff", "snr_db"}, where)
    chans = d["channels"]
    if not isinstance(chans, (list, tuple)) or len(chans) < 2:
        raise _err(f"{where}.channels", "expected a list of >= 2 channels")
    for c in chans:
        if isinstance(c, bool) or not isinstance(c, int) or not (0 <= c < N):
            raise _err(f"{where}.channels", f"every channel must be an integer in [0, {N}), got {chans}")
    if len(set(chans)) != len(chans):
        raise _err(f"{where}.channels", f"channels must be distinct, got {chans}")
    hop_probs = d.get("hop_probs")
    if hop_probs is not None:
        if not isinstance(hop_probs, (list, tuple)) or len(hop_probs) != len(chans):
            raise _err(f"{where}.hop_probs", f"length must equal len(channels) = {len(chans)}")
        for p in hop_probs:
            if isinstance(p, bool) or not isinstance(p, (int, float)) or p < 0:
                raise _err(f"{where}.hop_probs", f"entries must be numbers >= 0, got {hop_probs}")
        s = float(sum(hop_probs))
        if abs(s - 1.0) > 1e-6:
            raise _err(f"{where}.hop_probs", f"must sum to 1, got {s}")
        hop_probs = tuple(float(p) for p in hop_probs)
    tm = d.get("transition_matrix")
    if tm is not None:
        K = len(chans)
        if not isinstance(tm, (list, tuple)) or len(tm) != K:
            raise _err(f"{where}.transition_matrix", f"must be a {K}x{K} row-stochastic matrix")
        rows = []
        for r in tm:
            if not isinstance(r, (list, tuple)) or len(r) != K:
                raise _err(f"{where}.transition_matrix", f"must be a {K}x{K} row-stochastic matrix")
            for p in r:
                if isinstance(p, bool) or not isinstance(p, (int, float)) or p < 0:
                    raise _err(f"{where}.transition_matrix", "entries must be numbers >= 0")
            if abs(float(sum(r)) - 1.0) > 1e-6:
                raise _err(f"{where}.transition_matrix", "every row must sum to 1")
            rows.append(tuple(float(p) for p in r))
        tm = tuple(rows)
    return AgileCfg(
        name=str(d["name"]),
        channels=tuple(int(c) for c in chans),
        hop_probs=hop_probs,
        hop_period=_int(d, "hop_period", where, 1),
        p_stay=_float(d, "p_stay", where, 0.0, 1.0),
        onoff=_onoff(d, where, T),
        snr_db=_float(d, "snr_db", where),
        transition_matrix=tm,
    )


def _parse_scanning(d: dict, where: str, N: int, T: int) -> ScanningCfg:
    _check_keys(d, _SCANNING_KEYS, {"type", "name", "channel", "T_rot", "beamwidth_deg", "snr_peak_db"}, where)
    mode = d.get("mode", "lighthouse")
    if mode not in SCANNING_MODES:
        raise _err(f"{where}.mode", f"must be one of {SCANNING_MODES}, got {mode!r}")
    T_rot = _int(d, "T_rot", where, 1)
    if T_rot >= T:
        raise _err(f"{where}.T_rot", f"must be < T = {T}, got {T_rot}")
    return ScanningCfg(
        name=str(d["name"]),
        mode=mode,
        channel=_int(d, "channel", where, 0, N),
        T_rot=T_rot,
        beamwidth_deg=_float(d, "beamwidth_deg", where, 0.0, 360.0, lo_open=True, hi_open=True),
        phase0_deg=_float(d, "phase0_deg", where) if "phase0_deg" in d else 0.0,
        snr_peak_db=_float(d, "snr_peak_db", where),
        presence_gain_db=_float(d, "presence_gain_db", where, hi=0.0) if "presence_gain_db" in d else -10.0,
    )


_EMITTER_PARSERS = {"periodic": _parse_periodic, "agile": _parse_agile, "scanning": _parse_scanning}

# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------

_TOP_KEYS = {"name", "N", "T", "dwell_s", "receiver", "reward", "intel_weights", "randomize", "emitters"}
_RECEIVER_KEYS = {"tau_switch", "pfa", "noise_model", "pd_fixed", "n_pulses", "initial_channel"}
_REWARD_KEYS = {"R_new", "R_hit", "C_empty", "C_tune"}
_RANDOMIZE_KEYS = {"periodic_phase", "agile_start_channel", "scanning_phase"}


def resolve_scenario_path(path: str | Path) -> Path:
    """Relative paths resolve against the cwd first, then the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    alt = REPO_ROOT / p
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"scenario file not found: {path} (tried cwd and {REPO_ROOT})")


def load_scenario(scenario: str | Path | dict) -> ScenarioConfig:
    """Load and validate a scenario from a path or a dict.

    Raises ``ValueError`` naming the field for every contract violation.
    """
    if isinstance(scenario, ScenarioConfig):
        return scenario
    path: str | None = None
    if isinstance(scenario, (str, Path)):
        p = resolve_scenario_path(scenario)
        path = str(p)
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif isinstance(scenario, dict):
        raw = scenario
    else:
        raise TypeError(f"scenario must be a path or dict, got {type(scenario).__name__}")

    _check_keys(raw, _TOP_KEYS, {"N", "T", "emitters", "receiver"}, "")
    N = _int(raw, "N", "", 1)
    T = _int(raw, "T", "", 1)
    dwell_s = _float(raw, "dwell_s", "", 0.0, lo_open=True) if "dwell_s" in raw else 1e-3
    name = str(raw.get("name", Path(path).stem if path else "dict_scenario"))

    r = raw["receiver"]
    _check_keys(r, _RECEIVER_KEYS, {"tau_switch", "pfa"}, "receiver")
    noise_model = r.get("noise_model", "albersheim")
    if noise_model not in NOISE_MODELS:
        raise _err("receiver.noise_model", f"must be one of {NOISE_MODELS}, got {noise_model!r}")
    receiver = ReceiverCfg(
        tau_switch=_int(r, "tau_switch", "receiver", 0),
        pfa=_float(r, "pfa", "receiver", 0.0, 1.0, lo_open=True, hi_open=True),
        noise_model=noise_model,
        pd_fixed=_float(r, "pd_fixed", "receiver", 0.0, 1.0, lo_open=True) if "pd_fixed" in r else 0.85,
        n_pulses=_int(r, "n_pulses", "receiver", 1) if "n_pulses" in r else 1,
        initial_channel=_int(r, "initial_channel", "receiver", 0, N) if "initial_channel" in r else 0,
    )

    rw = raw.get("reward", {})
    _check_keys(rw, _REWARD_KEYS, set(), "reward")
    reward = RewardCfg(
        R_new=_float(rw, "R_new", "reward", 0.0) if "R_new" in rw else R_NEW,
        R_hit=_float(rw, "R_hit", "reward", 0.0) if "R_hit" in rw else R_HIT,
        C_empty=_float(rw, "C_empty", "reward", 0.0) if "C_empty" in rw else C_EMPTY,
        C_tune=_float(rw, "C_tune", "reward", 0.0) if "C_tune" in rw else C_TUNE,
    )

    rz = raw.get("randomize", {})
    _check_keys(rz, _RANDOMIZE_KEYS, set(), "randomize")
    randomize = RandomizeCfg(
        periodic_phase=_bool(rz, "periodic_phase", "randomize") if "periodic_phase" in rz else False,
        agile_start_channel=_bool(rz, "agile_start_channel", "randomize") if "agile_start_channel" in rz else False,
        scanning_phase=_bool(rz, "scanning_phase", "randomize") if "scanning_phase" in rz else False,
    )

    iw = raw.get("intel_weights")
    intel_weights: tuple[float, ...] | None = None
    if iw is not None:
        if not isinstance(iw, (list, tuple)) or len(iw) != N:
            raise _err("intel_weights", f"length must equal N = {N}")
        for w in iw:
            if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
                raise _err("intel_weights", "entries must be numbers >= 0")
        if float(sum(iw)) <= 0.0:
            raise _err("intel_weights", "sum must be > 0")
        intel_weights = tuple(float(w) for w in iw)

    ems = raw["emitters"]
    if not isinstance(ems, list):
        raise _err("emitters", "expected a list")
    emitters: list[EmitterCfg] = []
    fixed_channels: dict[int, str] = {}
    names: set[str] = set()
    for i, e in enumerate(ems):
        where = f"emitters[{i}]"
        if not isinstance(e, dict):
            raise _err(where, "expected an object")
        etype = e.get("type")
        if etype not in EMITTER_TYPES:
            raise _err(f"{where}.type", f"must be one of {EMITTER_TYPES}, got {etype!r}")
        if "name" not in e or not isinstance(e["name"], str) or not e["name"]:
            raise _err(f"{where}.name", "expected a non-empty string")
        cfg = _EMITTER_PARSERS[etype](e, where, N, T)
        if cfg.name in names:
            raise _err(f"{where}.name", f"duplicate emitter name {cfg.name!r}")
        names.add(cfg.name)
        if isinstance(cfg, (PeriodicCfg, ScanningCfg)):
            if cfg.channel in fixed_channels:
                raise _err(f"{where}.channel",
                           f"channel {cfg.channel} already used by fixed-channel emitter {fixed_channels[cfg.channel]!r}")
            fixed_channels[cfg.channel] = cfg.name
        emitters.append(cfg)

    return ScenarioConfig(
        name=name, N=N, T=T, dwell_s=dwell_s, receiver=receiver, reward=reward,
        randomize=randomize, emitters=tuple(emitters), intel_weights=intel_weights,
        path=path, raw=raw,
    )


def to_mission_spec(cfg: ScenarioConfig, tau_switch: int | None = None, pfa: float | None = None) -> MissionSpec:
    """Operator-visible mission spec.  ``tau_switch`` / ``pfa`` may be overridden by the receiver in use."""
    return MissionSpec(
        N=cfg.N,
        T=cfg.T,
        dwell_s=cfg.dwell_s,
        tau_switch=cfg.receiver.tau_switch if tau_switch is None else int(tau_switch),
        pfa=cfg.receiver.pfa if pfa is None else float(pfa),
        initial_channel=cfg.receiver.initial_channel,
        intel_weights=cfg.intel_weights,
    )
