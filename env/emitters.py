"""Open-loop emitter models and truth rendering (contract C4 / C6).

Every emitter is open-loop (it never reacts to the receiver), so the whole
truth tensor is rendered once at ``reset()``; per-step writes would give an
identical result and are not needed.

Attribution: the ``onoff = [start, on, off]`` duty-cycle convention, weighted
channel hopping via ``rng.choice(channels, p=hop_probs)``, the per-cell
emitter-id matrix and the JSON type registry are ideas borrowed from
rfrl-gym (MIT licence, https://github.com/jsnyder13/rfrl-gym).  The code here
is written fresh; nothing is copied.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

from common.types import Truth
from env.receiver_model import sinc2_gain
from env.scenario import AgileCfg, PeriodicCfg, ScanningCfg, ScenarioConfig, load_scenario

MAINLOBE_HALFWIDTH = 1.129  # first null of the sinc^2 pattern in units of the 3-dB beamwidth


def onoff_mask(T: int, start: int, on: int, off: int) -> np.ndarray:
    """Closed-form duty cycle: ``active[t] = t >= start and ((t - start) % (on + off)) < on``."""
    t = np.arange(T, dtype=np.int64)
    return (t >= start) & (((t - start) % (on + off)) < on)


class Emitter:
    """Base class.  ``render(N, T) -> (S_m int8 (N,T), SNR_m float32 (N,T))``."""

    type: str = "base"

    def __init__(self, name: str, rng: np.random.Generator) -> None:
        self.name = name
        self.rng = rng

    def render(self, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError


class PeriodicRadar(Emitter):
    """Fixed-channel pulsed radar: on for ``on`` slots, off for ``off`` slots, PRI = on + off."""

    type = "periodic"

    def __init__(self, channel: int, onoff: tuple[int, int, int], snr_db: float, name: str = "periodic",
                 rng: np.random.Generator | None = None, randomize_phase: bool = False) -> None:
        super().__init__(name, rng if rng is not None else np.random.default_rng(0))
        self.channel = int(channel)
        self.start, self.on, self.off = (int(x) for x in onoff)
        self.pri = self.on + self.off
        self.snr_db = float(snr_db)
        if randomize_phase:
            self.start += int(self.rng.integers(0, self.pri))

    def render(self, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        S = np.zeros((N, T), dtype=np.int8)
        SNR = np.full((N, T), -np.inf, dtype=np.float32)
        mask = onoff_mask(T, self.start, self.on, self.off)
        S[self.channel, mask] = 1
        SNR[self.channel, mask] = self.snr_db
        return S, SNR


class AgileJammer(Emitter):
    """Frequency hopper over a channel set.

    Every ``hop_period`` slots it keeps the current channel with probability
    ``p_stay``, else draws a *different* channel from ``channels`` with
    ``hop_probs`` (uniform if ``None``) renormalised over the others.  It emits
    only while onoff-active and occupies exactly one channel per active slot.

    Because the draw excludes the current channel, the long-run occupancy is
    the stationary distribution of the induced Markov chain, which equals
    ``hop_probs`` exactly only when ``hop_probs`` is uniform.  Use
    :meth:`stationary_distribution` to get the analytic occupancy.

    ``transition_matrix`` (row-stochastic over ``channels``) replaces the
    ``p_stay`` / ``hop_probs`` rule when given (stretch S26).
    """

    type = "agile"

    def __init__(self, channels: tuple[int, ...], hop_period: int, hop_probs: tuple[float, ...] | None,
                 p_stay: float, onoff: tuple[int, int, int], snr_db: float, name: str = "agile",
                 rng: np.random.Generator | None = None, randomize_start: bool = False,
                 transition_matrix: tuple[tuple[float, ...], ...] | None = None) -> None:
        super().__init__(name, rng if rng is not None else np.random.default_rng(0))
        self.channels = tuple(int(c) for c in channels)
        K = len(self.channels)
        self.hop_period = int(hop_period)
        self.p_stay = float(p_stay)
        self.start, self.on, self.off = (int(x) for x in onoff)
        self.snr_db = float(snr_db)
        probs = np.full(K, 1.0 / K) if hop_probs is None else np.asarray(hop_probs, dtype=np.float64)
        self.hop_probs = probs / probs.sum()
        if transition_matrix is not None:
            P = np.asarray(transition_matrix, dtype=np.float64)
            P = P / P.sum(axis=1, keepdims=True)
        else:
            P = np.zeros((K, K))
            for c in range(K):
                others = self.hop_probs.copy()
                others[c] = 0.0
                s = others.sum()
                if s > 0:
                    P[c] = (1.0 - self.p_stay) * others / s
                    P[c, c] = self.p_stay
                else:  # only one channel has mass: cannot leave it
                    P[c, c] = 1.0
        self.P = P
        # cumulative rows as Python lists for a fast per-hop bisect
        self._cum = [list(np.cumsum(P[c])) for c in range(K)]
        self.randomize_start = randomize_start

    def stationary_distribution(self) -> np.ndarray:
        """Left eigenvector of the hop transition matrix (occupancy over ``channels``)."""
        w, v = np.linalg.eig(self.P.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, i])
        pi = pi / pi.sum()
        return pi

    def hop_sequence(self, T: int) -> np.ndarray:
        """Channel index (into ``channels``) for each hop; length ``ceil(T / hop_period)``."""
        n_hops = max(1, math.ceil(T / self.hop_period))
        if self.randomize_start:
            cur = int(self.rng.choice(len(self.channels), p=self.hop_probs))
        else:
            cur = 0
        u = self.rng.random(n_hops)
        seq = np.empty(n_hops, dtype=np.int64)
        seq[0] = cur
        cum = self._cum
        for k in range(1, n_hops):
            cur = bisect.bisect_right(cum[cur], float(u[k]))
            if cur >= len(cum):  # guard against cumulative rounding at exactly 1.0
                cur = len(cum) - 1
            seq[k] = cur
        return seq

    def render(self, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        S = np.zeros((N, T), dtype=np.int8)
        SNR = np.full((N, T), -np.inf, dtype=np.float32)
        seq = self.hop_sequence(T)
        chan_idx = np.repeat(seq, self.hop_period)[:T]
        chan = np.asarray(self.channels, dtype=np.int64)[chan_idx]
        active = onoff_mask(T, self.start, self.on, self.off)
        t_act = np.nonzero(active)[0]
        S[chan[t_act], t_act] = 1
        SNR[chan[t_act], t_act] = self.snr_db
        return S, SNR


class ScanningRadar(Emitter):
    """Rotating search radar (``mode="lighthouse"``) on a fixed channel.

    ``theta(t) = (360*t/T_rot + phase0) mod 360``; the receiver sits at azimuth 0
    so ``delta = wrap(theta(t))``.  ``SNR_m = snr_peak_db + 10*log10(sinc2_gain(delta, bw))``
    and ``S_m = 1`` iff ``|delta| < 1.129*bw`` (mainlobe only) and
    ``SNR_m - snr_peak_db >= presence_gain_db``.  The first sidelobe is -13.3 dB
    and never registers at the default -10 dB threshold.  Exactly one burst per
    rotation of length ``round(T_rot * 1.667*bw / 360) +- 1`` at -10 dB.
    """

    type = "scanning"

    def __init__(self, channel: int, T_rot: int, beamwidth_deg: float, snr_peak_db: float,
                 phase0_deg: float = 0.0, presence_gain_db: float = -10.0, mode: str = "lighthouse",
                 name: str = "scanning", rng: np.random.Generator | None = None,
                 randomize_phase: bool = False) -> None:
        super().__init__(name, rng if rng is not None else np.random.default_rng(0))
        if mode != "lighthouse":
            raise ValueError(f"unsupported scanning mode {mode!r}")
        self.mode = mode
        self.channel = int(channel)
        self.T_rot = int(T_rot)
        self.beamwidth_deg = float(beamwidth_deg)
        self.snr_peak_db = float(snr_peak_db)
        self.phase0_deg = float(phase0_deg)
        self.presence_gain_db = float(presence_gain_db)
        if randomize_phase:
            self.phase0_deg = (self.phase0_deg + float(self.rng.uniform(0.0, 360.0))) % 360.0

    def render(self, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:
        S = np.zeros((N, T), dtype=np.int8)
        SNR = np.full((N, T), -np.inf, dtype=np.float32)
        t = np.arange(T, dtype=np.float64)
        theta = (360.0 * t / self.T_rot + self.phase0_deg) % 360.0
        delta = np.where(theta > 180.0, theta - 360.0, theta)  # wrap to (-180, 180]
        gain = sinc2_gain(delta, self.beamwidth_deg)
        with np.errstate(divide="ignore"):
            rel_db = 10.0 * np.log10(gain)
        mainlobe = np.abs(delta) < MAINLOBE_HALFWIDTH * self.beamwidth_deg
        present = mainlobe & (rel_db >= self.presence_gain_db)
        S[self.channel, present] = 1
        SNR[self.channel, present] = (self.snr_peak_db + rel_db[present]).astype(np.float32)
        return S, SNR


def build_emitters(cfg: ScenarioConfig, seed: int) -> list[Emitter]:
    """Instantiate every emitter in ``cfg``; emitter ``i`` uses ``default_rng([seed, i])``."""
    out: list[Emitter] = []
    rz = cfg.randomize
    for i, e in enumerate(cfg.emitters):
        rng = np.random.default_rng([int(seed), i])
        if isinstance(e, PeriodicCfg):
            out.append(PeriodicRadar(e.channel, e.onoff, e.snr_db, name=e.name, rng=rng,
                                     randomize_phase=rz.periodic_phase))
        elif isinstance(e, AgileCfg):
            out.append(AgileJammer(e.channels, e.hop_period, e.hop_probs, e.p_stay, e.onoff, e.snr_db,
                                   name=e.name, rng=rng, randomize_start=rz.agile_start_channel,
                                   transition_matrix=e.transition_matrix))
        elif isinstance(e, ScanningCfg):
            out.append(ScanningRadar(e.channel, e.T_rot, e.beamwidth_deg, e.snr_peak_db, e.phase0_deg,
                                     e.presence_gain_db, mode=e.mode, name=e.name, rng=rng,
                                     randomize_phase=rz.scanning_phase))
        else:  # pragma: no cover - validator guarantees the closed set
            raise TypeError(f"unknown emitter cfg {type(e).__name__}")
    return out


def render_truth(cfg: ScenarioConfig | dict | str, seed: int) -> Truth:
    """Render the full hidden truth for ``(scenario, seed)``.

    ``S = OR``, ``SNR = max``, ``E = argmax SNR`` (ties -> lowest id), ``U`` from
    ``default_rng([seed, 10_000])``.  Same ``(scenario, seed)`` gives bit-identical
    output for every scheduler (common random numbers).
    """
    cfg = load_scenario(cfg)
    N, T = cfg.N, cfg.T
    emitters = build_emitters(cfg, seed)
    M = len(emitters)
    S_by = np.zeros((M, N, T), dtype=np.int8)
    SNR_by = np.full((M, N, T), -np.inf, dtype=np.float32)
    for m, em in enumerate(emitters):
        S_m, SNR_m = em.render(N, T)
        S_by[m] = S_m
        SNR_by[m] = SNR_m
    if M > 0:
        S = S_by.any(axis=0).astype(np.int8)
        SNR = SNR_by.max(axis=0).astype(np.float32)
        E = SNR_by.argmax(axis=0).astype(np.int16)
        E[S == 0] = -1
    else:
        S = np.zeros((N, T), dtype=np.int8)
        SNR = np.full((N, T), -np.inf, dtype=np.float32)
        E = np.full((N, T), -1, dtype=np.int16)
    U = np.random.default_rng([int(seed), 10_000]).random((N, T), dtype=np.float32)
    for arr in (S, S_by, SNR, E, U):
        arr.setflags(write=False)
    return Truth(
        S=S, S_by_emitter=S_by, SNR=SNR, E=E, U=U,
        emitter_names=tuple(em.name for em in emitters),
        emitter_types=tuple(em.type for em in emitters),
    )
