"""Single-channel receiver model (contract C3).

The receiver is the single owner of the previous-channel and blind state.  It
is *pure* with respect to randomness: the env owns the pre-drawn uniform noise
``U`` and passes ``u = U[a, t]`` into :meth:`ReceiverModel.observe`, which is
what makes the comparison between schedulers paired (common random numbers).

Detection model
---------------
Bernoulli detection with probability ``Pd(SNR)`` when the channel is truly
occupied and ``Pfa`` when it is empty.  ``Pd(SNR)`` comes from Albersheim's
equation (``n_pulses`` non-coherently integrated pulses, default 1) with a
documented low-SNR blend; ``noise_model="fixed"`` uses a constant ``pd_fixed``
instead (the theory-guide 0.85 / 0.05 worked examples).

Validity: Albersheim's closed form is quoted accurate to ~0.2 dB for
``1e-7 <= Pfa <= 1e-3`` and ``0.1 <= Pd <= 0.9``.  Below ``Pd = 0.1`` the
closed form floors at ~0.069 (Pfa = 1e-3) as SNR -> -inf, so this module blends
to ``Pfa`` with a 10 dB/decade law below ``snr_lo = albersheim_snr_db(0.1, pfa)``.
The blend is continuous and monotone and is a stated limitation.
"""
from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np

_ArrayLike = float | np.ndarray


def _k(n_pulses: int) -> float:
    return 6.2 + 4.54 / math.sqrt(n_pulses + 0.44)


def albersheim_snr_db(pd: float, pfa: float, n_pulses: int = 1) -> float:
    """SNR (dB) required for ``pd`` at ``pfa`` with ``n_pulses`` (Albersheim).

    ``A = ln(0.62/pfa)``, ``B = ln(pd/(1-pd))``, ``K = 6.2 + 4.54/sqrt(n+0.44)``,
    ``snr = -5*log10(n) + K*log10(A + 0.12*A*B + 1.7*B)``.
    Golden value: ``(0.9, 1e-6) -> 13.1 +/- 0.15 dB``.
    """
    if not (0.0 < pd < 1.0):
        raise ValueError(f"pd must be in (0,1), got {pd}")
    if not (0.0 < pfa < 1.0):
        raise ValueError(f"pfa must be in (0,1), got {pfa}")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    A = math.log(0.62 / pfa)
    B = math.log(pd / (1.0 - pd))
    K = _k(n_pulses)
    inner = A + 0.12 * A * B + 1.7 * B
    if inner <= 0.0:
        return -math.inf
    return -5.0 * math.log10(n_pulses) + K * math.log10(inner)


def albersheim_pd(snr_db: _ArrayLike, pfa: float, n_pulses: int = 1, blend: bool = True) -> _ArrayLike:
    """Pd for a given SNR (dB) at ``pfa`` (inverse Albersheim), scalar or array.

    ``X = 10**((snr + 5*log10(n))/K)``, ``B = (X-A)/(0.12*A+1.7)``, ``pd = 1/(1+exp(-B))``,
    clamped to ``[pfa, 1]``.  With ``blend=True`` (default), below
    ``snr_lo = albersheim_snr_db(0.1, pfa)`` the value is
    ``pd = pfa + (0.1-pfa) * 10**((snr-snr_lo)/10)`` so that Pd -> Pfa as SNR -> -inf.

    Reference @ pfa=1e-3: -5 dB .032, 0 dB .10, 5 dB .21, 10 dB .81, 13 dB .996, 15 dB > .999.
    ``-inf`` SNR (empty cell) returns ``pfa``.
    """
    if not (0.0 < pfa < 1.0):
        raise ValueError(f"pfa must be in (0,1), got {pfa}")
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    scalar = np.isscalar(snr_db)
    snr = np.asarray(snr_db, dtype=np.float64)
    A = math.log(0.62 / pfa)
    K = _k(n_pulses)
    with np.errstate(over="ignore", invalid="ignore"):
        X = np.power(10.0, (snr + 5.0 * math.log10(n_pulses)) / K)
        B = (X - A) / (0.12 * A + 1.7)
        pd = 1.0 / (1.0 + np.exp(-B))
    pd = np.clip(pd, pfa, 1.0)
    if blend:
        snr_lo = albersheim_snr_db(0.1, pfa, n_pulses)
        low = snr < snr_lo
        if np.any(low):
            with np.errstate(over="ignore", invalid="ignore"):
                pd_low = pfa + (0.1 - pfa) * np.power(10.0, (snr - snr_lo) / 10.0)
            pd = np.where(low, pd_low, pd)
    pd = np.where(np.isneginf(snr), pfa, pd)
    if scalar:
        return float(pd)
    return pd


def sinc2_gain(delta_deg: _ArrayLike, beamwidth_deg: float) -> _ArrayLike:
    """One-way sinc^2 antenna pattern: ``x = 2.783*delta/bw``, ``gain = (sin x / x)^2``.

    ``gain(0) = 1``, ``gain(+-bw/2) = 0.5`` (3-dB beamwidth convention), first null at ``1.129*bw``.
    """
    if beamwidth_deg <= 0.0:
        raise ValueError(f"beamwidth_deg must be > 0, got {beamwidth_deg}")
    scalar = np.isscalar(delta_deg)
    x = 2.783 * np.asarray(delta_deg, dtype=np.float64) / beamwidth_deg
    g = np.sinc(x / np.pi) ** 2  # np.sinc(z) = sin(pi z)/(pi z), exact 1 at 0
    if scalar:
        return float(g)
    return g


class Observation(NamedTuple):
    obs: int
    switched: bool
    blind: bool
    blind_remaining: int


class ReceiverModel:
    """Single-channel receiver with retune blindness and Bernoulli detection.

    Blind rule: on a switch ``blind_left = tau_switch``; ``blind = blind_left > 0``;
    if blind, ``blind_left -= 1``; ``blind_remaining`` is reported post-decrement;
    a switch during blindness restarts the counter; ``tau_switch = 0`` never blinds.
    """

    def __init__(
        self,
        tau_switch: int,
        pfa: float,
        noise_model: Literal["fixed", "albersheim"] = "albersheim",
        pd_fixed: float = 0.85,
        n_pulses: int = 1,
    ) -> None:
        if tau_switch < 0:
            raise ValueError(f"tau_switch must be >= 0, got {tau_switch}")
        if not (0.0 < pfa < 1.0):
            raise ValueError(f"pfa must be in (0,1), got {pfa}")
        if noise_model not in ("fixed", "albersheim"):
            raise ValueError(f"noise_model must be 'fixed' or 'albersheim', got {noise_model!r}")
        if not (0.0 < pd_fixed <= 1.0):
            raise ValueError(f"pd_fixed must be in (0,1], got {pd_fixed}")
        self.tau_switch = int(tau_switch)
        self.pfa = float(pfa)
        self.noise_model = noise_model
        self.pd_fixed = float(pd_fixed)
        self.n_pulses = int(n_pulses)
        self._prev: int | None = None  # no previous channel until reset(); first dwell is then not a switch
        self._blind_left: int = 0
        # Albersheim constants cached for the per-step scalar path.
        self._A = math.log(0.62 / self.pfa)
        self._K = _k(self.n_pulses)
        self._snr_lo = albersheim_snr_db(0.1, self.pfa, self.n_pulses)
        self._n_off = 5.0 * math.log10(self.n_pulses)

    # -- state ---------------------------------------------------------------
    def reset(self, initial_channel: int) -> None:
        self._prev = int(initial_channel)
        self._blind_left = 0

    @property
    def current_channel(self) -> int | None:
        return self._prev

    @property
    def blind_left(self) -> int:
        return self._blind_left

    # -- detection curve -----------------------------------------------------
    def pd(self, snr_db: _ArrayLike) -> _ArrayLike:
        """Probability of detection for a truly occupied cell at ``snr_db``."""
        if self.noise_model == "fixed":
            if np.isscalar(snr_db):
                return self.pd_fixed
            return np.full(np.shape(snr_db), self.pd_fixed, dtype=np.float64)
        return albersheim_pd(snr_db, self.pfa, self.n_pulses, blend=True)

    def _pd_scalar(self, snr_db: float) -> float:
        # Fast scalar path used by observe(); identical to albersheim_pd().
        if self.noise_model == "fixed":
            return self.pd_fixed
        if snr_db == -math.inf:
            return self.pfa
        if snr_db < self._snr_lo:
            return self.pfa + (0.1 - self.pfa) * 10.0 ** ((snr_db - self._snr_lo) / 10.0)
        X = 10.0 ** ((snr_db + self._n_off) / self._K)
        B = (X - self._A) / (0.12 * self._A + 1.7)
        try:
            p = 1.0 / (1.0 + math.exp(-B))
        except OverflowError:
            p = 0.0
        if p < self.pfa:
            return self.pfa
        return 1.0 if p > 1.0 else p

    # -- one dwell -----------------------------------------------------------
    def observe(self, action: int, truth_bit: int, snr_db: float, u: float) -> Observation:
        """Execute one dwell on ``action``.  Pure given ``u``; the env owns ``U``."""
        switched = self._prev is not None and action != self._prev
        if switched:
            self._blind_left = self.tau_switch
        self._prev = action
        blind = self._blind_left > 0
        if blind:
            self._blind_left -= 1
            return Observation(0, switched, True, self._blind_left)
        p = self._pd_scalar(float(snr_db)) if truth_bit else self.pfa
        return Observation(1 if u < p else 0, switched, False, 0)


class PassThroughReceiver(ReceiverModel):
    """Debug / verification receiver: ``tau_switch = 0``, ``O_t == truth_bit``, ignores ``u``."""

    def __init__(self, pfa: float = 1e-3, **_: object) -> None:
        super().__init__(tau_switch=0, pfa=pfa, noise_model="fixed", pd_fixed=1.0)

    def pd(self, snr_db: _ArrayLike) -> _ArrayLike:
        if np.isscalar(snr_db):
            return 1.0
        return np.ones(np.shape(snr_db), dtype=np.float64)

    def observe(self, action: int, truth_bit: int, snr_db: float, u: float) -> Observation:
        switched = self._prev is not None and action != self._prev
        self._prev = action
        return Observation(1 if truth_bit else 0, switched, False, 0)
