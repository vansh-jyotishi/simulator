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

Public API
----------
albersheim_snr_db, albersheim_pd, sinc2_gain
    Pure numerical helpers (Albersheim detection curve and the sinc^2 antenna pattern).
Observation
    Result tuple returned by one dwell.
ReceiverModel, PassThroughReceiver
    Stateful receivers; ``PassThroughReceiver`` is the debug / verification variant.
"""
from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np

_ArrayLike = float | np.ndarray


def _k(n_pulses: int) -> float:
    """Albersheim ``K`` constant: ``6.2 + 4.54/sqrt(n_pulses + 0.44)``."""
    return 6.2 + 4.54 / math.sqrt(n_pulses + 0.44)


def albersheim_snr_db(pd: float, pfa: float, n_pulses: int = 1) -> float:
    """SNR (dB) required for ``pd`` at ``pfa`` with ``n_pulses`` (Albersheim).

    Closed-form Albersheim equation for the SNR needed to reach a target
    probability of detection under non-coherent pulse integration.

    Parameters
    ----------
    pd : float
        Target probability of detection, strictly inside ``(0, 1)``.
    pfa : float
        Probability of false alarm, strictly inside ``(0, 1)``.
    n_pulses : int, optional
        Number of non-coherently integrated pulses (``>= 1``). Default is 1.

    Returns
    -------
    float
        Required SNR in dB.  ``-inf`` when the logarithm argument is non-positive
        (only reachable for extreme ``pd`` / ``pfa`` combinations).

    Raises
    ------
    ValueError
        If ``pd`` or ``pfa`` is outside ``(0, 1)``, or ``n_pulses < 1``.

    Notes
    -----
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

    Parameters
    ----------
    snr_db : float or numpy.ndarray
        Signal-to-noise ratio in dB.  A scalar yields a scalar; an array is
        evaluated element-wise.  ``-inf`` denotes an empty cell.
    pfa : float
        Probability of false alarm, strictly inside ``(0, 1)``.
    n_pulses : int, optional
        Number of non-coherently integrated pulses (``>= 1``). Default is 1.
    blend : bool, optional
        If True (default), apply the documented low-SNR blend below
        ``snr_lo = albersheim_snr_db(0.1, pfa, n_pulses)`` so that Pd -> Pfa as SNR -> -inf.

    Returns
    -------
    float or numpy.ndarray
        Probability of detection clamped to ``[pfa, 1]``; a Python ``float`` for scalar
        input, otherwise a ``float64`` array with the shape of ``snr_db``.
        ``-inf`` SNR (empty cell) returns ``pfa``.

    Raises
    ------
    ValueError
        If ``pfa`` is outside ``(0, 1)`` or ``n_pulses < 1``.

    Notes
    -----
    ``X = 10**((snr + 5*log10(n))/K)``, ``B = (X-A)/(0.12*A+1.7)``, ``pd = 1/(1+exp(-B))``,
    clamped to ``[pfa, 1]``.  With ``blend=True`` (default), below
    ``snr_lo = albersheim_snr_db(0.1, pfa)`` the value is
    ``pd = pfa + (0.1-pfa) * 10**((snr-snr_lo)/10)`` so that Pd -> Pfa as SNR -> -inf.

    Reference @ pfa=1e-3: -5 dB .032, 0 dB .10, 5 dB .21, 10 dB .81, 13 dB .996, 15 dB > .999.
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

    Parameters
    ----------
    delta_deg : float or numpy.ndarray
        Angular offset from boresight in degrees (scalar or element-wise array).
    beamwidth_deg : float
        3-dB beamwidth in degrees; must be ``> 0``.

    Returns
    -------
    float or numpy.ndarray
        One-way power gain in ``[0, 1]``; a Python ``float`` for scalar input, otherwise a
        ``float64`` array with the shape of ``delta_deg``.

    Raises
    ------
    ValueError
        If ``beamwidth_deg <= 0``.

    Notes
    -----
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
    """Result of one receiver dwell (see :meth:`ReceiverModel.observe`).

    Attributes
    ----------
    obs : int
        Detection bit ``O_t``: 1 if the receiver reported a detection, else 0.  Always 0 while blind.
    switched : bool
        True if this dwell retuned to a different channel than the previous dwell.
    blind : bool
        True if the receiver was blind (post-retune) during this dwell; no detection is possible.
    blind_remaining : int
        Blind dwells still to come after this one (reported post-decrement); 0 when not blind.
    """
    obs: int
    switched: bool
    blind: bool
    blind_remaining: int


class ReceiverModel:
    """Single-channel receiver with retune blindness and Bernoulli detection.

    The receiver owns the previous-channel and blind state; randomness is supplied
    by the caller as the uniform draw ``u`` passed to :meth:`observe`.

    Parameters
    ----------
    tau_switch : int
        Number of blind dwells after a channel switch (``>= 0``); ``0`` never blinds.
    pfa : float
        Probability of false alarm, strictly inside ``(0, 1)``.
    noise_model : {"fixed", "albersheim"}, optional
        ``"albersheim"`` (default) derives ``Pd`` from SNR via :func:`albersheim_pd`;
        ``"fixed"`` uses the constant ``pd_fixed`` regardless of SNR.
    pd_fixed : float, optional
        Constant probability of detection for ``noise_model="fixed"``, in ``(0, 1]``. Default 0.85.
    n_pulses : int, optional
        Non-coherently integrated pulses for the Albersheim curve (``>= 1``). Default 1.

    Attributes
    ----------
    tau_switch : int
        Validated blind duration in dwells.
    pfa : float
        Validated false-alarm probability.
    noise_model : str
        ``"fixed"`` or ``"albersheim"``.
    pd_fixed : float
        Constant ``Pd`` used by the ``"fixed"`` model.
    n_pulses : int
        Integrated pulse count used by the Albersheim curve.
    current_channel : int or None
        Channel of the most recent dwell (``None`` before :meth:`reset`).
    blind_left : int
        Blind dwells remaining.

    Raises
    ------
    ValueError
        If any constructor argument is outside its documented range.

    Notes
    -----
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
        """Validate the arguments, store them and cache the Albersheim constants."""
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
        """Reset the receiver state for a new episode.

        Parameters
        ----------
        initial_channel : int
            Channel the receiver is tuned to before the first dwell.  The first
            :meth:`observe` call on this channel is therefore not a switch, and the
            blind counter is cleared.
        """
        self._prev = int(initial_channel)
        self._blind_left = 0

    @property
    def current_channel(self) -> int | None:
        """Channel of the most recent dwell.

        Returns
        -------
        int or None
            Channel index of the last dwell, or ``None`` before :meth:`reset`.
        """
        return self._prev

    @property
    def blind_left(self) -> int:
        """Number of blind dwells remaining after the most recent dwell.

        Returns
        -------
        int
            Remaining blind dwells; ``0`` when the receiver is not blind.
        """
        return self._blind_left

    # -- detection curve -----------------------------------------------------
    def pd(self, snr_db: _ArrayLike) -> _ArrayLike:
        """Probability of detection for a truly occupied cell at ``snr_db``.

        Parameters
        ----------
        snr_db : float or numpy.ndarray
            Signal-to-noise ratio in dB (scalar or element-wise array).

        Returns
        -------
        float or numpy.ndarray
            ``pd_fixed`` for ``noise_model="fixed"``, otherwise the blended
            :func:`albersheim_pd` curve at ``pfa`` and ``n_pulses``.  A scalar for
            scalar input, otherwise a ``float64`` array with the shape of ``snr_db``.
        """
        if self.noise_model == "fixed":
            if np.isscalar(snr_db):
                return self.pd_fixed
            return np.full(np.shape(snr_db), self.pd_fixed, dtype=np.float64)
        return albersheim_pd(snr_db, self.pfa, self.n_pulses, blend=True)

    def _pd_scalar(self, snr_db: float) -> float:
        """Scalar ``Pd(snr_db)`` for :meth:`observe`; matches :meth:`pd` / :func:`albersheim_pd`."""
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
        """Execute one dwell on ``action``.  Pure given ``u``; the env owns ``U``.

        Parameters
        ----------
        action : int
            Channel index to dwell on.
        truth_bit : int
            Ground-truth occupancy of ``action`` this step (1 occupied, 0 empty).
        snr_db : float
            SNR in dB of the emitter on ``action`` (``-inf`` for an empty cell); only
            used when ``truth_bit`` is 1.
        u : float
            Pre-drawn uniform noise ``U[a, t]`` in ``[0, 1)``; a detection is reported
            when ``u < p`` with ``p = Pd(snr_db)`` if occupied, else ``p = Pfa``.

        Returns
        -------
        Observation
            ``obs`` is 0 while blind; ``switched``, ``blind`` and ``blind_remaining``
            follow the blind rule in the class docstring.

        Notes
        -----
        A retune (``action != current_channel``) sets ``blind_left = tau_switch`` before the
        blind check, so a switch during blindness restarts the counter.  The first dwell
        after :meth:`reset` on the initial channel is not a switch.
        """
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
    """Debug / verification receiver: ``tau_switch = 0``, ``O_t == truth_bit``, ignores ``u``.

    Parameters
    ----------
    pfa : float, optional
        Nominal false-alarm probability stored on the instance (never used for
        detection). Default 1e-3.
    **_ : object
        Any other ``ReceiverModel`` keyword arguments are accepted and ignored so the
        class is a drop-in replacement in config-driven construction.

    Notes
    -----
    Constructed as ``ReceiverModel(tau_switch=0, pfa=pfa, noise_model="fixed", pd_fixed=1.0)``;
    :meth:`pd` is identically 1 and :meth:`observe` returns ``truth_bit`` unchanged.
    """

    def __init__(self, pfa: float = 1e-3, **_: object) -> None:
        super().__init__(tau_switch=0, pfa=pfa, noise_model="fixed", pd_fixed=1.0)

    def pd(self, snr_db: _ArrayLike) -> _ArrayLike:
        """Probability of detection, identically 1.

        Parameters
        ----------
        snr_db : float or numpy.ndarray
            Ignored except for its shape.

        Returns
        -------
        float or numpy.ndarray
            ``1.0`` for scalar input, otherwise an array of ones with the shape of ``snr_db``.
        """
        if np.isscalar(snr_db):
            return 1.0
        return np.ones(np.shape(snr_db), dtype=np.float64)

    def observe(self, action: int, truth_bit: int, snr_db: float, u: float) -> Observation:
        """Execute one dwell, reporting the truth bit directly.

        Parameters
        ----------
        action : int
            Channel index to dwell on.
        truth_bit : int
            Ground-truth occupancy of ``action`` this step (1 occupied, 0 empty).
        snr_db : float
            Ignored.
        u : float
            Ignored.

        Returns
        -------
        Observation
            ``obs == truth_bit``, ``switched`` as for :meth:`ReceiverModel.observe`,
            ``blind`` always False and ``blind_remaining`` always 0.
        """
        switched = self._prev is not None and action != self._prev
        self._prev = action
        return Observation(1 if truth_bit else 0, switched, False, 0)
