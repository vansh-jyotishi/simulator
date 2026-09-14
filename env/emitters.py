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
    """Closed-form duty-cycle mask over ``T`` slots.

    ``active[t] = t >= start and ((t - start) % (on + off)) < on``.  The
    ``onoff = [start, on, off]`` convention follows rfrl-gym (see the module docstring).

    Parameters
    ----------
    T : int
        Number of time slots; the returned mask has this length.
    start : int
        First slot at which the emitter may be active; every earlier slot is off.
    on : int
        Number of consecutive active slots in each period.
    off : int
        Number of consecutive inactive slots following each ``on`` run.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(T,)``; ``True`` where the emitter is active.
    """
    t = np.arange(T, dtype=np.int64)
    return (t >= start) & (((t - start) % (on + off)) < on)


class Emitter:
    """Base class for open-loop emitters.

    Subclasses implement :meth:`render`, whose contract is
    ``render(N, T) -> (S_m int8 (N,T), SNR_m float32 (N,T))``: ``S_m`` is 1 where the
    emitter is present and ``SNR_m`` holds its per-cell SNR in dB (``-inf`` where absent).

    Parameters
    ----------
    name : str
        Human-readable emitter name, surfaced in ``Truth.emitter_names``.
    rng : numpy.random.Generator
        Random generator owned by this emitter; used for any phase / start-channel randomisation.

    Attributes
    ----------
    type : str
        Emitter type tag (``"base"`` here, overridden per subclass), surfaced in ``Truth.emitter_types``.
    name : str
        As passed to the constructor.
    rng : numpy.random.Generator
        As passed to the constructor.
    """

    type: str = "base"

    def __init__(self, name: str, rng: np.random.Generator) -> None:
        self.name = name
        self.rng = rng

    def render(self, N: int, T: int) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - abstract
        """Render this emitter's presence and SNR over the full ``(N, T)`` grid.

        Parameters
        ----------
        N : int
            Number of channels.
        T : int
            Number of time slots.

        Returns
        -------
        S_m : numpy.ndarray
            ``int8`` array of shape ``(N, T)``; 1 where this emitter is present, else 0.
        SNR_m : numpy.ndarray
            ``float32`` array of shape ``(N, T)``; SNR in dB where present, ``-inf`` elsewhere.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override.
        """
        raise NotImplementedError


class PeriodicRadar(Emitter):
    """Fixed-channel pulsed radar.

    On for ``on`` slots, off for ``off`` slots, starting at ``start``; the pulse
    repetition interval is ``PRI = on + off``.  Presence follows :func:`onoff_mask` on a
    single channel with a constant SNR.

    Parameters
    ----------
    channel : int
        Channel index the radar occupies.
    onoff : tuple[int, int, int]
        ``(start, on, off)`` duty-cycle triple (see :func:`onoff_mask`).
    snr_db : float
        SNR in dB written to every active cell.
    name : str, optional
        Emitter name.  Default ``"periodic"``.
    rng : numpy.random.Generator or None, optional
        Random generator; ``default_rng(0)`` when ``None``.  Only consumed when ``randomize_phase`` is set.
    randomize_phase : bool, optional
        If ``True``, offset ``start`` by a uniform draw in ``[0, PRI)`` so the pulse phase is random.
        Default ``False``.

    Attributes
    ----------
    channel : int
        Occupied channel.
    start, on, off : int
        Duty-cycle parameters after any phase randomisation.
    pri : int
        Pulse repetition interval ``on + off``.
    snr_db : float
        Constant SNR in dB.
    """

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
        """Render the pulsed on/off pattern on ``channel``.

        Parameters
        ----------
        N : int
            Number of channels.
        T : int
            Number of time slots.

        Returns
        -------
        S_m : numpy.ndarray
            ``int8`` array of shape ``(N, T)``; 1 on ``channel`` in active slots, else 0.
        SNR_m : numpy.ndarray
            ``float32`` array of shape ``(N, T)``; ``snr_db`` where present, ``-inf`` elsewhere.
        """
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

    Parameters
    ----------
    channels : tuple[int, ...]
        Channel indices the jammer hops over; hop-sequence indices refer to positions in this tuple.
    hop_period : int
        Number of slots between successive hop decisions.
    hop_probs : tuple[float, ...] or None
        Unnormalised weight per entry of ``channels``; uniform when ``None``.  Also the distribution
        of the initial channel when ``randomize_start`` is set.
    p_stay : float
        Probability of staying on the current channel at each hop.
    onoff : tuple[int, int, int]
        ``(start, on, off)`` duty-cycle triple (see :func:`onoff_mask`).
    snr_db : float
        SNR in dB written to every active cell.
    name : str, optional
        Emitter name.  Default ``"agile"``.
    rng : numpy.random.Generator or None, optional
        Random generator used for the hop sequence; ``default_rng(0)`` when ``None``.
    randomize_start : bool, optional
        If ``True``, draw the initial channel from ``hop_probs``; otherwise start on ``channels[0]``.
        Default ``False``.
    transition_matrix : tuple[tuple[float, ...], ...] or None, optional
        Explicit row-stochastic hop matrix over ``channels`` (rows are renormalised).  When given it
        overrides the ``p_stay`` / ``hop_probs`` construction.  Default ``None``.

    Attributes
    ----------
    channels : tuple[int, ...]
        Hop set as integers.
    hop_period : int
        Slots per hop.
    p_stay : float
        Stay probability.
    start, on, off : int
        Duty-cycle parameters.
    snr_db : float
        Constant SNR in dB.
    hop_probs : numpy.ndarray
        Normalised hop weights of shape ``(K,)`` where ``K = len(channels)``.
    P : numpy.ndarray
        Row-stochastic hop transition matrix of shape ``(K, K)``.
    randomize_start : bool
        Whether the initial channel is drawn at random.
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
        """Analytic long-run channel occupancy of the hop chain.

        Computed as the left eigenvector of the hop transition matrix ``P`` with eigenvalue 1,
        normalised to sum to one.

        Returns
        -------
        numpy.ndarray
            Occupancy probability of shape ``(K,)``, aligned with ``channels``.
        """
        w, v = np.linalg.eig(self.P.T)
        i = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(v[:, i])
        pi = pi / pi.sum()
        return pi

    def hop_sequence(self, T: int) -> np.ndarray:
        """Draw the channel index (into ``channels``) for each hop.

        The initial index is drawn from ``hop_probs`` when ``randomize_start`` is set, else 0; every
        subsequent index is sampled from the matching row of ``P`` using ``self.rng``.

        Parameters
        ----------
        T : int
            Number of time slots to cover.

        Returns
        -------
        numpy.ndarray
            ``int64`` array of length ``max(1, ceil(T / hop_period))`` holding positions into ``channels``.
        """
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
        """Render the hop pattern gated by the on/off duty cycle.

        Parameters
        ----------
        N : int
            Number of channels.
        T : int
            Number of time slots.

        Returns
        -------
        S_m : numpy.ndarray
            ``int8`` array of shape ``(N, T)``; exactly one channel is 1 in each active slot, else 0.
        SNR_m : numpy.ndarray
            ``float32`` array of shape ``(N, T)``; ``snr_db`` where present, ``-inf`` elsewhere.
        """
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

    Parameters
    ----------
    channel : int
        Channel index the radar occupies.
    T_rot : int
        Rotation period in slots.
    beamwidth_deg : float
        3-dB beamwidth ``bw`` of the antenna pattern in degrees.
    snr_peak_db : float
        Boresight SNR in dB.
    phase0_deg : float, optional
        Initial azimuth offset in degrees.  Default ``0.0``.
    presence_gain_db : float, optional
        Relative gain (dB below peak) at or above which the radar counts as present.  Default ``-10.0``.
    mode : str, optional
        Scan mode; only ``"lighthouse"`` is supported.  Default ``"lighthouse"``.
    name : str, optional
        Emitter name.  Default ``"scanning"``.
    rng : numpy.random.Generator or None, optional
        Random generator; ``default_rng(0)`` when ``None``.  Only consumed when ``randomize_phase`` is set.
    randomize_phase : bool, optional
        If ``True``, add a uniform draw in ``[0, 360)`` degrees to ``phase0_deg``.  Default ``False``.

    Attributes
    ----------
    mode : str
        Scan mode.
    channel : int
        Occupied channel.
    T_rot : int
        Rotation period in slots.
    beamwidth_deg : float
        3-dB beamwidth in degrees.
    snr_peak_db : float
        Boresight SNR in dB.
    phase0_deg : float
        Initial azimuth in degrees after any phase randomisation.
    presence_gain_db : float
        Presence threshold relative to peak, in dB.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"lighthouse"``.
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
        """Render the rotating mainlobe sweep on ``channel``.

        Parameters
        ----------
        N : int
            Number of channels.
        T : int
            Number of time slots.

        Returns
        -------
        S_m : numpy.ndarray
            ``int8`` array of shape ``(N, T)``; 1 on ``channel`` while the mainlobe illuminates the
            receiver at or above ``presence_gain_db``, else 0.
        SNR_m : numpy.ndarray
            ``float32`` array of shape ``(N, T)``; ``snr_peak_db + 10*log10(gain)`` where present,
            ``-inf`` elsewhere.
        """
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
    """Instantiate every emitter in ``cfg``.

    Emitter ``i`` uses ``default_rng([seed, i])`` so each emitter's randomisation is independent of
    the others and of the scenario's noise stream.  The ``cfg.randomize`` flags decide whether the
    periodic phase, agile start channel and scanning phase are randomised.

    Parameters
    ----------
    cfg : ScenarioConfig
        Validated scenario; ``cfg.emitters`` is a sequence of ``PeriodicCfg`` / ``AgileCfg`` /
        ``ScanningCfg`` and ``cfg.randomize`` holds the per-type randomisation flags.
    seed : int
        Episode seed.

    Returns
    -------
    list[Emitter]
        Emitter instances in the same order as ``cfg.emitters``.

    Raises
    ------
    TypeError
        If an entry of ``cfg.emitters`` is not one of the known config types (unreachable after
        scenario validation).
    """
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

    Parameters
    ----------
    cfg : ScenarioConfig or dict or str
        Scenario config, raw mapping or path; resolved with :func:`env.scenario.load_scenario`.
    seed : int
        Episode seed; drives both the emitters (``default_rng([seed, i])``) and the noise stream ``U``.

    Returns
    -------
    Truth
        Frozen truth with read-only arrays: ``S`` (``int8 (N,T)``, OR over emitters), ``S_by_emitter``
        (``int8 (M,N,T)``), ``SNR`` (``float32 (N,T)`` dB, ``-inf`` where ``S == 0``), ``E``
        (``int16 (N,T)`` emitter id with max SNR, ``-1`` where ``S == 0``), ``U`` (``float32 (N,T)``
        Uniform(0,1)), plus the ``emitter_names`` and ``emitter_types`` tuples.  With no emitters
        ``S`` / ``SNR`` / ``E`` are all-off.
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
