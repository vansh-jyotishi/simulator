"""Unit tests for the receiver curves and the three emitter models (S10-S12)."""
from __future__ import annotations

import numpy as np
import pytest

from env.emitters import AgileJammer, PeriodicRadar, ScanningRadar, onoff_mask, render_truth
from env.receiver_model import albersheim_pd, albersheim_snr_db, sinc2_gain


def _runs(row):
    pad = np.concatenate(([0], row.astype(np.int8), [0]))
    d = np.diff(pad)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    return starts, ends  # ends exclusive


# ---------------------------------------------------------------- Albersheim / sinc


def test_albersheim_golden():
    assert abs(albersheim_snr_db(0.9, 1e-6) - 13.1) < 0.15


@pytest.mark.parametrize("pd", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("pfa", [1e-3, 1e-6])
def test_albersheim_round_trip(pd, pfa):
    snr = albersheim_snr_db(pd, pfa)
    assert abs(albersheim_pd(snr, pfa, blend=False) - pd) < 1e-9
    assert abs(albersheim_pd(snr, pfa, blend=True) - pd) < 1e-9


def test_albersheim_reference_values():
    assert abs(albersheim_pd(10, 1e-3) - 0.81) < 0.01
    assert abs(albersheim_pd(-5, 1e-3) - 0.032) < 0.005
    assert abs(albersheim_pd(0, 1e-3) - 0.10) < 0.01
    assert abs(albersheim_pd(5, 1e-3) - 0.21) < 0.01
    assert abs(albersheim_pd(13, 1e-3) - 0.996) < 0.002
    assert albersheim_pd(15, 1e-3) > 0.999
    assert albersheim_pd(-np.inf, 1e-3) == 1e-3


def test_albersheim_monotone_and_continuous():
    x = np.linspace(-15, 20, 71)
    y = albersheim_pd(x, 1e-3)
    assert np.all(np.diff(y) >= 0)
    assert y[0] >= 1e-3 and y[-1] <= 1.0
    lo = albersheim_snr_db(0.1, 1e-3)
    assert abs(albersheim_pd(lo - 1e-6, 1e-3) - albersheim_pd(lo + 1e-6, 1e-3)) < 1e-4
    assert isinstance(albersheim_pd(3.0, 1e-3), float)
    assert albersheim_pd(np.array([3.0, 4.0]), 1e-3).shape == (2,)


def test_sinc2_gain():
    assert sinc2_gain(0, 20) == 1.0
    assert abs(sinc2_gain(10, 20) - 0.5) < 1e-3
    assert abs(sinc2_gain(-10, 20) - 0.5) < 1e-3
    assert sinc2_gain(1.129 * 20, 20) < 1e-5  # first null
    assert sinc2_gain(np.array([0.0, 10.0]), 20).shape == (2,)


# ---------------------------------------------------------------- periodic


def test_periodic_closed_form_and_autocorrelation():
    p = PeriodicRadar(3, (0, 3, 17), 10.0)
    S, SNR = p.render(10, 2000)
    row = S[3].astype(float)
    assert row.sum() == 300
    assert S.sum() == 300 and np.all(SNR[3, S[3] == 1] == 10.0) and np.all(np.isneginf(SNR[S == 0]))
    c = row - row.mean()
    ac = np.correlate(c, c, "full")[1999:]
    assert np.argmax(ac[1:60]) + 1 == 20
    starts, ends = _runs(S[3])
    assert set((ends - starts)[:-1].tolist()) == {3}
    assert set((starts[1:] - ends[:-1]).tolist()) == {17}
    assert np.array_equal(onoff_mask(2000, 0, 3, 17), S[3].astype(bool))


def test_periodic_randomized_phase_shifts_start_within_pri():
    starts = set()
    for seed in range(20):
        p = PeriodicRadar(3, (5, 3, 17), 10.0, rng=np.random.default_rng(seed), randomize_phase=True)
        assert 5 <= p.start < 5 + 20
        starts.add(p.start)
    assert len(starts) > 5


# ---------------------------------------------------------------- agile


def test_agile_one_channel_per_active_slot_and_hop_boundaries():
    T = 4000
    j = AgileJammer((20, 21, 22, 23, 24, 25), 8, None, 0.2, (5, 40, 10), 9.0, rng=np.random.default_rng(3))
    S, SNR = j.render(50, T)
    col = S.sum(axis=0)
    assert set(col.tolist()) <= {0, 1}
    assert np.array_equal(col.astype(bool), onoff_mask(T, 5, 40, 10))
    active = np.nonzero(col)[0]
    chan = S.argmax(axis=0)
    # channel may only change at hop boundaries (compare consecutive active slots)
    prev = active[:-1]
    nxt = active[1:]
    consecutive = nxt == prev + 1
    changed = chan[nxt[consecutive]] != chan[prev[consecutive]]
    assert np.all(nxt[consecutive][changed] % 8 == 0)
    assert np.all(np.isin(chan[active], (20, 21, 22, 23, 24, 25)))


def test_agile_stay_fraction():
    j = AgileJammer((1, 2, 3, 4), 1, None, 0.5, (0, 5000, 1), 9.0, rng=np.random.default_rng(0))
    seq = j.hop_sequence(5001)
    stay = float(np.mean(seq[1:] == seq[:-1]))
    assert abs(stay - 0.5) < 0.03


def test_agile_occupancy_matches_hop_probs_uniform_and_stationary_general():
    j = AgileJammer((1, 2, 3, 4, 5), 1, None, 0.0, (0, 49999, 1), 9.0, rng=np.random.default_rng(1))
    seq = j.hop_sequence(50000)
    occ = np.bincount(seq, minlength=5) / len(seq)
    assert np.all(np.abs(occ - 0.2) < 0.03)
    probs = (0.4, 0.3, 0.15, 0.1, 0.05)
    j2 = AgileJammer((1, 2, 3, 4, 5), 1, probs, 0.0, (0, 49999, 1), 9.0, rng=np.random.default_rng(2))
    seq2 = j2.hop_sequence(50000)
    occ2 = np.bincount(seq2, minlength=5) / len(seq2)
    assert np.all(np.abs(occ2 - j2.stationary_distribution()) < 0.03)
    assert np.all(seq2[1:] != seq2[:-1])  # p_stay=0 never repeats a channel


def test_agile_transition_matrix_stretch():
    j = AgileJammer((1, 2), 1, None, 0.0, (0, 9, 1), 9.0, rng=np.random.default_rng(4),
                    transition_matrix=((0.9, 0.1), (0.1, 0.9)))
    seq = j.hop_sequence(20000)
    assert abs(float(np.mean(seq[1:] == seq[:-1])) - 0.9) < 0.03


def test_agile_same_rng_same_sequence_and_randomized_start():
    a = AgileJammer((1, 2, 3), 4, None, 0.2, (0, 10, 5), 9.0, rng=np.random.default_rng(7))
    b = AgileJammer((1, 2, 3), 4, None, 0.2, (0, 10, 5), 9.0, rng=np.random.default_rng(7))
    assert np.array_equal(a.render(5, 500)[0], b.render(5, 500)[0])
    starts = {AgileJammer((1, 2, 3), 4, None, 0.2, (0, 10, 5), 9.0, rng=np.random.default_rng(s),
                          randomize_start=True).hop_sequence(100)[0] for s in range(30)}
    assert len(starts) == 3


# ---------------------------------------------------------------- scanning


def test_lighthouse_bursts():
    r = ScanningRadar(41, 130, 20.0, 15.0, phase0_deg=180.0, presence_gain_db=-10.0)
    S, SNR = r.render(50, 5000)
    assert S.sum() == S[41].sum()  # only its own channel
    starts, ends = _runs(S[41])
    assert set(np.diff(starts).tolist()) == {130}
    if ends[-1] == 5000:  # last burst truncated by the episode end
        starts, ends = starts[:-1], ends[:-1]
    lengths = ends - starts
    assert np.all(np.abs(lengths - 12) <= 1)
    for s, e in zip(starts, ends):
        assert abs(SNR[41, s:e].max() - 15.0) < 0.05
    assert np.all(SNR[41, S[41] == 1] >= 15.0 - 10.0 - 1e-4)
    # no cell outside the mainlobe: -10 dB edge is inside the first null
    theta = (360.0 * np.arange(5000) / 130 + 180.0) % 360.0
    delta = np.where(theta > 180, theta - 360, theta)
    assert np.all(np.abs(delta[S[41] == 1]) < 1.129 * 20)
    # first sidelobe (-13.3 dB) never registers even at a -13 dB threshold
    r2 = ScanningRadar(41, 130, 20.0, 15.0, phase0_deg=180.0, presence_gain_db=-13.0)
    s2, _ = _runs(r2.render(50, 5000)[0][41])
    assert set(np.diff(s2).tolist()) == {130}


def test_scanning_randomize_phase_differs_by_seed():
    a = ScanningRadar(41, 130, 20.0, 15.0, rng=np.random.default_rng(0), randomize_phase=True)
    b = ScanningRadar(41, 130, 20.0, 15.0, rng=np.random.default_rng(1), randomize_phase=True)
    assert a.phase0_deg != b.phase0_deg
    assert not np.array_equal(a.render(50, 1000)[0], b.render(50, 1000)[0])


# ---------------------------------------------------------------- overlap / truth


def _overlap_scenario():
    return {
        "name": "overlap", "N": 8, "T": 600,
        "receiver": {"tau_switch": 1, "pfa": 1e-3},
        "emitters": [
            {"type": "periodic", "name": "radar", "channel": 3, "onoff": [0, 4, 6], "snr_db": 12.0},
            {"type": "agile", "name": "jam", "channels": [2, 3, 4], "hop_period": 3, "p_stay": 0.3,
             "onoff": [0, 598, 1], "snr_db": 6.0},
        ],
    }


def test_overlap_or_max_argmax():
    tr = render_truth(_overlap_scenario(), 0)
    S_by, SNR_by = tr.S_by_emitter, None
    assert np.array_equal(tr.S, S_by.any(axis=0).astype(np.int8))
    both = (S_by[0] == 1) & (S_by[1] == 1)
    assert both.sum() > 0, "the jammer must sometimes sit on the radar channel"
    assert np.all(tr.SNR[both] == 12.0) and np.all(tr.E[both] == 0)
    only_jam = (S_by[1] == 1) & (S_by[0] == 0)
    assert np.all(tr.SNR[only_jam] == 6.0) and np.all(tr.E[only_jam] == 1)
    assert np.all(tr.E[tr.S == 0] == -1)
    assert tr.emitter_types == ("periodic", "agile")


def test_render_truth_seed_layout():
    a = render_truth(_overlap_scenario(), 0)
    b = render_truth(_overlap_scenario(), 0)
    c = render_truth(_overlap_scenario(), 1)
    assert np.array_equal(a.S, b.S) and np.array_equal(a.U, b.U)
    assert not np.array_equal(a.U, c.U)
    U = np.random.default_rng([0, 10_000]).random((8, 600), dtype=np.float32)
    assert np.array_equal(a.U, U)


def test_render_truth_no_emitters():
    tr = render_truth({"N": 3, "T": 10, "receiver": {"tau_switch": 0, "pfa": 0.1}, "emitters": []}, 0)
    assert tr.S.sum() == 0 and tr.S_by_emitter.shape == (0, 3, 10) and np.all(tr.E == -1)
