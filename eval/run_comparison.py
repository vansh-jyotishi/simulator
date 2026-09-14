"""The single side-by-side comparison table (contract C9).

    python -m eval.run_comparison --scenarios scenarios/train_default.json scenarios/heldout_mix.json \
        --seeds 0-9 --heldout-seeds 100-109 --schedulers round_robin weighted_priority clairvoyant \
        [--extra whittle=schedulers.whittle_rmab:WhittleScheduler] \
        [--scheduler-kwargs '{"whittle": {"beta": 0.95}}'] [--episodes 1] [--save-logs] [--snr-sweep] \
        [--out results/comparison.csv] [--md results/comparison.md]

Every scheduler runs every scenario on both seed lists (asserted disjoint).
A row is ``(scheduler, scenario, seed_split)``; ``scenario_split`` is
``heldout`` iff the file name starts with ``heldout_``.  Cells are
``mean ± std`` over seeds (and episodes).  Per-(scheduler, scenario, seed,
episode) rows go to the CSV.  The header prints the git SHA, ``sha256[:8]``
of every scenario file, the seed lists and the numpy / python versions so a
table can be reproduced bit-for-bit.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from env.spectrum_env import make_env  # the only env import allowed outside env/ (INTERFACE.md)
from eval.runner import BUILTIN_TRUTH_CLASSES, discover_registry, load_extra, run_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_scenario_path(path: str | Path) -> Path:
    """Relative paths resolve against the cwd first, then the repo root (mirrors the scenario loader)."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    alt = REPO_ROOT / p
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"scenario file not found: {path} (tried cwd and {REPO_ROOT})")


def scenario_name(path: str | Path) -> str:
    """Scenario ``name`` field, obtained through ``make_env`` so eval never imports env internals."""
    return str(make_env(path, 0).cfg.name)

TABLE_COLUMNS = [
    ("poi", "POI", "{:.3f}"),
    ("poi_time", "POI_time", "{:.3f}"),
    ("sensor_pd", "Pd_sensor", "{:.3f}"),
    ("effective_pd", "Pd_eff", "{:.3f}"),
    ("sensor_pfa", "Pfa", "{:.4f}"),
    ("mean_tti", "TTI_mean", "{:.2f}"),
    ("median_tti", "TTI_med", "{:.1f}"),
    ("intercept_rate_per_s", "int/s", "{:.1f}"),
    ("correct_pred_pct", "pred%", "{:.1f}"),
    ("avg_pred_time_error", "pred_err", "{:.2f}"),
    ("total_reward", "reward", "{:.0f}"),
    ("blind_pct", "blind%", "{:.1f}"),
    ("n_switches", "switches", "{:.0f}"),
    ("us_per_decision_mean", "us_mean", "{:.1f}"),
    ("us_per_decision_p99", "us_p99", "{:.1f}"),
]


# ----------------------------------------------------------------------------- helpers


def parse_seeds(text: str | None) -> list[int]:
    """``"0-9"`` -> 0..9, ``"1,3,5"`` -> [1,3,5], mixes allowed."""
    if not text:
        return []
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


def fmt_cell(mean: float, std: float, fmt: str) -> str:
    if np.isnan(mean):
        return "nan"
    if np.isnan(std) or std == 0:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def build_schedulers(names: list[str], extras: list[str], kwargs_json: str | None) -> dict[str, tuple[type, dict]]:
    reg = discover_registry()
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    out: dict[str, tuple[type, dict]] = {}
    for n in names:
        if n not in reg:
            raise SystemExit(f"unknown scheduler {n!r}; available: {sorted(reg)}")
        out[n] = (reg[n], dict(kwargs.get(n, {})))
    for spec in extras or []:
        try:
            alias, cls = load_extra(spec)
        except (ValueError, ImportError, AttributeError) as e:
            raise SystemExit(f"--extra {spec!r} refused: {e}") from e
        if alias in out:
            raise SystemExit(f"--extra alias {alias!r} collides with an existing label")
        out[alias] = (cls, dict(kwargs.get(alias, {})))
    for label in kwargs:
        if label not in out:
            print(f"warning: --scheduler-kwargs for unknown label {label!r} ignored", file=sys.stderr)
    return out


def save_log(res, out_dir: Path) -> Path:
    env = res.env
    truth, hist = env.get_truth(), env.get_history()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{res.scenario}__{res.label}__seed{res.seed}_ep{res.episode}.npz"
    np.savez_compressed(p, S=truth.S, S_by_emitter=truth.S_by_emitter, E=truth.E, SNR=truth.SNR,
                        actions=hist.actions, obs=hist.obs, blind=hist.blind, switched=hist.switched,
                        new_detect=hist.new_detect, rewards=hist.rewards, seed=res.seed,
                        scenario=res.scenario, scheduler=res.label)
    return p


# ----------------------------------------------------------------------------- core


def run_grid(scenarios: list[str], seeds: list[int], heldout_seeds: list[int], schedulers: dict,
             episodes: int = 1, save_logs: bool = False, log_dir: Path | None = None,
             delta_guard: int = 1, verbose: bool = True) -> pd.DataFrame:
    rows = []
    splits = [("train", seeds), ("heldout", heldout_seeds)]
    for sc in scenarios:
        sc_path = resolve_scenario_path(sc)
        sc_name = scenario_name(sc_path)
        sc_split = "heldout" if sc_path.name.startswith("heldout_") else "train"
        for label, (cls, kw) in schedulers.items():
            for split, seed_list in splits:
                for seed in seed_list:
                    for ep in range(episodes):
                        try:
                            sched = cls(**kw)
                            res = run_episode(sc_path, seed, sched, label=label, episode=ep,
                                              keep_env=save_logs, delta_guard=delta_guard,
                                              scheduler_seed=seed + ep * 1_000_000)
                        except Exception as e:  # noqa: BLE001
                            raise SystemExit(f"FAILED scheduler={label!r} scenario={sc_name!r} seed={seed} "
                                             f"episode={ep}: {type(e).__name__}: {e}") from e
                        row = res.row()
                        row.update({"scenario": sc_name, "scenario_file": str(sc_path), "seed_split": split,
                                    "scenario_split": sc_split, "T": res.spec.T, "N": res.spec.N,
                                    "tau_switch": res.spec.tau_switch})
                        if save_logs:
                            row["log"] = str(save_log(res, log_dir or (REPO_ROOT / "results" / "logs")))
                        rows.append(row)
                        if verbose:
                            m = res.metrics
                            print(f"  {label:<18} {sc_name:<22} {split:<7} seed={seed:<4} ep={ep} "
                                  f"poi={m.poi:.3f} pd={m.sensor_pd:.3f} pfa={m.sensor_pfa:.4f} "
                                  f"blind={100 * m.blind_fraction:.0f}% reward={m.total_reward:.0f}",
                                  file=sys.stderr)
    df = pd.DataFrame(rows)
    if len(df):
        df["blind_pct"] = 100.0 * df["blind_fraction"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (scheduler, scenario, seed_split) into ``mean ± std`` strings."""
    if df.empty:
        return df
    keys = ["scheduler", "scenario", "scenario_split", "seed_split"]
    g = df.groupby(keys, sort=False)
    out = []
    for name, sub in g:
        row = dict(zip(keys, name))
        row["n"] = len(sub)
        for col, head, fmt in TABLE_COLUMNS:
            vals = sub[col].astype(float)
            row[head] = fmt_cell(float(np.nanmean(vals)) if vals.notna().any() else float("nan"),
                                 float(np.nanstd(vals)) if vals.notna().sum() > 1 else float("nan"), fmt)
        out.append(row)
    return pd.DataFrame(out)


def header_lines(scenarios: list[str], seeds: list[int], heldout: list[int], schedulers: dict) -> list[str]:
    lines = [
        f"git {git_sha()} | python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__}",
        f"train seeds: {seeds} | heldout seeds: {heldout}",
        "schedulers: " + ", ".join(f"{k}={v[0].__module__}.{v[0].__name__}"
                                   + (" [CHEATING]" if issubclass(v[0], BUILTIN_TRUTH_CLASSES) else "")
                                   for k, v in schedulers.items()),
    ]
    for sc in scenarios:
        p = resolve_scenario_path(sc)
        lines.append(f"scenario {p.name}: sha256[:8]={file_sha(p)}")
    return lines


def to_markdown(summary: pd.DataFrame, header: list[str]) -> str:
    cols = list(summary.columns)
    lines = ["# Comparison table", ""] + [f"- {h}" for h in header] + [""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in summary.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("Cells are mean ± std over seeds. `clairvoyant` reads the truth (upper bound, CHEATING). "
                 "Pd_sensor excludes blind dwells, Pd_eff counts them as misses. Channels/slots are 0-based.")
    return "\n".join(lines)


def snr_sweep(seeds: list[int], schedulers: dict, episodes: int = 1) -> pd.DataFrame:
    files = sorted(glob.glob(str(REPO_ROOT / "scenarios" / "snr_sweep_*.json")),
                   key=lambda p: float(Path(p).stem.split("snr_sweep_")[1]))
    if not files:
        raise SystemExit("no scenarios/snr_sweep_*.json files found")
    df = run_grid(files, seeds, [], schedulers, episodes=episodes, verbose=False)
    df["snr_db"] = df["scenario_file"].map(lambda p: float(Path(p).stem.split("snr_sweep_")[1]))
    tab = df.groupby(["scheduler", "snr_db"], sort=True).agg(
        sensor_pd=("sensor_pd", "mean"), sensor_pd_std=("sensor_pd", "std"),
        sensor_pfa=("sensor_pfa", "mean"), poi=("poi", "mean"), n=("seed", "count")).reset_index()
    return tab


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=[], help="scenario JSON files")
    ap.add_argument("--seeds", default="0-2", help="train seeds, e.g. 0-9 or 0,1,2")
    ap.add_argument("--heldout-seeds", default=None, help="held-out seeds, must be disjoint from --seeds")
    ap.add_argument("--schedulers", nargs="*", default=["round_robin", "weighted_priority"])
    ap.add_argument("--extra", nargs="*", default=[], help="alias=pkg.mod:Class (needs_truth=True refused)")
    ap.add_argument("--scheduler-kwargs", default=None, help='JSON: {"label": {"kw": value}}')
    ap.add_argument("--episodes", type=int, default=1,
                    help="repeats per seed; episode e>0 offsets the scheduler seed by e*1_000_000 (env seed fixed)")
    ap.add_argument("--save-logs", action="store_true", help="write results/logs/*.npz per (scenario, scheduler, seed)")
    ap.add_argument("--snr-sweep", action="store_true", help="run scenarios/snr_sweep_*.json and print Pd vs SNR")
    ap.add_argument("--delta-guard", type=int, default=1)
    ap.add_argument("--out", default=None, help="CSV of per-episode rows")
    ap.add_argument("--md", default=None, help="markdown pipe table output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    heldout = parse_seeds(args.heldout_seeds)
    if set(seeds) & set(heldout):
        raise SystemExit(f"--seeds and --heldout-seeds must be disjoint; overlap: {sorted(set(seeds) & set(heldout))}")
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    schedulers = build_schedulers(args.schedulers, args.extra, args.scheduler_kwargs)

    if args.snr_sweep:
        tab = snr_sweep(seeds, schedulers, args.episodes)
        print("\n".join(header_lines([], seeds, [], schedulers)))
        print("\nSensor Pd vs SNR (StayPut-equivalent: single always-on radar, tau_switch=0)")
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            tab.to_csv(args.out, index=False)
        if not args.scenarios:
            return 0

    if not args.scenarios:
        raise SystemExit("--scenarios is required (or use --snr-sweep)")
    header = header_lines(args.scenarios, seeds, heldout, schedulers)
    print("\n".join(header))
    df = run_grid(args.scenarios, seeds, heldout, schedulers, episodes=args.episodes, save_logs=args.save_logs,
                  delta_guard=args.delta_guard, verbose=not args.quiet)
    summary = summarize(df)
    print()
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 30):
        print(summary.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out} ({len(df)} rows)")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(to_markdown(summary, header), encoding="utf-8")
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
