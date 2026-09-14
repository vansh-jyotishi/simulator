"""Print the ASCII spectrum waterfall for one mission (demo / smoke view).

    python -m eval.show_waterfall
    python -m eval.show_waterfall --scenario scenarios/train_default.json --scheduler clairvoyant --slots 600

Rows are channels, columns are time slots.  ``#`` marks a cell where a
transmitter was genuinely active; the receiver's track is overlaid as ``@``
true positive, ``x`` blind dwell after a retune, ``?`` false alarm, ``m``
missed a real signal, ``o`` listened to an empty channel.
"""
from __future__ import annotations

import argparse
import sys

from common.protocol import sanitize_info
from env.spectrum_env import make_env
from eval.runner import discover_registry
from oracle.metrics_engine import compute_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="scenarios/train_default.json")
    ap.add_argument("--scheduler", default="round_robin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=240, help="how many slots to run and display")
    args = ap.parse_args(argv)

    reg = discover_registry()
    if args.scheduler not in reg:
        print(f"unknown scheduler {args.scheduler!r}; available: {sorted(reg)}", file=sys.stderr)
        return 2

    env = make_env(args.scenario, args.seed)
    obs, info = env.reset()
    sched = reg[args.scheduler]()
    if getattr(sched, "needs_truth", False):
        sched.set_truth(env.get_truth())
    sched.reset(env.mission_spec, args.seed)

    n = min(args.slots, env.mission_spec.T)
    for _ in range(n):
        obs, r, term, trunc, info = env.step(sched.select_action(obs, sanitize_info(info)))
        if trunc:
            break

    print()
    print(env.render(width=n))
    print()
    print("  legend:  # transmitter active   @ intercepted   x blind (retuning)   "
          "? false alarm   m missed   o empty")
    print()

    m = compute_metrics(env.get_truth(), env.get_history(), dwell_s=env.mission_spec.dwell_s)
    print(f"  scheduler        {args.scheduler}")
    print(f"  slots shown      {n} of {env.mission_spec.T}")
    print(f"  intercepted      {m.n_caught} of {m.n_bursts} transmissions so far")
    print(f"  blind            {100 * m.blind_fraction:.0f}% of dwells")
    print(f"  retunes          {m.n_switches}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
