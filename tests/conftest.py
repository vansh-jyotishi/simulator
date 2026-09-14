"""pytest options: ``--scheduler pkg.mod:Class`` (repeatable) for the protocol conformance test."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--scheduler", action="append", default=None,
        help="scheduler to check, as pkg.mod:Class (repeatable). Default: both baselines.",
    )


def pytest_generate_tests(metafunc):
    if "scheduler_spec" in metafunc.fixturenames:
        specs = metafunc.config.getoption("--scheduler") or [
            "baselines.round_robin:RoundRobin",
            "baselines.weighted_priority:WeightedPriority",
        ]
        metafunc.parametrize("scheduler_spec", specs, ids=[s.split(":")[-1] for s in specs])
