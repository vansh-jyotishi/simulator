"""Build the standalone dashboard page and open it in a browser.

    python -m dashboard.build            # export, build, open
    python -m dashboard.build --no-open  # export and build only

Inlines ``dashboard/runs.json`` into ``dashboard/template.html`` and writes
``dashboard/index.html``: a single self-contained file that needs no server.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template.html")
DATA = os.path.join(HERE, "runs.json")
OUT = os.path.join(HERE, "index.html")
PLACEHOLDER = "__RUNS__"


def build(refresh: bool = True) -> str:
    """Write ``dashboard/index.html`` and return its path.

    Parameters
    ----------
    refresh : bool, optional
        Re-run the exporter first so the page carries current mission data.

    Returns
    -------
    str
        Absolute path of the generated page.
    """
    if refresh or not os.path.exists(DATA):
        from dashboard.export_runs import main as export
        export()

    tpl = io.open(TEMPLATE, encoding="utf-8").read()
    if PLACEHOLDER not in tpl:
        raise RuntimeError(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    data = io.open(DATA, encoding="utf-8").read()
    if "</script" in data.lower():
        raise RuntimeError("runs.json contains a closing script tag; refusing to inline it")

    io.open(OUT, "w", encoding="utf-8").write(tpl.replace(PLACEHOLDER, data))
    return OUT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-open", action="store_true", help="build without launching a browser")
    ap.add_argument("--no-refresh", action="store_true", help="reuse the existing runs.json")
    args = ap.parse_args(argv)

    path = build(refresh=not args.no_refresh)
    size = os.path.getsize(path) // 1024
    print(f"\nbuilt {path} ({size} KB)")

    if args.no_open:
        print("open it in any browser to view the dashboard.")
        return 0

    url = "file:///" + path.replace(os.sep, "/")
    print(f"opening {url}")
    webbrowser.open(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
