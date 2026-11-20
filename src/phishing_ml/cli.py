"""CLI entry point.

    python -m phishing_ml.cli fetch-external        # download Hassan + JPCERT external sets
    python -m phishing_ml.cli run [--mode quick]    # full experiment pipeline -> outputs/
"""
from __future__ import annotations

import argparse
import sys

from phishing_ml.download import fetch_external
from phishing_ml.pipeline import ROOT, run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="phishing_ml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="audit -> val selection -> test (once) -> external -> PCA/cluster -> reports")
    p.add_argument("--mode", default="full", choices=["quick", "full"], help="quick skips PCA/clustering")
    sub.add_parser("fetch-external", help="download Hassan + JPCERT external evaluation sets into data/external/")
    args = ap.parse_args()
    if args.cmd == "fetch-external":
        sys.exit(fetch_external(ROOT / "data" / "external"))
    sys.exit(run_pipeline(mode=args.mode))


if __name__ == "__main__":
    main()
