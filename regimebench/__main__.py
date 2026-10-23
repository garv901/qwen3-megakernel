"""``python -m regimebench run|report ...``"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regimebench")
    sub = p.add_subparsers(dest="cmd", required=True)
    from .run import add_run_arguments
    add_run_arguments(sub.add_parser("run", help="measure arms over workloads"))
    rp = sub.add_parser("report", help="records -> REPORT.md, summary.csv, plots")
    rp.add_argument("runs", nargs="+", help="run dirs or records.jsonl files (merged)")
    rp.add_argument("--out", required=True, help="output dir, e.g. results/<date>-b200/")
    a = p.parse_args(argv)
    if a.cmd == "run":
        from .run import main_run
        return main_run(a)
    from .report import main_report
    return main_report(a.runs, a.out)


if __name__ == "__main__":
    sys.exit(main())
