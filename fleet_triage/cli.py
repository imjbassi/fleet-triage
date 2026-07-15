"""Command line interface.

Subcommands:
    generate    Produce a synthetic fleet log (JSON Lines)
    triage      Full fleet triage summary from a log file
    report      RCA-style report for one robot and fault code
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import __version__
from .analysis import find_recurring_faults
from .generator import generate_fleet_log, write_jsonl
from .parser import parse_file
from .report import build_rca_report, triage_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fleet-triage",
        description="Synthetic robot fleet log analysis and RCA tooling",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a synthetic fleet log")
    g.add_argument("--out", default="fleet_log.jsonl", help="output path")
    g.add_argument("--robots", type=int, default=10)
    g.add_argument("--hours", type=float, default=72.0)
    g.add_argument("--heartbeat-seconds", type=int, default=300)
    g.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("triage", help="fleet triage summary")
    t.add_argument("logfile", help="JSON Lines log file")
    t.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="occurrences before a fault counts as recurring (default 3)",
    )
    t.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="rolling window for recurrence detection (default 24)",
    )

    r = sub.add_parser("report", help="RCA report for one robot and fault")
    r.add_argument("logfile", help="JSON Lines log file")
    r.add_argument("--robot", help="robot id, e.g. R-03")
    r.add_argument("--fault", help="fault code, e.g. JOINT_OVERCURRENT")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        events = generate_fleet_log(
            robots=args.robots,
            hours=args.hours,
            heartbeat_seconds=args.heartbeat_seconds,
            seed=args.seed,
        )
        with open(args.out, "w", encoding="utf-8") as fp:
            n = write_jsonl(events, fp)
        print(f"Wrote {n} events for {args.robots} robots over {args.hours}h to {args.out}")
        return 0

    parsed = parse_file(args.logfile)
    if not parsed.events:
        print("No parseable events found in the log file.", file=sys.stderr)
        for err in parsed.errors[:5]:
            print(f"  {err}", file=sys.stderr)
        return 1

    if args.command == "triage":
        print(triage_summary(parsed, args.threshold, args.window_hours))
        return 0

    if args.command == "report":
        robot, fault = args.robot, args.fault
        if not robot or not fault:
            # Default to the worst recurring fault so the command is
            # useful with no arguments beyond the log file.
            recurring = find_recurring_faults(parsed.events)
            if recurring:
                robot = robot or recurring[0].robot_id
                fault = fault or recurring[0].fault_code
            else:
                counts = Counter(
                    (e.robot_id, e.fault_code)
                    for e in parsed.events
                    if e.event == "fault" and e.fault_code
                )
                if not counts:
                    print("No fault events in the log file.", file=sys.stderr)
                    return 1
                (robot_top, fault_top), _ = counts.most_common(1)[0]
                robot = robot or robot_top
                fault = fault or fault_top
            print(f"No robot or fault specified; using {robot} / {fault}\n")
        print(build_rca_report(parsed.events, robot, fault))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
