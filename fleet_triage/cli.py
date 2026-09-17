"""Command line interface.

Subcommands:
    generate    Produce a synthetic fleet log (JSON Lines)
    triage      Full fleet triage summary from a log file
    report      RCA-style report for one robot and fault code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
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
    t.add_argument(
        "--min-robots",
        type=int,
        default=3,
        help="robots that must share a fault burst to flag an anomaly (default 3)",
    )
    t.add_argument(
        "--spike-ratio",
        type=float,
        default=3.0,
        help="burst size relative to baseline to flag an anomaly (default 3.0)",
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
        try:
            events = list(events)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            n = write_jsonl(events, fp)
        print(f"Wrote {n} events for {args.robots} robots over {args.hours}h to {args.out}")
        return 0

    try:
        parsed = parse_file(args.logfile)
    except OSError as exc:
        print(f"error: cannot read {args.logfile}: {exc}", file=sys.stderr)
        return 1
    if not parsed.events:
        print("No parseable events found in the log file.", file=sys.stderr)
        for err in parsed.errors[:5]:
            print(f"  {err}", file=sys.stderr)
        return 1

    if args.command == "triage":
        print(
            triage_summary(
                parsed,
                args.threshold,
                args.window_hours,
                args.min_robots,
                args.spike_ratio,
            )
        )
        return 0

    if args.command == "report":
        robot, fault = args.robot, args.fault
        if not robot or not fault:
            # Default to the worst recurring fault so the command is
            # useful with no arguments beyond the log file.
            # Only consider candidates matching whatever was specified, so
            # --robot R-01 never yields a fault picked from another robot.
            def matches(r_id: str, code: str) -> bool:
                return (not robot or r_id == robot) and (not fault or code == fault)

            recurring = [
                r
                for r in find_recurring_faults(parsed.events)
                if matches(r.robot_id, r.fault_code)
            ]
            if recurring:
                robot = robot or recurring[0].robot_id
                fault = fault or recurring[0].fault_code
            else:
                counts = Counter(
                    (e.robot_id, e.fault_code)
                    for e in parsed.events
                    if e.event == "fault" and e.fault_code
                    and matches(e.robot_id, e.fault_code)
                )
                if not counts:
                    print("No matching fault events in the log file.", file=sys.stderr)
                    return 1
                (robot_top, fault_top), _ = counts.most_common(1)[0]
                robot = robot or robot_top
                fault = fault or fault_top
            if args.robot:
                missing = "No fault specified"
            elif args.fault:
                missing = "No robot specified"
            else:
                missing = "No robot or fault specified"
            print(f"{missing}; using {robot} / {fault}\n")
        elif not any(
            e.robot_id == robot and e.fault_code == fault and e.event == "fault"
            for e in parsed.events
        ):
            print(f"No {fault} fault events found for robot {robot}.", file=sys.stderr)
            return 1
        print(build_rca_report(parsed.events, robot, fault))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
