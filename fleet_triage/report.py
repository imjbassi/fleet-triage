"""Plain-text report formatting for the CLI."""

from __future__ import annotations

from .analysis import (
    FleetAnomaly,
    RecurringFault,
    RobotHealth,
    classify_faults,
    compute_robot_health,
    daily_fault_trend,
    find_fleet_anomalies,
    find_recurring_faults,
)
from .models import LogEvent
from .parser import ParseResult
from .rca import RcaFinding, analyze_fault

RULE = "=" * 72
THIN = "-" * 72


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def triage_summary(
    parsed: ParseResult,
    threshold: int = 3,
    window_hours: float = 24.0,
) -> str:
    """Full fleet triage: classification, recurrence, anomalies, health."""
    events = parsed.events
    buckets = classify_faults(events)
    recurring = find_recurring_faults(events, threshold, window_hours)
    anomalies = find_fleet_anomalies(events)
    health = compute_robot_health(events)

    out: list[str] = []
    out.append(RULE)
    out.append("FLEET FAULT TRIAGE SUMMARY")
    out.append(RULE)
    if events:
        out.append(
            f"Window: {events[0].ts.isoformat()} to {events[-1].ts.isoformat()}"
        )
    out.append(
        f"Events: {len(events)} parsed, {parsed.skipped} malformed lines skipped"
    )
    out.append(f"Robots: {len(parsed.robots)}")
    out.append("")

    out.append("FAULTS BY CATEGORY")
    out.append(THIN)
    total_faults = sum(buckets["by_category"].values())
    for cat, n in buckets["by_category"].most_common():
        pct = 100.0 * n / total_faults if total_faults else 0.0
        out.append(f"  {cat:<12} {n:>4}  ({pct:.1f} pct)")
    out.append("")

    out.append("TOP FAULT CODES")
    out.append(THIN)
    for code, n in buckets["by_code"].most_common(8):
        out.append(f"  {code:<26} {n:>4}")
    out.append("")

    out.append(f"RECURRING FAULTS (>= {threshold} hits in a {window_hours:.0f}h window)")
    out.append(THIN)
    if recurring:
        rows = [
            [
                r.robot_id,
                r.fault_code,
                str(r.count),
                r.first_seen.strftime("%m-%d %H:%M"),
                r.last_seen.strftime("%m-%d %H:%M"),
            ]
            for r in recurring
        ]
        out.append(
            _table(["robot", "fault_code", "total", "first_seen", "last_seen"], rows)
        )
    else:
        out.append("  none detected")
    out.append("")

    out.append("FLEET-WIDE ANOMALIES (multi-robot fault bursts)")
    out.append(THIN)
    if anomalies:
        for a in anomalies[:5]:
            out.append(
                f"  {a.fault_code}: {a.count} events across "
                f"{a.robots_affected} robots in "
                f"{a.window_start.strftime('%m-%d %H:%M')} to "
                f"{a.window_end.strftime('%H:%M')} "
                f"({a.ratio:.1f}x baseline)"
            )
        out.append(
            "  Interpretation: simultaneous multi-robot bursts point at shared"
        )
        out.append(
            "  infrastructure (network, fleet server, facility power), not robots."
        )
    else:
        out.append("  none detected")
    out.append("")

    out.append("ROBOT HEALTH")
    out.append(THIN)
    rows = []
    for h in sorted(health, key=lambda h: h.uptime_pct):
        rows.append(
            [
                h.robot_id,
                f"{h.uptime_pct:.1f}",
                str(h.fault_count),
                f"{h.downtime_minutes:.0f}",
                f"{h.mtbf_hours:.1f}" if h.mtbf_hours else "n/a",
                ", ".join(c for c, _ in h.faults_by_code.most_common(2)) or "none",
            ]
        )
    out.append(
        _table(
            ["robot", "uptime_pct", "faults", "downtime_min", "mtbf_h", "top_faults"],
            rows,
        )
    )
    out.append("")

    trend = daily_fault_trend(events)
    if trend:
        out.append("DAILY FAULT TREND")
        out.append(THIN)
        peak = max(n for _, n in trend)
        for day, n in trend:
            bar = "#" * max(1, round(30 * n / peak))
            out.append(f"  {day}  {n:>4}  {bar}")
        out.append("")

    if recurring:
        worst = recurring[0]
        out.append("SUGGESTED NEXT ACTION")
        out.append(THIN)
        out.append(
            f"  Run: fleet-triage report <logfile> --robot {worst.robot_id} "
            f"--fault {worst.fault_code}"
        )
    return "\n".join(out)


def rca_report(finding: RcaFinding) -> str:
    """Formatted RCA-style report for one (robot, fault) pair."""
    out: list[str] = []
    out.append(RULE)
    out.append(f"ROOT CAUSE ANALYSIS: {finding.fault_code} on {finding.robot_id}")
    out.append(RULE)
    out.append(f"Subsystem:   {finding.subsystem}")
    out.append(f"Category:    {finding.category}")
    out.append(f"Occurrences: {finding.occurrences}")
    out.append("")

    out.append("OBSERVED RECOVERY PROFILE")
    out.append(THIN)
    if finding.recovery_profile:
        total = sum(finding.recovery_profile.values())
        for action, n in finding.recovery_profile.most_common():
            out.append(f"  {action:<22} {n:>3}  ({100.0 * n / total:.0f} pct)")
    else:
        out.append("  no recovery events recorded")
    out.append("")

    out.append("ROOT CAUSE HYPOTHESES (ranked)")
    out.append(THIN)
    for i, h in enumerate(finding.hypotheses, start=1):
        out.append(f"  {i}. {h.label} (confidence {h.confidence:.2f})")
        out.append(f"     {h.rationale}")
    out.append("")

    out.append("SUGGESTED NEXT DIAGNOSTIC STEP")
    out.append(THIN)
    out.append(f"  {finding.next_step}")
    out.append("")

    out.append("5 WHYS SKELETON")
    out.append(THIN)
    for i, why in enumerate(finding.five_whys, start=1):
        out.append(f"  {i}. {why}")
    return "\n".join(out)


def build_rca_report(events: list[LogEvent], robot_id: str, fault_code: str) -> str:
    return rca_report(analyze_fault(events, robot_id, fault_code))
