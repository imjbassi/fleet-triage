"""Fault classification, recurrence detection, fleet anomalies, and metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import FAULT_CATALOG, LogEvent


def fault_category(fault_code: str) -> str:
    spec = FAULT_CATALOG.get(fault_code)
    return spec.category if spec else "unknown"


@dataclass
class RecurringFault:
    robot_id: str
    fault_code: str
    count: int
    window_hours: float
    first_seen: datetime
    last_seen: datetime


@dataclass
class FleetAnomaly:
    fault_code: str
    window_start: datetime
    window_end: datetime
    count: int
    robots_affected: int
    baseline_per_window: float

    @property
    def ratio(self) -> float:
        return self.count / self.baseline_per_window if self.baseline_per_window else float("inf")


@dataclass
class RobotHealth:
    robot_id: str
    total_events: int = 0
    fault_count: int = 0
    faults_by_code: Counter = field(default_factory=Counter)
    faults_by_category: Counter = field(default_factory=Counter)
    downtime_minutes: float = 0.0
    uptime_pct: float = 100.0
    mtbf_hours: float | None = None


def faults_only(events: list[LogEvent]) -> list[LogEvent]:
    return [e for e in events if e.event == "fault" and e.fault_code]


def classify_faults(events: list[LogEvent]) -> dict[str, Counter]:
    """Bucket fault events by category, code, and subsystem."""
    faults = faults_only(events)
    return {
        "by_category": Counter(fault_category(e.fault_code) for e in faults),
        "by_code": Counter(e.fault_code for e in faults),
        "by_subsystem": Counter(e.subsystem or "unknown" for e in faults),
        "by_severity": Counter(e.severity for e in faults),
    }


def find_recurring_faults(
    events: list[LogEvent],
    threshold: int = 3,
    window_hours: float = 24.0,
) -> list[RecurringFault]:
    """Flag (robot, fault_code) pairs that fired >= threshold times within
    any rolling window of window_hours."""
    window = timedelta(hours=window_hours)
    per_key: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for e in faults_only(events):
        per_key[(e.robot_id, e.fault_code)].append(e.ts)

    recurring: list[RecurringFault] = []
    for (robot_id, code), stamps in per_key.items():
        stamps.sort()
        best = 0
        lo = 0
        for hi in range(len(stamps)):
            while stamps[hi] - stamps[lo] > window:
                lo += 1
            best = max(best, hi - lo + 1)
        if best >= threshold:
            recurring.append(
                RecurringFault(
                    robot_id=robot_id,
                    fault_code=code,
                    count=len(stamps),
                    window_hours=window_hours,
                    first_seen=stamps[0],
                    last_seen=stamps[-1],
                )
            )
    recurring.sort(key=lambda r: r.count, reverse=True)
    return recurring


def find_fleet_anomalies(
    events: list[LogEvent],
    bucket_hours: float = 1.0,
    min_robots: int = 3,
    spike_ratio: float = 3.0,
) -> list[FleetAnomaly]:
    """Detect fleet-wide fault bursts: hour buckets where one fault code
    fires across many robots at well above its own baseline rate."""
    faults = faults_only(events)
    if not faults:
        return []
    bucket = timedelta(hours=bucket_hours)
    t0 = min(e.ts for e in faults)
    t1 = max(e.ts for e in faults)
    n_buckets = max(1, int((t1 - t0) / bucket) + 1)

    per_code_bucket: dict[str, dict[int, list[LogEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for e in faults:
        idx = int((e.ts - t0) / bucket)
        per_code_bucket[e.fault_code][idx].append(e)

    anomalies: list[FleetAnomaly] = []
    for code, buckets in per_code_bucket.items():
        total = sum(len(v) for v in buckets.values())
        baseline = total / n_buckets
        for idx, evs in buckets.items():
            robots = {e.robot_id for e in evs}
            if len(robots) >= min_robots and len(evs) >= spike_ratio * max(baseline, 0.5):
                anomalies.append(
                    FleetAnomaly(
                        fault_code=code,
                        window_start=t0 + idx * bucket,
                        window_end=t0 + (idx + 1) * bucket,
                        count=len(evs),
                        robots_affected=len(robots),
                        baseline_per_window=baseline,
                    )
                )
    anomalies.sort(key=lambda a: a.count, reverse=True)
    return anomalies


def compute_robot_health(events: list[LogEvent]) -> list[RobotHealth]:
    """Per-robot uptime, downtime, fault counts, and MTBF.

    Downtime is measured from each fault event to the matching recovery
    event (same robot and fault code, first recovery after the fault).
    """
    by_robot: dict[str, list[LogEvent]] = defaultdict(list)
    for e in events:
        by_robot[e.robot_id].append(e)

    results: list[RobotHealth] = []
    for robot_id, evs in sorted(by_robot.items()):
        evs.sort(key=lambda e: e.ts)
        health = RobotHealth(robot_id=robot_id, total_events=len(evs))
        span_hours = max(
            (evs[-1].ts - evs[0].ts).total_seconds() / 3600.0, 1e-9
        )

        open_faults: dict[str, datetime] = {}
        fault_stamps: list[datetime] = []
        for e in evs:
            if e.event == "fault" and e.fault_code:
                health.fault_count += 1
                health.faults_by_code[e.fault_code] += 1
                health.faults_by_category[fault_category(e.fault_code)] += 1
                open_faults.setdefault(e.fault_code, e.ts)
                fault_stamps.append(e.ts)
            elif e.event == "recovery" and e.fault_code in open_faults:
                started = open_faults.pop(e.fault_code)
                health.downtime_minutes += (e.ts - started).total_seconds() / 60.0

        # Faults never recovered within the log window count as down
        # until the end of the robot's log span.
        for code, started in open_faults.items():
            health.downtime_minutes += (evs[-1].ts - started).total_seconds() / 60.0

        health.uptime_pct = max(
            0.0, 100.0 * (1.0 - health.downtime_minutes / 60.0 / span_hours)
        )
        if len(fault_stamps) >= 2:
            gaps = [
                (b - a).total_seconds() / 3600.0
                for a, b in zip(fault_stamps, fault_stamps[1:])
            ]
            health.mtbf_hours = sum(gaps) / len(gaps)
        results.append(health)
    return results


def daily_fault_trend(events: list[LogEvent]) -> list[tuple[str, int]]:
    """Fault counts per calendar day, sorted by day."""
    counts = Counter(e.ts.date().isoformat() for e in faults_only(events))
    return sorted(counts.items())
