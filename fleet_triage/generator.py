"""Synthetic fleet log generator.

Produces realistic JSON Lines telemetry for a fleet of robots: periodic
heartbeats, workflow state changes, injected faults drawn from the fault
catalog, and matching recovery events. A few robots are designated
"lemons" with elevated fault rates so recurrence detection has something
real to find, and one fault type is given a fleet-wide burst window so
fleet-level anomaly detection has a signal too.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Iterator, TextIO

from .models import FAULT_CATALOG, RECOVERY_ACTIONS, LogEvent

WORKFLOW_CYCLE = ("idle", "navigating", "picking", "placing", "navigating")


def generate_fleet_log(
    robots: int = 10,
    hours: float = 72.0,
    heartbeat_seconds: int = 300,
    base_fault_rate_per_hour: float = 0.12,
    lemon_count: int = 2,
    lemon_multiplier: float = 4.0,
    seed: int | None = 42,
    start: datetime | None = None,
) -> Iterator[LogEvent]:
    """Yield synthetic LogEvents for the whole fleet in timestamp order."""
    if robots < 1:
        raise ValueError("robots must be at least 1")
    if hours <= 0:
        raise ValueError("hours must be positive")
    if heartbeat_seconds < 1:
        raise ValueError("heartbeat_seconds must be at least 1")
    rng = random.Random(seed)
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
    end = start + timedelta(hours=hours)

    robot_ids = [f"R-{i:02d}" for i in range(1, robots + 1)]
    lemons = set(rng.sample(robot_ids, min(lemon_count, robots)))
    fault_codes = list(FAULT_CATALOG)

    # Each lemon robot gets a signature fault it keeps hitting, which is
    # what makes per-robot recurrence detection meaningful.
    lemon_signature = {rid: rng.choice(fault_codes) for rid in lemons}

    # A fleet-wide comms burst: every robot sees elevated COMMS_TIMEOUT
    # rates inside this window, simulating a facility network incident.
    burst_start = start + timedelta(hours=hours * rng.uniform(0.3, 0.6))
    burst_end = burst_start + timedelta(hours=min(4.0, hours / 6))

    events: list[LogEvent] = []
    for rid in robot_ids:
        events.extend(
            _robot_stream(
                rng,
                rid,
                start,
                end,
                heartbeat_seconds,
                base_fault_rate_per_hour
                * (lemon_multiplier if rid in lemons else 1.0),
                lemon_signature.get(rid),
                fault_codes,
                burst_start,
                burst_end,
            )
        )
    events.sort(key=lambda e: e.ts)
    yield from events


def _robot_stream(
    rng: random.Random,
    robot_id: str,
    start: datetime,
    end: datetime,
    heartbeat_seconds: int,
    fault_rate_per_hour: float,
    signature_fault: str | None,
    fault_codes: list[str],
    burst_start: datetime,
    burst_end: datetime,
) -> list[LogEvent]:
    events: list[LogEvent] = []
    t = start + timedelta(seconds=rng.uniform(0, heartbeat_seconds))
    state_idx = rng.randrange(len(WORKFLOW_CYCLE))
    state = WORKFLOW_CYCLE[state_idx]
    faulted_until: datetime | None = None

    while t < end:
        in_burst = burst_start <= t <= burst_end

        if faulted_until is not None and t >= faulted_until:
            faulted_until = None

        if faulted_until is None:
            # Advance the workflow occasionally.
            if rng.random() < 0.15:
                state_idx = (state_idx + 1) % len(WORKFLOW_CYCLE)
                state = WORKFLOW_CYCLE[state_idx]
                events.append(
                    LogEvent(
                        ts=t,
                        robot_id=robot_id,
                        event="state_change",
                        workflow_state=state,
                        message=f"Entered workflow state {state}",
                    )
                )

            events.append(
                LogEvent(
                    ts=t,
                    robot_id=robot_id,
                    event="heartbeat",
                    workflow_state=state,
                    extra={
                        "battery_pct": round(rng.uniform(20, 100), 1),
                        "cpu_pct": round(rng.uniform(5, 85), 1),
                    },
                )
            )

            # Poisson-ish fault injection per heartbeat interval.
            p_fault = fault_rate_per_hour * heartbeat_seconds / 3600.0
            if in_burst:
                p_fault += 0.20  # elevated comms failures fleet-wide
            if rng.random() < p_fault:
                if in_burst and rng.random() < 0.7:
                    code = "COMMS_TIMEOUT"
                elif signature_fault and rng.random() < 0.6:
                    code = signature_fault
                else:
                    code = rng.choice(fault_codes)
                spec = FAULT_CATALOG[code]
                action = rng.choices(RECOVERY_ACTIONS, spec.recovery_weights)[0]
                lo, hi = spec.downtime_minutes
                downtime = timedelta(minutes=rng.uniform(lo, hi))

                events.append(
                    LogEvent(
                        ts=t,
                        robot_id=robot_id,
                        event="fault",
                        workflow_state="faulted",
                        severity=spec.severity,
                        fault_code=code,
                        subsystem=spec.subsystem,
                        message=spec.description,
                    )
                )
                recovery_ts = t + downtime
                if recovery_ts < end:
                    events.append(
                        LogEvent(
                            ts=recovery_ts,
                            robot_id=robot_id,
                            event="recovery",
                            workflow_state=state,
                            severity="info",
                            fault_code=code,
                            subsystem=spec.subsystem,
                            recovery_action=action,
                            message=f"Fault {code} cleared via {action}",
                        )
                    )
                faulted_until = recovery_ts

        t += timedelta(seconds=heartbeat_seconds)
    return events


def write_jsonl(events: Iterator[LogEvent], fp: TextIO) -> int:
    """Serialize events to a JSON Lines stream. Returns the line count."""
    n = 0
    for ev in events:
        fp.write(json.dumps(ev.to_dict()) + "\n")
        n += 1
    return n
