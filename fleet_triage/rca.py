"""Rule-based root cause suggestion engine.

The core diagnostic heuristic encodes how field technicians actually
reason about faults: the escalation level that finally clears a fault is
a strong signal about where the fault lives.

    clears on auto_retry            -> transient or environmental
    clears on software_restart      -> software state corruption
    clears only on power_cycle      -> firmware or hardware latch
    needs manual_intervention       -> mechanical or physical hardware

Those recovery-pattern rules are combined with per-fault-code knowledge
from the catalog to produce a ranked hypothesis list plus a suggested
next diagnostic step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import FAULT_CATALOG, LogEvent

RECOVERY_INFERENCE = {
    "auto_retry": (
        "transient",
        "Fault clears on automatic retry, which points at a transient or "
        "environmental cause (RF interference, momentary contention, a "
        "marginal reading near a threshold) rather than a persistent defect.",
    ),
    "software_restart": (
        "software",
        "Fault clears after a process restart, which points at software "
        "state corruption: a wedged state machine, a resource leak, or a "
        "stale cache that a restart flushes.",
    ),
    "power_cycle": (
        "firmware_or_hardware_latch",
        "Fault requires a full power cycle, which points at a latched "
        "condition below the OS: a firmware lockup, a tripped hardware "
        "protection circuit, or an MCU that stopped servicing its watchdog.",
    ),
    "manual_intervention": (
        "mechanical_or_physical",
        "Fault requires hands-on intervention, which points at a physical "
        "cause: a mechanical jam, cable or connector damage, or a "
        "component at end of life.",
    ),
}

NEXT_DIAGNOSTIC_STEP = {
    "JOINT_OVERCURRENT": "Pull the motor controller current trace for the affected joint and compare against the same joint on a healthy robot; check for mechanical binding through the full range of motion.",
    "JOINT_ENCODER_DRIFT": "Run the joint calibration routine and diff commanded versus measured position; inspect the encoder cable and connector for intermittent contact.",
    "GRIPPER_STALL": "Cycle the gripper unloaded and watch the current profile; inspect fingers and linkage for debris or wear.",
    "COMMS_TIMEOUT": "Correlate timeout timestamps with access point logs and robot location; check whether other robots in the same zone timed out in the same window.",
    "WIFI_DISCONNECT": "Map disconnect locations against the facility RF survey; check for roaming between access points at the failure positions.",
    "FLEET_SERVER_UNREACHABLE": "Check fleet server load and restart logs for the failure window; verify whether the outage was robot-local or fleet-wide.",
    "PLANNER_EXCEPTION": "Pull the planner stack trace from the robot log and reproduce with the recorded scene snapshot in simulation.",
    "WORKFLOW_STALL": "Dump the workflow state machine history and identify the state it failed to exit; check for an unacknowledged upstream dependency.",
    "LOCALIZATION_LOST": "Replay the localization bag against the map; check whether the failure clusters in a specific facility area with changed features.",
    "BATTERY_UNDERVOLT": "Pull the battery management system log for cell voltages under load; compare pack internal resistance against fleet baseline.",
    "FIRMWARE_WATCHDOG": "Capture the MCU reset reason register on next occurrence; check firmware version against the fleet and review recent firmware rollouts.",
    "SENSOR_DROPOUT": "Check the sensor's USB or ethernet link errors in the kernel log; reseat the connector and monitor frame rate for 24 hours.",
    "ESTOP_TRIGGERED": "Review the safety controller log to identify which channel tripped; interview the operator if a physical button press is recorded.",
}


@dataclass
class RootCauseHypothesis:
    label: str
    confidence: float  # 0..1, heuristic
    rationale: str


@dataclass
class RcaFinding:
    robot_id: str
    fault_code: str
    occurrences: int
    category: str
    subsystem: str
    recovery_profile: Counter
    hypotheses: list[RootCauseHypothesis] = field(default_factory=list)
    next_step: str = ""
    five_whys: list[str] = field(default_factory=list)


def analyze_fault(
    events: list[LogEvent], robot_id: str, fault_code: str
) -> RcaFinding:
    """Build an RCA finding for one (robot, fault_code) pair."""
    spec = FAULT_CATALOG.get(fault_code)
    fault_events = [
        e
        for e in events
        if e.robot_id == robot_id and e.fault_code == fault_code
    ]
    occurrences = sum(1 for e in fault_events if e.event == "fault")
    recovery_profile = Counter(
        e.recovery_action
        for e in fault_events
        if e.event == "recovery" and e.recovery_action
    )

    finding = RcaFinding(
        robot_id=robot_id,
        fault_code=fault_code,
        occurrences=occurrences,
        category=spec.category if spec else "unknown",
        subsystem=spec.subsystem if spec else "unknown",
        recovery_profile=recovery_profile,
        next_step=NEXT_DIAGNOSTIC_STEP.get(
            fault_code,
            "Capture full logs on next occurrence and compare against a healthy robot.",
        ),
    )

    total_recoveries = sum(recovery_profile.values())
    if total_recoveries:
        for action, count in recovery_profile.most_common():
            label, rationale = RECOVERY_INFERENCE.get(
                action,
                (
                    "unclassified",
                    f"Fault cleared via an unrecognized recovery action "
                    f"({action!r}); no inference rule applies.",
                ),
            )
            finding.hypotheses.append(
                RootCauseHypothesis(
                    label=label,
                    confidence=round(count / total_recoveries, 2),
                    rationale=rationale,
                )
            )
    elif spec:
        finding.hypotheses.append(
            RootCauseHypothesis(
                label=spec.category,
                confidence=0.5,
                rationale=(
                    "No recovery events observed for this fault; falling "
                    "back to the catalog category for the fault code."
                ),
            )
        )

    finding.five_whys = _five_whys_skeleton(finding)
    return finding


def _five_whys_skeleton(finding: RcaFinding) -> list[str]:
    top = finding.hypotheses[0] if finding.hypotheses else None
    spec = FAULT_CATALOG.get(finding.fault_code)
    desc = spec.description if spec else finding.fault_code
    whys = [
        f"Why did {finding.robot_id} stop working? It raised {finding.fault_code} ({desc}).",
        f"Why did {finding.fault_code} occur? Observed recovery pattern suggests: {top.label if top else 'unknown'}.",
        f"Why does that condition arise? {top.rationale if top else 'Insufficient data; gather recovery outcomes.'}",
        "Why was it not caught earlier? (Fill in: monitoring gap, missing alert threshold, or known issue without a tracking ticket.)",
        "Why does the process allow that gap? (Fill in: process or design root cause, and the corrective action that closes it.)",
    ]
    return whys
