"""Core data model: log event schema and the fault catalog.

Every log line is a single JSON object (JSON Lines). The schema mirrors the
shape of real fleet telemetry: an event type, a robot id, a workflow state,
and optional fault fields when the event is a fault or a recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

EVENT_TYPES = ("heartbeat", "fault", "recovery", "state_change")

SEVERITIES = ("info", "warning", "error", "critical")

WORKFLOW_STATES = (
    "idle",
    "navigating",
    "picking",
    "placing",
    "charging",
    "faulted",
)

# Recovery actions ordered by escalation level. The action that finally
# clears a fault is a strong root cause signal, which the RCA engine uses.
RECOVERY_ACTIONS = (
    "auto_retry",
    "software_restart",
    "power_cycle",
    "manual_intervention",
)


@dataclass
class FaultSpec:
    """Static description of a fault type in the catalog."""

    code: str
    subsystem: str
    category: str  # hardware | comms | software | firmware | safety
    severity: str
    description: str
    # Probability weights over RECOVERY_ACTIONS: which action tends to
    # clear this fault. Used by the generator and, inversely, by the RCA
    # engine when it reasons from observed recovery patterns.
    recovery_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    # Typical downtime in minutes (min, max) once the fault fires.
    downtime_minutes: tuple[int, int] = (2, 20)


FAULT_CATALOG: dict[str, FaultSpec] = {
    spec.code: spec
    for spec in [
        FaultSpec(
            code="JOINT_OVERCURRENT",
            subsystem="actuation",
            category="hardware",
            severity="error",
            description="Joint motor drew current above the safety limit",
            recovery_weights=(0.15, 0.15, 0.45, 0.25),
            downtime_minutes=(5, 45),
        ),
        FaultSpec(
            code="JOINT_ENCODER_DRIFT",
            subsystem="actuation",
            category="hardware",
            severity="warning",
            description="Encoder position disagrees with commanded position",
            recovery_weights=(0.10, 0.20, 0.40, 0.30),
            downtime_minutes=(10, 60),
        ),
        FaultSpec(
            code="GRIPPER_STALL",
            subsystem="actuation",
            category="hardware",
            severity="error",
            description="Gripper failed to reach commanded closure",
            recovery_weights=(0.30, 0.10, 0.20, 0.40),
            downtime_minutes=(3, 30),
        ),
        FaultSpec(
            code="COMMS_TIMEOUT",
            subsystem="network",
            category="comms",
            severity="warning",
            description="Heartbeat acknowledgment not received within timeout",
            recovery_weights=(0.70, 0.20, 0.05, 0.05),
            downtime_minutes=(1, 5),
        ),
        FaultSpec(
            code="WIFI_DISCONNECT",
            subsystem="network",
            category="comms",
            severity="error",
            description="Robot dropped off the facility wireless network",
            recovery_weights=(0.55, 0.30, 0.10, 0.05),
            downtime_minutes=(2, 15),
        ),
        FaultSpec(
            code="FLEET_SERVER_UNREACHABLE",
            subsystem="network",
            category="comms",
            severity="error",
            description="Fleet coordination server did not respond",
            recovery_weights=(0.60, 0.30, 0.05, 0.05),
            downtime_minutes=(2, 20),
        ),
        FaultSpec(
            code="PLANNER_EXCEPTION",
            subsystem="autonomy",
            category="software",
            severity="error",
            description="Motion planner raised an unhandled exception",
            recovery_weights=(0.25, 0.60, 0.10, 0.05),
            downtime_minutes=(2, 10),
        ),
        FaultSpec(
            code="WORKFLOW_STALL",
            subsystem="autonomy",
            category="software",
            severity="warning",
            description="Workflow state machine made no progress past deadline",
            recovery_weights=(0.35, 0.50, 0.10, 0.05),
            downtime_minutes=(5, 25),
        ),
        FaultSpec(
            code="LOCALIZATION_LOST",
            subsystem="autonomy",
            category="software",
            severity="error",
            description="Localization confidence fell below operational floor",
            recovery_weights=(0.30, 0.40, 0.10, 0.20),
            downtime_minutes=(5, 40),
        ),
        FaultSpec(
            code="BATTERY_UNDERVOLT",
            subsystem="power",
            category="hardware",
            severity="critical",
            description="Battery voltage sagged below the undervolt cutoff",
            recovery_weights=(0.05, 0.05, 0.30, 0.60),
            downtime_minutes=(20, 120),
        ),
        FaultSpec(
            code="FIRMWARE_WATCHDOG",
            subsystem="controller",
            category="firmware",
            severity="critical",
            description="Motor controller watchdog reset the MCU",
            recovery_weights=(0.05, 0.15, 0.70, 0.10),
            downtime_minutes=(5, 30),
        ),
        FaultSpec(
            code="SENSOR_DROPOUT",
            subsystem="perception",
            category="hardware",
            severity="warning",
            description="Depth camera or lidar stopped publishing frames",
            recovery_weights=(0.30, 0.35, 0.25, 0.10),
            downtime_minutes=(2, 15),
        ),
        FaultSpec(
            code="ESTOP_TRIGGERED",
            subsystem="safety",
            category="safety",
            severity="critical",
            description="Physical or software emergency stop engaged",
            recovery_weights=(0.00, 0.05, 0.10, 0.85),
            downtime_minutes=(5, 60),
        ),
    ]
}


@dataclass
class LogEvent:
    """One parsed log line."""

    ts: datetime
    robot_id: str
    event: str
    workflow_state: str
    severity: str = "info"
    fault_code: Optional[str] = None
    subsystem: Optional[str] = None
    message: str = ""
    recovery_action: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "ts": self.ts.isoformat(),
            "robot_id": self.robot_id,
            "event": self.event,
            "workflow_state": self.workflow_state,
            "severity": self.severity,
        }
        if self.fault_code:
            d["fault_code"] = self.fault_code
        if self.subsystem:
            d["subsystem"] = self.subsystem
        if self.message:
            d["message"] = self.message
        if self.recovery_action:
            d["recovery_action"] = self.recovery_action
        d.update(self.extra)
        return d
