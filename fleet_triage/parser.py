"""JSON Lines log parser with tolerant handling of malformed lines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import LogEvent

KNOWN_KEYS = {
    "ts",
    "robot_id",
    "event",
    "workflow_state",
    "severity",
    "fault_code",
    "subsystem",
    "message",
    "recovery_action",
}


@dataclass
class ParseResult:
    events: list[LogEvent] = field(default_factory=list)
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def robots(self) -> list[str]:
        return sorted({e.robot_id for e in self.events})


def _parse_ts(value) -> datetime:
    """Parse an ISO 8601 timestamp. A trailing "Z" is accepted, and naive
    timestamps are assumed to be UTC so mixed logs still sort and subtract."""
    if not isinstance(value, str):
        raise TypeError(f"timestamp must be a string, got {type(value).__name__}")
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _opt_str(value) -> str | None:
    return None if value is None else str(value)


def parse_lines(lines: Iterable[str]) -> ParseResult:
    """Parse JSON Lines log content. Bad lines are counted, not fatal."""
    result = ParseResult()
    for lineno, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise TypeError("log line is not a JSON object")
            event = LogEvent(
                ts=_parse_ts(obj["ts"]),
                robot_id=str(obj["robot_id"]),
                event=str(obj["event"]),
                workflow_state=str(obj.get("workflow_state", "unknown")),
                severity=str(obj.get("severity", "info")),
                fault_code=_opt_str(obj.get("fault_code")),
                subsystem=_opt_str(obj.get("subsystem")),
                message=str(obj.get("message", "")),
                recovery_action=_opt_str(obj.get("recovery_action")),
                extra={k: v for k, v in obj.items() if k not in KNOWN_KEYS},
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            result.skipped += 1
            if len(result.errors) < 20:
                result.errors.append(f"line {lineno}: {exc}")
            continue
        result.events.append(event)
    result.events.sort(key=lambda e: e.ts)
    return result


def parse_file(path: str | Path) -> ParseResult:
    with open(path, encoding="utf-8", errors="replace") as fp:
        return parse_lines(fp)
