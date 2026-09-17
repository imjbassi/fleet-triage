"""Tests for the parser, classifier, recurrence detection, and RCA engine."""

import io
import json
import unittest
from datetime import datetime, timedelta, timezone

from fleet_triage.analysis import (
    classify_faults,
    compute_robot_health,
    find_fleet_anomalies,
    find_recurring_faults,
)
from fleet_triage.generator import generate_fleet_log, write_jsonl
from fleet_triage.parser import parse_lines
from fleet_triage.rca import analyze_fault

T0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def make_line(ts, robot_id, event, **kw):
    d = {"ts": ts.isoformat(), "robot_id": robot_id, "event": event,
         "workflow_state": kw.pop("workflow_state", "idle")}
    d.update(kw)
    return json.dumps(d)


class ParserTests(unittest.TestCase):
    def test_parses_valid_and_skips_malformed(self):
        lines = [
            make_line(T0, "R-01", "heartbeat"),
            "not json at all",
            '{"robot_id": "R-01"}',
            make_line(T0 + timedelta(minutes=1), "R-01", "fault",
                      fault_code="COMMS_TIMEOUT", severity="warning"),
        ]
        result = parse_lines(lines)
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.skipped, 2)
        self.assertEqual(result.robots, ["R-01"])

    def test_events_sorted_by_timestamp(self):
        lines = [
            make_line(T0 + timedelta(minutes=5), "R-02", "heartbeat"),
            make_line(T0, "R-01", "heartbeat"),
        ]
        result = parse_lines(lines)
        self.assertEqual(result.events[0].robot_id, "R-01")


class ClassifierTests(unittest.TestCase):
    def test_category_buckets(self):
        lines = [
            make_line(T0, "R-01", "fault", fault_code="COMMS_TIMEOUT"),
            make_line(T0, "R-01", "fault", fault_code="WIFI_DISCONNECT"),
            make_line(T0, "R-02", "fault", fault_code="JOINT_OVERCURRENT"),
            make_line(T0, "R-02", "fault", fault_code="PLANNER_EXCEPTION"),
        ]
        buckets = classify_faults(parse_lines(lines).events)
        self.assertEqual(buckets["by_category"]["comms"], 2)
        self.assertEqual(buckets["by_category"]["hardware"], 1)
        self.assertEqual(buckets["by_category"]["software"], 1)


class RecurrenceTests(unittest.TestCase):
    def test_flags_repeat_offender_within_window(self):
        lines = [
            make_line(T0 + timedelta(hours=i * 4), "R-03", "fault",
                      fault_code="GRIPPER_STALL")
            for i in range(3)
        ]
        recurring = find_recurring_faults(
            parse_lines(lines).events, threshold=3, window_hours=24
        )
        self.assertEqual(len(recurring), 1)
        self.assertEqual(recurring[0].robot_id, "R-03")
        self.assertEqual(recurring[0].fault_code, "GRIPPER_STALL")

    def test_spread_out_faults_not_flagged(self):
        lines = [
            make_line(T0 + timedelta(hours=i * 30), "R-03", "fault",
                      fault_code="GRIPPER_STALL")
            for i in range(3)
        ]
        recurring = find_recurring_faults(
            parse_lines(lines).events, threshold=3, window_hours=24
        )
        self.assertEqual(recurring, [])


class AnomalyTests(unittest.TestCase):
    def test_multi_robot_burst_detected(self):
        lines = []
        # Quiet baseline: one fault per day elsewhere.
        for d in range(3):
            lines.append(make_line(T0 + timedelta(days=d), "R-01", "fault",
                                   fault_code="COMMS_TIMEOUT"))
        # Burst: five robots all hit the same fault in one hour.
        burst = T0 + timedelta(days=1, hours=2)
        for i in range(5):
            lines.append(make_line(burst + timedelta(minutes=i * 5), f"R-0{i + 1}",
                                   "fault", fault_code="COMMS_TIMEOUT"))
        anomalies = find_fleet_anomalies(parse_lines(lines).events)
        self.assertTrue(anomalies)
        self.assertEqual(anomalies[0].fault_code, "COMMS_TIMEOUT")
        self.assertGreaterEqual(anomalies[0].robots_affected, 3)


class HealthTests(unittest.TestCase):
    def test_downtime_from_fault_to_recovery(self):
        lines = [
            make_line(T0, "R-01", "heartbeat"),
            make_line(T0 + timedelta(hours=1), "R-01", "fault",
                      fault_code="WIFI_DISCONNECT"),
            make_line(T0 + timedelta(hours=1, minutes=30), "R-01", "recovery",
                      fault_code="WIFI_DISCONNECT", recovery_action="auto_retry"),
            make_line(T0 + timedelta(hours=10), "R-01", "heartbeat"),
        ]
        health = compute_robot_health(parse_lines(lines).events)
        self.assertEqual(len(health), 1)
        self.assertAlmostEqual(health[0].downtime_minutes, 30.0, places=1)
        self.assertAlmostEqual(health[0].uptime_pct, 95.0, places=1)


class RcaTests(unittest.TestCase):
    def _events_with_recoveries(self, action):
        lines = []
        for i in range(4):
            t = T0 + timedelta(hours=i)
            lines.append(make_line(t, "R-05", "fault",
                                   fault_code="FIRMWARE_WATCHDOG"))
            lines.append(make_line(t + timedelta(minutes=10), "R-05", "recovery",
                                   fault_code="FIRMWARE_WATCHDOG",
                                   recovery_action=action))
        return parse_lines(lines).events

    def test_power_cycle_pattern_suggests_latch(self):
        finding = analyze_fault(
            self._events_with_recoveries("power_cycle"), "R-05", "FIRMWARE_WATCHDOG"
        )
        self.assertEqual(finding.occurrences, 4)
        self.assertEqual(finding.hypotheses[0].label, "firmware_or_hardware_latch")
        self.assertEqual(finding.hypotheses[0].confidence, 1.0)
        self.assertEqual(len(finding.five_whys), 5)

    def test_restart_pattern_suggests_software(self):
        finding = analyze_fault(
            self._events_with_recoveries("software_restart"), "R-05", "FIRMWARE_WATCHDOG"
        )
        self.assertEqual(finding.hypotheses[0].label, "software")


class RobustnessTests(unittest.TestCase):
    def test_mixed_naive_aware_and_z_timestamps(self):
        lines = [
            json.dumps({"ts": "2026-07-01T12:05:00", "robot_id": "R-01", "event": "heartbeat"}),
            json.dumps({"ts": "2026-07-01T12:00:00Z", "robot_id": "R-01", "event": "heartbeat"}),
            make_line(T0 + timedelta(minutes=10), "R-01", "heartbeat"),
        ]
        result = parse_lines(lines)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.events[0].ts, T0)
        compute_robot_health(result.events)

    def test_non_object_lines_skipped(self):
        result = parse_lines(["[1, 2]", "42", '{"ts": 5, "robot_id": "R-01", "event": "x"}'])
        self.assertEqual(result.skipped, 3)

    def test_unknown_recovery_action_does_not_crash(self):
        lines = [
            make_line(T0, "R-01", "fault", fault_code="GRIPPER_STALL"),
            make_line(T0 + timedelta(minutes=5), "R-01", "recovery",
                      fault_code="GRIPPER_STALL", recovery_action="swap_robot"),
        ]
        finding = analyze_fault(parse_lines(lines).events, "R-01", "GRIPPER_STALL")
        self.assertEqual(finding.hypotheses[0].label, "unclassified")

    def test_overlapping_faults_not_double_counted(self):
        lines = [
            make_line(T0, "R-01", "heartbeat"),
            make_line(T0 + timedelta(hours=1), "R-01", "fault", fault_code="WIFI_DISCONNECT"),
            make_line(T0 + timedelta(hours=1, minutes=10), "R-01", "fault",
                      fault_code="COMMS_TIMEOUT"),
            make_line(T0 + timedelta(hours=1, minutes=20), "R-01", "recovery",
                      fault_code="COMMS_TIMEOUT", recovery_action="auto_retry"),
            make_line(T0 + timedelta(hours=1, minutes=30), "R-01", "recovery",
                      fault_code="WIFI_DISCONNECT", recovery_action="auto_retry"),
            make_line(T0 + timedelta(hours=10), "R-01", "heartbeat"),
        ]
        health = compute_robot_health(parse_lines(lines).events)
        self.assertAlmostEqual(health[0].downtime_minutes, 30.0, places=1)


class CliTests(unittest.TestCase):
    def _log(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            for i in range(3):
                fp.write(make_line(T0 + timedelta(hours=i), "R-01", "fault",
                                   fault_code="GRIPPER_STALL") + "\n")
            fp.write(make_line(T0, "R-02", "fault", fault_code="COMMS_TIMEOUT") + "\n")
        self.addCleanup(os.remove, path)
        return path

    def _run(self, *argv):
        from contextlib import redirect_stderr, redirect_stdout
        from fleet_triage.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_report_robot_only_picks_fault_for_that_robot(self):
        code, out, _ = self._run("report", self._log(), "--robot", "R-02")
        self.assertEqual(code, 0)
        self.assertIn("COMMS_TIMEOUT on R-02", out)
        self.assertIn("No fault specified; using R-02 / COMMS_TIMEOUT", out)

    def test_report_unknown_pair_errors(self):
        code, _, err = self._run("report", self._log(), "--robot", "R-99", "--fault", "X")
        self.assertEqual(code, 1)
        self.assertIn("No X fault events", err)

    def test_missing_file_errors(self):
        code, _, err = self._run("triage", "does/not/exist.jsonl")
        self.assertEqual(code, 1)


class GeneratorTests(unittest.TestCase):
    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            list(generate_fleet_log(robots=0))

    def test_generated_log_round_trips_through_parser(self):
        buf = io.StringIO()
        n = write_jsonl(
            generate_fleet_log(robots=4, hours=12, heartbeat_seconds=600, seed=7),
            buf,
        )
        buf.seek(0)
        result = parse_lines(buf)
        self.assertEqual(len(result.events), n)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(len(result.robots), 4)
        self.assertTrue(any(e.event == "fault" for e in result.events))

    def test_deterministic_with_seed(self):
        a, b = io.StringIO(), io.StringIO()
        start = T0
        write_jsonl(generate_fleet_log(robots=3, hours=6, seed=9, start=start), a)
        write_jsonl(generate_fleet_log(robots=3, hours=6, seed=9, start=start), b)
        self.assertEqual(a.getvalue(), b.getvalue())


if __name__ == "__main__":
    unittest.main()
