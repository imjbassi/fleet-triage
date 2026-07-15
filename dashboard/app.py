"""Fleet health dashboard (Streamlit).

Run with:
    streamlit run dashboard/app.py

Upload a JSON Lines fleet log, or let the app fall back to the bundled
sample at data/sample_fleet.jsonl.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleet_triage.analysis import (
    classify_faults,
    compute_robot_health,
    fault_category,
    find_fleet_anomalies,
    find_recurring_faults,
)
from fleet_triage.parser import parse_lines
from fleet_triage.rca import analyze_fault

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_fleet.jsonl"

# Fixed category-to-color assignment (color follows the entity, never its
# rank). Palette values are from a CVD-validated categorical set.
CATEGORY_COLORS = {
    "hardware": "#2a78d6",
    "comms": "#1baf7a",
    "software": "#eda100",
    "firmware": "#4a3aa7",
    "safety": "#e34948",
    "unknown": "#898781",
}
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

st.set_page_config(page_title="Fleet Fault Triage", layout="wide")
st.title("Fleet Fault Triage Dashboard")


@st.cache_data
def load_events(content: str):
    return parse_lines(io.StringIO(content))


uploaded = st.sidebar.file_uploader("Fleet log (JSON Lines)", type=["jsonl", "log", "txt"])
if uploaded is not None:
    parsed = load_events(uploaded.getvalue().decode("utf-8"))
    st.sidebar.caption(f"Loaded {uploaded.name}")
elif SAMPLE_PATH.exists():
    parsed = load_events(SAMPLE_PATH.read_text(encoding="utf-8"))
    st.sidebar.caption("Using bundled sample log")
else:
    st.warning(
        "No log loaded. Upload a file, or generate one with: "
        "python -m fleet_triage generate --out data/sample_fleet.jsonl"
    )
    st.stop()

threshold = st.sidebar.slider("Recurrence threshold (hits)", 2, 10, 3)
window_hours = st.sidebar.slider("Recurrence window (hours)", 6, 72, 24)

events = parsed.events
health = compute_robot_health(events)
buckets = classify_faults(events)
recurring = find_recurring_faults(events, threshold, float(window_hours))
anomalies = find_fleet_anomalies(events)

fleet_uptime = sum(h.uptime_pct for h in health) / len(health) if health else 0.0
total_faults = sum(h.fault_count for h in health)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet uptime", f"{fleet_uptime:.1f} pct")
c2.metric("Robots", len(health))
c3.metric("Total faults", total_faults)
c4.metric("Recurring fault patterns", len(recurring))

if anomalies:
    a = anomalies[0]
    st.error(
        f"Fleet-wide anomaly: {a.fault_code} fired {a.count} times across "
        f"{a.robots_affected} robots between {a.window_start:%Y-%m-%d %H:%M} "
        f"and {a.window_end:%H:%M} UTC ({a.ratio:.1f}x baseline). "
        "Multi-robot bursts point at shared infrastructure, not individual robots."
    )

fault_df = pd.DataFrame(
    [
        {
            "ts": e.ts,
            "robot_id": e.robot_id,
            "fault_code": e.fault_code,
            "severity": e.severity,
            "category": fault_category(e.fault_code),
        }
        for e in events
        if e.event == "fault" and e.fault_code
    ]
)

left, right = st.columns(2)

with left:
    st.subheader("Uptime per robot")
    up_df = pd.DataFrame(
        {"robot": [h.robot_id for h in health], "uptime_pct": [h.uptime_pct for h in health]}
    ).sort_values("uptime_pct")
    fig = go.Figure(
        go.Bar(
            x=up_df["uptime_pct"],
            y=up_df["robot"],
            orientation="h",
            marker_color=[
                STATUS["critical"] if v < 85 else STATUS["warning"] if v < 95 else "#2a78d6"
                for v in up_df["uptime_pct"]
            ],
        )
    )
    fig.update_layout(
        xaxis_title="uptime pct",
        yaxis_title=None,
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(range=[max(0, up_df["uptime_pct"].min() - 5), 100])
    st.plotly_chart(fig, width="stretch")
    st.caption("Red below 85 pct, amber below 95 pct.")

with right:
    st.subheader("Top fault codes across the fleet")
    if not fault_df.empty:
        top = (
            fault_df.groupby(["fault_code", "category"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=True)
            .tail(10)
        )
        fig = px.bar(
            top,
            x="count",
            y="fault_code",
            color="category",
            orientation="h",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

st.subheader("Reliability trend (faults per day)")
if not fault_df.empty:
    trend = (
        fault_df.assign(day=fault_df["ts"].dt.date)
        .groupby(["day", "category"])
        .size()
        .reset_index(name="faults")
    )
    fig = px.bar(
        trend,
        x="day",
        y="faults",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

st.subheader("Recurring faults (per robot)")
if recurring:
    rec_df = pd.DataFrame(
        [
            {
                "robot": r.robot_id,
                "fault_code": r.fault_code,
                "total_hits": r.count,
                "first_seen": r.first_seen,
                "last_seen": r.last_seen,
            }
            for r in recurring
        ]
    )
    st.dataframe(rec_df, width="stretch", hide_index=True)
else:
    st.info("No recurring fault patterns at the current threshold.")

st.subheader("Root cause drilldown")
robot_ids = sorted({h.robot_id for h in health})
sel_robot = st.selectbox("Robot", robot_ids)
robot_faults = sorted(
    {e.fault_code for e in events if e.robot_id == sel_robot and e.fault_code}
)
if robot_faults:
    sel_fault = st.selectbox("Fault code", robot_faults)
    finding = analyze_fault(events, sel_robot, sel_fault)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"**{finding.fault_code}** on **{finding.robot_id}**: "
            f"{finding.occurrences} occurrences, subsystem {finding.subsystem}, "
            f"category {finding.category}"
        )
        st.markdown("**Ranked hypotheses**")
        for h in finding.hypotheses:
            st.markdown(f"- {h.label} (confidence {h.confidence:.2f}): {h.rationale}")
        st.markdown(f"**Next diagnostic step:** {finding.next_step}")
    with d2:
        if finding.recovery_profile:
            prof = pd.DataFrame(
                finding.recovery_profile.most_common(),
                columns=["recovery_action", "count"],
            )
            fig = px.bar(
                prof,
                x="count",
                y="recovery_action",
                orientation="h",
                color_discrete_sequence=["#2a78d6"],
            )
            fig.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=None
            )
            st.plotly_chart(fig, width="stretch")
            st.caption("What it took to clear this fault, each occurrence.")
else:
    st.info("Selected robot has no fault events in this log.")

st.subheader("Raw event table")
with st.expander("Show parsed events"):
    ev_df = pd.DataFrame([e.to_dict() for e in events])
    st.dataframe(ev_df, width="stretch", hide_index=True)
