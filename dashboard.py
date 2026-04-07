import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


RESULTS_DIR = Path("results")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def parse_summary(summary_path: Path) -> tuple[dict, list[str]]:
    metrics = {}
    alerts = []

    if not summary_path.exists():
        return metrics, alerts

    text = summary_path.read_text(encoding="utf-8")

    patterns = {
        "Input samples": "input_samples",
        "Normal segment samples": "normal_segment_samples",
        "Event segment samples": "event_segment_samples",
        "Quantization step size": "quantization_step_size",
        "Best SNR": "best_snr_db",
        "Final BER at best case": "best_ber",
        "Reconstruction RMSE": "rmse",
        "avg_temperature": "avg_temperature",
        "avg_pH": "avg_pH",
        "avg_dissolved_oxygen": "avg_dissolved_oxygen",
        "min_dissolved_oxygen": "min_dissolved_oxygen",
        "max_temperature": "max_temperature",
        "low_do_events": "low_do_events",
    }

    for label, key in patterns.items():
        match = re.search(rf"{re.escape(label)}:\s*([-+]?\d*\.?\d+)", text)
        if match:
            value = float(match.group(1))
            if key in ["input_samples", "normal_segment_samples", "event_segment_samples", "low_do_events"]:
                value = int(value)
            metrics[key] = value

    in_alerts = False
    for line in text.splitlines():
        line = line.strip()
        if line == "Alerts:":
            in_alerts = True
            continue
        if in_alerts and line.startswith("- "):
            alerts.append(line[2:])

    return metrics, alerts


def ensure_datetime(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
processed_df = read_csv_safe(RESULTS_DIR / "processed_data.csv")
final_df = read_csv_safe(RESULTS_DIR / "final_monitored_data.csv")
normal_df = read_csv_safe(RESULTS_DIR / "normal_segment.csv")
event_df = read_csv_safe(RESULTS_DIR / "event_segment.csv")
ber_df = read_csv_safe(RESULTS_DIR / "ber_results.csv")
summary_metrics, summary_alerts = parse_summary(RESULTS_DIR / "summary.txt")

if processed_df is not None:
    processed_df = ensure_datetime(processed_df)

if final_df is not None:
    final_df = ensure_datetime(final_df)

if normal_df is not None:
    normal_df = ensure_datetime(normal_df)

if event_df is not None:
    event_df = ensure_datetime(event_df)


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="TELE 523 Telemetry Dashboard",
    layout="wide"
)

st.title("TELE 523 Smart Water Quality Telemetry Dashboard")
st.caption("Simulation results dashboard for telemetry, signal processing, BER analysis, and monitoring.")

if not RESULTS_DIR.exists():
    st.error("No results folder found. Run tele523_end_to_end_simulation.py first.")
    st.stop()

if processed_df is None:
    st.error("processed_data.csv not found. Run the simulation first.")
    st.stop()


# ------------------------------------------------------------
# Top metrics
# ------------------------------------------------------------
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Input Samples", summary_metrics.get("input_samples", len(processed_df)))
col2.metric("Best SNR (dB)", summary_metrics.get("best_snr_db", "N/A"))
col3.metric("Best BER", summary_metrics.get("best_ber", "N/A"))
col4.metric("RMSE", summary_metrics.get("rmse", "N/A"))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Avg Temperature", round(summary_metrics.get("avg_temperature", processed_df["temperature"].mean()), 3))
col6.metric("Avg pH", round(summary_metrics.get("avg_pH", processed_df["pH"].mean()), 3))
col7.metric("Avg DO", round(summary_metrics.get("avg_dissolved_oxygen", processed_df["dissolved_oxygen"].mean()), 3))
col8.metric("Low DO Events", summary_metrics.get("low_do_events", int((processed_df["dissolved_oxygen"] < 4.0).sum())))


# ------------------------------------------------------------
# Alerts
# ------------------------------------------------------------
st.subheader("System Alerts")
if summary_alerts:
    for alert in summary_alerts:
        st.warning(alert)
else:
    st.success("No alerts found in summary.txt")


# ------------------------------------------------------------
# Trend charts
# ------------------------------------------------------------
st.subheader("Water Quality Trends")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Temperature", "pH", "Dissolved Oxygen", "Recovered DO"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(processed_df["timestamp"], processed_df["temperature"])
    ax.set_title("Temperature Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.grid(True)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(processed_df["timestamp"], processed_df["pH"])
    ax.set_title("pH Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("pH")
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(processed_df["timestamp"], processed_df["dissolved_oxygen"], label="Dissolved Oxygen")
    ax.axhline(4.0, linestyle="--", label="Alert Threshold")
    ax.set_title("Dissolved Oxygen Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Dissolved Oxygen")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab4:
    if final_df is not None and "recovered_dissolved_oxygen_norm" in final_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(final_df["timestamp"], final_df["dissolved_oxygen_norm"], label="Original Normalized DO")
        ax.plot(final_df["timestamp"], final_df["recovered_dissolved_oxygen_norm"], label="Recovered Normalized DO")
        ax.set_title("Original vs Recovered DO")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized DO")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Recovered dissolved oxygen data not found.")


# ------------------------------------------------------------
# BER Analysis
# ------------------------------------------------------------
st.subheader("Communication Performance")

if ber_df is not None and not ber_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(ber_df["snr_db"], ber_df["ber"], marker="o")
    ax.set_title("BER vs SNR")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True)
    st.pyplot(fig)

    st.dataframe(ber_df, use_container_width=True)
else:
    st.info("BER results not found.")


# ------------------------------------------------------------
# Segment analysis
# ------------------------------------------------------------
st.subheader("Segment Analysis")

seg_col1, seg_col2 = st.columns(2)

with seg_col1:
    if normal_df is not None:
        st.metric("Normal Segment Samples", len(normal_df))
    else:
        st.metric("Normal Segment Samples", 0)

with seg_col2:
    if event_df is not None:
        st.metric("Event Segment Samples", len(event_df))
    else:
        st.metric("Event Segment Samples", 0)

if normal_df is not None or event_df is not None:
    counts_df = pd.DataFrame(
        {
            "Segment": ["Normal", "Event"],
            "Count": [
                len(normal_df) if normal_df is not None else 0,
                len(event_df) if event_df is not None else 0,
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts_df["Segment"], counts_df["Count"])
    ax.set_title("Normal vs Event Samples")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")
    st.pyplot(fig)


# ------------------------------------------------------------
# Raw tables
# ------------------------------------------------------------
st.subheader("Raw Data Tables")

table_choice = st.selectbox(
    "Choose table to view",
    [
        "Processed Data",
        "Final Monitored Data",
        "Normal Segment",
        "Event Segment",
        "BER Results",
    ],
)

if table_choice == "Processed Data" and processed_df is not None:
    st.dataframe(processed_df, use_container_width=True)

elif table_choice == "Final Monitored Data" and final_df is not None:
    st.dataframe(final_df, use_container_width=True)

elif table_choice == "Normal Segment" and normal_df is not None:
    st.dataframe(normal_df, use_container_width=True)

elif table_choice == "Event Segment" and event_df is not None:
    st.dataframe(event_df, use_container_width=True)

elif table_choice == "BER Results" and ber_df is not None:
    st.dataframe(ber_df, use_container_width=True)

else:
    st.info("Selected table not available.")


# ------------------------------------------------------------
# Downloads
# ------------------------------------------------------------
st.subheader("Download Results")

download_options = {
    "processed_data.csv": RESULTS_DIR / "processed_data.csv",
    "final_monitored_data.csv": RESULTS_DIR / "final_monitored_data.csv",
    "normal_segment.csv": RESULTS_DIR / "normal_segment.csv",
    "event_segment.csv": RESULTS_DIR / "event_segment.csv",
    "ber_results.csv": RESULTS_DIR / "ber_results.csv",
    "summary.txt": RESULTS_DIR / "summary.txt",
}

for name, path in download_options.items():
    if path.exists():
        with open(path, "rb") as f:
            st.download_button(
                label=f"Download {name}",
                data=f,
                file_name=name,
                mime="application/octet-stream",
            )