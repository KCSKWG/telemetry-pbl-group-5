import re
from pathlib import Path

import numpy as np
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


def metric_value(summary_metrics: dict, key: str, fallback):
    return summary_metrics.get(key, fallback)


def safe_round(value, digits=3):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def compute_alert_breakdown(alerts: list[str]) -> dict:
    breakdown = {
        "Low DO": 0,
        "pH Risk": 0,
        "High Temperature": 0,
        "Other": 0,
    }

    for alert in alerts:
        lower = alert.lower()
        if "dissolved oxygen" in lower:
            breakdown["Low DO"] += 1
        elif "ph" in lower:
            breakdown["pH Risk"] += 1
        elif "temperature" in lower:
            breakdown["High Temperature"] += 1
        else:
            breakdown["Other"] += 1

    return breakdown


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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("TELE 523 Smart Water Quality Telemetry Dashboard")
st.caption("Telemetry simulation analytics dashboard for signal processing, communication performance, and monitoring outputs.")

if not RESULTS_DIR.exists():
    st.error("No results folder found. Run tele523_end_to_end_simulation.py first.")
    st.stop()

if processed_df is None:
    st.error("processed_data.csv not found. Run the simulation first.")
    st.stop()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Dashboard Controls")

if "timestamp" in processed_df.columns and not processed_df["timestamp"].isna().all():
    min_date = processed_df["timestamp"].min()
    max_date = processed_df["timestamp"].max()

    selected_range = st.sidebar.date_input(
        "Filter date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered_df = processed_df[
            (processed_df["timestamp"].dt.date >= start_date) &
            (processed_df["timestamp"].dt.date <= end_date)
        ].copy()
    else:
        filtered_df = processed_df.copy()
else:
    filtered_df = processed_df.copy()

show_raw_tables = st.sidebar.checkbox("Show raw tables", value=True)
show_histograms = st.sidebar.checkbox("Show histograms", value=True)
show_correlation = st.sidebar.checkbox("Show correlation heatmap", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Run order:\n"
    "1. fetch_water_quality_data.py\n"
    "2. tele523_end_to_end_simulation.py\n"
    "3. streamlit dashboard"
)


# ------------------------------------------------------------
# Top metrics
# ------------------------------------------------------------
st.subheader("Executive Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Input Samples", metric_value(summary_metrics, "input_samples", len(filtered_df)))
col2.metric("Best SNR (dB)", metric_value(summary_metrics, "best_snr_db", "N/A"))
col3.metric("Best BER", metric_value(summary_metrics, "best_ber", "N/A"))
col4.metric("RMSE", metric_value(summary_metrics, "rmse", "N/A"))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Avg Temperature", safe_round(metric_value(summary_metrics, "avg_temperature", filtered_df["temperature"].mean())))
col6.metric("Avg pH", safe_round(metric_value(summary_metrics, "avg_pH", filtered_df["pH"].mean())))
col7.metric("Avg DO", safe_round(metric_value(summary_metrics, "avg_dissolved_oxygen", filtered_df["dissolved_oxygen"].mean())))
col8.metric("Low DO Events", metric_value(summary_metrics, "low_do_events", int((filtered_df["dissolved_oxygen"] < 4.0).sum())))


# ------------------------------------------------------------
# Alerts and health summary
# ------------------------------------------------------------
st.subheader("System Health and Alerts")

health_col1, health_col2 = st.columns([1.2, 1])

with health_col1:
    if summary_alerts:
        for alert in summary_alerts:
            st.warning(alert)
    else:
        st.success("No alerts found in summary.txt")

with health_col2:
    alert_breakdown = compute_alert_breakdown(summary_alerts)
    breakdown_df = pd.DataFrame(
        {
            "Category": list(alert_breakdown.keys()),
            "Count": list(alert_breakdown.values()),
        }
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    nonzero_df = breakdown_df[breakdown_df["Count"] > 0]
    if len(nonzero_df) > 0:
        ax.pie(nonzero_df["Count"], labels=nonzero_df["Category"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Alert Category Share")
    else:
        ax.text(0.5, 0.5, "No alerts", ha="center", va="center")
        ax.set_title("Alert Category Share")
    st.pyplot(fig)


# ------------------------------------------------------------
# Water quality trends
# ------------------------------------------------------------
st.subheader("Water Quality Trend Analysis")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Temperature", "pH", "Dissolved Oxygen", "Recovered DO"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["timestamp"], filtered_df["temperature"])
    ax.set_title("Temperature Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature")
    ax.grid(True)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["timestamp"], filtered_df["pH"])
    ax.set_title("pH Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("pH")
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_df["timestamp"], filtered_df["dissolved_oxygen"], label="Dissolved Oxygen")
    ax.axhline(4.0, linestyle="--", label="Alert Threshold")
    ax.set_title("Dissolved Oxygen Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Dissolved Oxygen")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab4:
    if final_df is not None and "recovered_dissolved_oxygen_norm" in final_df.columns:
        final_filtered = final_df.copy()
        if "timestamp" in final_filtered.columns and not final_filtered["timestamp"].isna().all():
            if "timestamp" in filtered_df.columns and len(filtered_df) > 0:
                min_t = filtered_df["timestamp"].min()
                max_t = filtered_df["timestamp"].max()
                final_filtered = final_filtered[
                    (final_filtered["timestamp"] >= min_t) &
                    (final_filtered["timestamp"] <= max_t)
                ].copy()

        fig, ax = plt.subplots(figsize=(10, 4))
        if "dissolved_oxygen_norm" in final_filtered.columns:
            ax.plot(final_filtered["timestamp"], final_filtered["dissolved_oxygen_norm"], label="Original Normalized DO")
        ax.plot(final_filtered["timestamp"], final_filtered["recovered_dissolved_oxygen_norm"], label="Recovered Normalized DO")
        ax.set_title("Original vs Recovered DO")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized DO")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Recovered dissolved oxygen data not found.")


# ------------------------------------------------------------
# Distribution analysis
# ------------------------------------------------------------
if show_histograms:
    st.subheader("Signal Distribution Analysis")

    h1, h2, h3 = st.columns(3)

    with h1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(filtered_df["temperature"].dropna(), bins=20)
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        st.pyplot(fig)

    with h2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(filtered_df["pH"].dropna(), bins=20)
        ax.set_title("pH Distribution")
        ax.set_xlabel("pH")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        st.pyplot(fig)

    with h3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(filtered_df["dissolved_oxygen"].dropna(), bins=20)
        ax.set_title("Dissolved Oxygen Distribution")
        ax.set_xlabel("Dissolved Oxygen")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        st.pyplot(fig)


# ------------------------------------------------------------
# BER analysis
# ------------------------------------------------------------
st.subheader("Communication Performance")

if ber_df is not None and not ber_df.empty:
    perf1, perf2 = st.columns([1.2, 1])

    with perf1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(ber_df["snr_db"], ber_df["ber"], marker="o")
        ax.set_title("BER vs SNR")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True)
        st.pyplot(fig)

    with perf2:
        best_ber_row = ber_df.loc[ber_df["ber"].idxmin()]
        worst_ber_row = ber_df.loc[ber_df["ber"].idxmax()]

        st.metric("Best BER SNR", f'{best_ber_row["snr_db"]} dB')
        st.metric("Best BER", f'{best_ber_row["ber"]:.6f}')
        st.metric("Worst BER SNR", f'{worst_ber_row["snr_db"]} dB')
        st.metric("Worst BER", f'{worst_ber_row["ber"]:.6f}')

    st.dataframe(ber_df, use_container_width=True)
else:
    st.info("BER results not found.")


# ------------------------------------------------------------
# Segment analysis
# ------------------------------------------------------------
st.subheader("Normal vs Event Segment Analysis")

seg_col1, seg_col2, seg_col3 = st.columns(3)

normal_count = len(normal_df) if normal_df is not None else 0
event_count = len(event_df) if event_df is not None else 0
total_seg = normal_count + event_count

with seg_col1:
    st.metric("Normal Segment Samples", normal_count)

with seg_col2:
    st.metric("Event Segment Samples", event_count)

with seg_col3:
    ratio = (event_count / total_seg * 100) if total_seg > 0 else 0
    st.metric("Event Share (%)", f"{ratio:.2f}")

seg_viz1, seg_viz2 = st.columns(2)

with seg_viz1:
    counts_df = pd.DataFrame(
        {
            "Segment": ["Normal", "Event"],
            "Count": [normal_count, event_count],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts_df["Segment"], counts_df["Count"])
    ax.set_title("Normal vs Event Samples")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y")
    st.pyplot(fig)

with seg_viz2:
    fig, ax = plt.subplots(figsize=(6, 4))
    if total_seg > 0:
        ax.pie(
            [normal_count, event_count],
            labels=["Normal", "Event"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Segment Share")
    else:
        ax.text(0.5, 0.5, "No segment data", ha="center", va="center")
        ax.set_title("Segment Share")
    st.pyplot(fig)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------
if show_correlation:
    st.subheader("Correlation Analysis")

    corr_cols = ["temperature", "pH", "dissolved_oxygen"]
    corr_df = filtered_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr_df, aspect="auto")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols)
    ax.set_yticklabels(corr_cols)
    ax.set_title("Correlation Heatmap")

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(cax)
    st.pyplot(fig)


# ------------------------------------------------------------
# Raw tables
# ------------------------------------------------------------
if show_raw_tables:
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

download_cols = st.columns(3)
idx = 0

for name, path in download_options.items():
    if path.exists():
        with open(path, "rb") as f:
            with download_cols[idx % 3]:
                st.download_button(
                    label=f"Download {name}",
                    data=f,
                    file_name=name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        idx += 1