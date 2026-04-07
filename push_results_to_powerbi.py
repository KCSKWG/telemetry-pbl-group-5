import json
import re
import urllib.request
from pathlib import Path
import pandas as pd


# ============================================================
# TELE 523 -> Power BI Push Script
# Pushes simulation outputs into a Power BI push semantic model
# No 'requests' package needed
# ============================================================

RESULTS_DIR = Path("results")

# -----------------------------
# Fill these before running
# -----------------------------
POWER_BI_ACCESS_TOKEN = "PASTE_YOUR_BEARER_TOKEN_HERE"
WORKSPACE_ID = "https://app.powerbi.com/groups/me/list?experience=power-bi"

# If DATASET_ID is empty, the script will create a new push dataset
DATASET_ID = ""

DATASET_NAME = "TELE523 Water Quality Telemetry"
SUMMARY_TABLE = "SimulationSummary"
BER_TABLE = "BerResults"
ALERTS_TABLE = "Alerts"


# ============================================================
# HTTP helper
# ============================================================
def powerbi_request(method: str, url: str, token: str, payload: dict | None = None) -> dict | None:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    with urllib.request.urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8")
        if not text.strip():
            return None
        return json.loads(text)


# ============================================================
# Power BI dataset creation
# ============================================================
def create_push_dataset(token: str, workspace_id: str, dataset_name: str) -> str:
    url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets"

    payload = {
        "name": dataset_name,
        "defaultMode": "Push",
        "tables": [
            {
                "name": SUMMARY_TABLE,
                "columns": [
                    {"name": "run_id", "dataType": "string"},
                    {"name": "input_samples", "dataType": "Int64"},
                    {"name": "normal_segment_samples", "dataType": "Int64"},
                    {"name": "event_segment_samples", "dataType": "Int64"},
                    {"name": "quantization_step_size", "dataType": "double"},
                    {"name": "best_snr_db", "dataType": "double"},
                    {"name": "best_ber", "dataType": "double"},
                    {"name": "reconstruction_rmse", "dataType": "double"},
                    {"name": "avg_temperature", "dataType": "double"},
                    {"name": "avg_pH", "dataType": "double"},
                    {"name": "avg_dissolved_oxygen", "dataType": "double"},
                    {"name": "min_dissolved_oxygen", "dataType": "double"},
                    {"name": "max_temperature", "dataType": "double"},
                    {"name": "low_do_events", "dataType": "Int64"},
                ],
            },
            {
                "name": BER_TABLE,
                "columns": [
                    {"name": "run_id", "dataType": "string"},
                    {"name": "snr_db", "dataType": "double"},
                    {"name": "ber", "dataType": "double"},
                ],
            },
            {
                "name": ALERTS_TABLE,
                "columns": [
                    {"name": "run_id", "dataType": "string"},
                    {"name": "alert_message", "dataType": "string"},
                ],
            },
        ],
    }

    response = powerbi_request("POST", url, token, payload)
    if response is None or "id" not in response:
        raise RuntimeError("Failed to create Power BI dataset.")

    return response["id"]


def push_rows(token: str, workspace_id: str, dataset_id: str, table_name: str, rows: list[dict]) -> None:
    if not rows:
        return

    url = f"https://api.powerbi.com/v1.0/myorg/groups/{workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows"
    payload = {"rows": rows}
    powerbi_request("POST", url, token, payload)


# ============================================================
# Parse simulation outputs
# ============================================================
def parse_summary_file(summary_path: Path) -> dict:
    text = summary_path.read_text(encoding="utf-8")

    def grab_number(label: str, as_int: bool = False, default=0):
        match = re.search(rf"{re.escape(label)}:\s*([-+]?\d*\.?\d+)", text)
        if not match:
            return default
        return int(float(match.group(1))) if as_int else float(match.group(1))

    summary = {
        "input_samples": grab_number("Input samples", as_int=True),
        "normal_segment_samples": grab_number("Normal segment samples", as_int=True),
        "event_segment_samples": grab_number("Event segment samples", as_int=True),
        "quantization_step_size": grab_number("Quantization step size"),
        "best_snr_db": grab_number("Best SNR"),
        "best_ber": grab_number("Final BER at best case"),
        "reconstruction_rmse": grab_number("Reconstruction RMSE"),
    }

    return summary


def parse_features_and_alerts(summary_path: Path) -> tuple[dict, list[str]]:
    text = summary_path.read_text(encoding="utf-8")

    features = {}
    alerts = []

    in_features = False
    in_alerts = False

    for line in text.splitlines():
        line = line.strip()

        if line == "Extracted Features:":
            in_features = True
            in_alerts = False
            continue

        if line == "Alerts:":
            in_features = False
            in_alerts = True
            continue

        if in_features and line.startswith("- "):
            item = line[2:]
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        features[key] = float(value)
                    else:
                        features[key] = int(value)
                except ValueError:
                    features[key] = value

        if in_alerts and line.startswith("- "):
            alerts.append(line[2:].strip())

    return features, alerts


def build_run_id() -> str:
    return pd.Timestamp.now().strftime("run_%Y%m%d_%H%M%S")


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError("results folder not found. Run the simulation first.")

    if not POWER_BI_ACCESS_TOKEN or "PASTE_YOUR_BEARER_TOKEN_HERE" in POWER_BI_ACCESS_TOKEN:
        raise ValueError("Set POWER_BI_ACCESS_TOKEN first.")

    if not WORKSPACE_ID or "PASTE_YOUR_WORKSPACE_ID_HERE" in WORKSPACE_ID:
        raise ValueError("Set WORKSPACE_ID first.")

    summary_path = RESULTS_DIR / "summary.txt"
    ber_path = RESULTS_DIR / "ber_results.csv"

    if not summary_path.exists():
        raise FileNotFoundError("results/summary.txt not found.")
    if not ber_path.exists():
        raise FileNotFoundError("results/ber_results.csv not found.")

    run_id = build_run_id()

    summary = parse_summary_file(summary_path)
    features, alerts = parse_features_and_alerts(summary_path)
    ber_df = pd.read_csv(ber_path)

    summary_row = {
        "run_id": run_id,
        "input_samples": summary.get("input_samples", 0),
        "normal_segment_samples": summary.get("normal_segment_samples", 0),
        "event_segment_samples": summary.get("event_segment_samples", 0),
        "quantization_step_size": summary.get("quantization_step_size", 0.0),
        "best_snr_db": summary.get("best_snr_db", 0.0),
        "best_ber": summary.get("best_ber", 0.0),
        "reconstruction_rmse": summary.get("reconstruction_rmse", 0.0),
        "avg_temperature": float(features.get("avg_temperature", 0.0)),
        "avg_pH": float(features.get("avg_pH", 0.0)),
        "avg_dissolved_oxygen": float(features.get("avg_dissolved_oxygen", 0.0)),
        "min_dissolved_oxygen": float(features.get("min_dissolved_oxygen", 0.0)),
        "max_temperature": float(features.get("max_temperature", 0.0)),
        "low_do_events": int(features.get("low_do_events", 0)),
    }

    ber_rows = []
    for _, row in ber_df.iterrows():
        ber_rows.append(
            {
                "run_id": run_id,
                "snr_db": float(row["snr_db"]),
                "ber": float(row["ber"]),
            }
        )

    alert_rows = [{"run_id": run_id, "alert_message": msg} for msg in alerts]

    dataset_id = DATASET_ID
    if not dataset_id:
        print("Creating Power BI push dataset...")
        dataset_id = create_push_dataset(POWER_BI_ACCESS_TOKEN, WORKSPACE_ID, DATASET_NAME)
        print(f"Created dataset: {dataset_id}")
    else:
        print(f"Using existing dataset: {dataset_id}")

    print("Pushing summary row...")
    push_rows(POWER_BI_ACCESS_TOKEN, WORKSPACE_ID, dataset_id, SUMMARY_TABLE, [summary_row])

    print("Pushing BER rows...")
    push_rows(POWER_BI_ACCESS_TOKEN, WORKSPACE_ID, dataset_id, BER_TABLE, ber_rows)

    print("Pushing alert rows...")
    push_rows(POWER_BI_ACCESS_TOKEN, WORKSPACE_ID, dataset_id, ALERTS_TABLE, alert_rows)

    print("Power BI push complete.")
    print(f"Dataset ID: {dataset_id}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
