import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path


# ============================================================
# TELE 523 - End-to-End Smart Water Quality Telemetry Simulation
# Domain: Smart Water Quality Monitoring
# Pipeline:
# Local Dataset -> Preprocess -> Quantize -> PCM -> Modulate
# -> Channel -> Demodulate -> Decode -> BER -> Feature Extraction
# -> Monitoring
# ============================================================

DATA_FILE = "water_quality_data.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

USE_DEMO_FALLBACK = True


# ------------------------------------------------------------
# 1. LOAD LOCAL DATA OR FALL BACK TO DEMO
# ------------------------------------------------------------
def generate_demo_data() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=400, freq="30min")
    t = np.arange(len(timestamps))

    temperature = 24 + 2 * np.sin(2 * np.pi * t / 48) + 0.25 * np.random.randn(len(t))
    pH = 7.2 + 0.15 * np.sin(2 * np.pi * t / 48 + 0.5) + 0.03 * np.random.randn(len(t))
    dissolved_oxygen = 6 + 1.0 * np.sin(2 * np.pi * t / 48 + 1.2) + 0.2 * np.random.randn(len(t))

    # Inject an anomaly/event window
    dissolved_oxygen[220:260] -= 2.5

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pH": pH,
            "dissolved_oxygen": dissolved_oxygen,
        }
    )


def load_local_or_demo_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        if USE_DEMO_FALLBACK:
            print("Local CSV not found. Using generated demo dataset...")
            return generate_demo_data()
        raise FileNotFoundError(f"Could not find {file_path}")

    print(f"Found local file: {file_path}")
    print("Trying to read local dataset...")

    # Catch cases where the CSV accidentally contains Python code
    try:
        sample_text = path.read_text(encoding="utf-8", errors="ignore")[:2000].lower()
        code_markers = [
            "import pandas",
            "def ",
            "raw_data_url",
            "output_file",
            "if __name__",
            "pd.dataframe",
        ]
        if any(marker in sample_text for marker in code_markers):
            if USE_DEMO_FALLBACK:
                print("WARNING: CSV appears to contain Python code, not telemetry data.")
                print("Falling back to generated demo dataset...")
                return generate_demo_data()
            raise ValueError(f"{file_path} contains code instead of CSV data.")
    except UnicodeDecodeError:
        pass

    read_attempts = [
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": None, "engine": "python"},
    ]

    df = None
    last_error = None

    for opts in read_attempts:
        try:
            trial_df = pd.read_csv(path, **opts)
            if trial_df.shape[1] > 1 and trial_df.shape[0] > 0:
                df = trial_df
                print(f"Loaded local CSV using options: {opts}")
                break
        except Exception as e:
            last_error = e

    if df is None:
        if USE_DEMO_FALLBACK:
            print(f"WARNING: Could not parse local CSV. Last issue: {last_error}")
            print("Falling back to generated demo dataset...")
            return generate_demo_data()
        raise ValueError(
            f"Could not parse {file_path} as a normal table.\n"
            f"Last parsing error: {last_error}"
        )

    df.columns = [str(c).strip() for c in df.columns]

    possible_time_cols = [
        "timestamp", "time", "date", "datetime",
        "Timestamp", "Time", "Date", "Datetime"
    ]
    found_time = None
    for c in possible_time_cols:
        if c in df.columns:
            found_time = c
            break

    if found_time is None:
        if USE_DEMO_FALLBACK:
            print("WARNING: No timestamp column found in local CSV.")
            print("Falling back to generated demo dataset...")
            return generate_demo_data()
        raise ValueError(f"No timestamp column found. Available columns: {list(df.columns)}")

    df = df.rename(columns={found_time: "timestamp"})

    rename_map = {}
    for c in df.columns:
        c_low = str(c).strip().lower()

        if "temp" in c_low:
            rename_map[c] = "temperature"
        elif c_low == "ph" or "ph_" in c_low or "ph " in c_low or "(ph)" in c_low:
            rename_map[c] = "pH"
        elif (
            "dissolved oxygen" in c_low
            or c_low in {"do", "do_mg_l", "do mg/l", "d.o.", "do_mgl"}
        ):
            rename_map[c] = "dissolved_oxygen"

    df = df.rename(columns=rename_map)

    required = ["timestamp", "temperature", "pH", "dissolved_oxygen"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if USE_DEMO_FALLBACK:
            print(f"WARNING: Missing required columns: {missing}")
            print("Falling back to generated demo dataset...")
            return generate_demo_data()
        raise ValueError(
            f"Missing required columns after reading file: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["pH"] = pd.to_numeric(df["pH"], errors="coerce")
    df["dissolved_oxygen"] = pd.to_numeric(df["dissolved_oxygen"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        if USE_DEMO_FALLBACK:
            print("WARNING: Local CSV became empty after cleaning.")
            print("Falling back to generated demo dataset...")
            return generate_demo_data()
        raise ValueError("Dataset is empty after cleaning.")

    print("Local dataset loaded successfully.")
    return df


# ------------------------------------------------------------
# 2. SIGNAL PROCESSING SECTION
# ------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset="timestamp")

    out.loc[(out["temperature"] < -5) | (out["temperature"] > 60), "temperature"] = np.nan
    out.loc[(out["pH"] < 0) | (out["pH"] > 14), "pH"] = np.nan
    out.loc[(out["dissolved_oxygen"] < 0) | (out["dissolved_oxygen"] > 20), "dissolved_oxygen"] = np.nan

    out = out.set_index("timestamp").interpolate(method="time").ffill().bfill().reset_index()
    return out


def resample_uniform(df: pd.DataFrame, interval: str = "30min") -> pd.DataFrame:
    out = df.copy().set_index("timestamp").resample(interval).mean(numeric_only=True)
    out = out.interpolate(method="time").ffill().bfill().reset_index()
    return out


def normalize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        mn, mx = out[c].min(), out[c].max()
        out[f"{c}_norm"] = 0.0 if mx == mn else (out[c] - mn) / (mx - mn)
    return out


def segment_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_segment = df[df["dissolved_oxygen"] >= 4.0].copy()
    event_segment = df[df["dissolved_oxygen"] < 4.0].copy()
    return normal_segment, event_segment


def plot_before_after(raw_df: pd.DataFrame, processed_df: pd.DataFrame, column: str, fname: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(raw_df["timestamp"], raw_df[column], label="Raw")
    plt.plot(processed_df["timestamp"], processed_df[column], label="Processed")
    plt.title(f"Before vs After Processing: {column}")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


def plot_psd(df: pd.DataFrame, column: str, fname: str) -> None:
    if len(df) < 4:
        print(f"Skipping PSD for {column}: not enough samples.")
        return

    dt = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
    fs = 1.0 / dt if dt > 0 else 1.0

    x = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 4:
        print(f"Skipping PSD for {column}: signal became too short after cleaning.")
        return

    nperseg = min(128, len(x))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg)

    plt.figure(figsize=(8, 4))
    plt.semilogy(f, pxx)
    plt.title(f"PSD of {column}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# 3. MODULATION / CHANNEL / DEMODULATION SECTION
# ------------------------------------------------------------
def nrz_linecode(bits: np.ndarray) -> np.ndarray:
    return np.where(bits > 0, 1.0, -1.0)


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    return np.where(bits > 0, 1.0, -1.0)


def add_awgn(signal_in: np.ndarray, snr_db: float) -> np.ndarray:
    power_signal = np.mean(signal_in ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power_signal / snr_linear if snr_linear != 0 else power_signal
    noise = np.sqrt(noise_power) * np.random.randn(*signal_in.shape)
    return signal_in + noise


def bpsk_demodulate(rx: np.ndarray) -> np.ndarray:
    return (rx >= 0).astype(np.uint8)


def ber(original_bits: np.ndarray, received_bits: np.ndarray) -> float:
    n = min(len(original_bits), len(received_bits))
    if n == 0:
        return 0.0
    return float(np.mean(original_bits[:n] != received_bits[:n]))


# ------------------------------------------------------------
# 4. DIGITAL TELEMETRY SECTION
# ------------------------------------------------------------
def linear_quantize(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    levels = 2 ** bits
    xmin, xmax = np.min(x), np.max(x)

    if xmax == xmin:
        q = np.zeros_like(x, dtype=np.uint8)
        recon = np.full_like(x, fill_value=xmin, dtype=float)
        return q, recon, 0.0, float(xmin), float(xmax)

    step = (xmax - xmin) / (levels - 1)
    q = np.round((x - xmin) / step).astype(np.uint8)
    recon = xmin + q * step
    return q, recon, step, float(xmin), float(xmax)


def pcm_encode(q: np.ndarray, bits: int = 8) -> np.ndarray:
    q = q.astype(np.uint8)
    return np.unpackbits(q.reshape(-1, 1), axis=1)[:, -bits:].reshape(-1)


def pcm_decode(bitstream: np.ndarray, bits: int = 8) -> np.ndarray:
    usable_len = len(bitstream) - (len(bitstream) % bits)
    trimmed = bitstream[:usable_len].reshape(-1, bits)

    padded = np.zeros((trimmed.shape[0], 8), dtype=np.uint8)
    padded[:, -bits:] = trimmed

    return np.packbits(padded, axis=1).flatten()


def reconstruct_from_quantized(decoded_q: np.ndarray, xmin: float, xmax: float, bits: int = 8) -> np.ndarray:
    if xmax == xmin:
        return np.full_like(decoded_q, fill_value=xmin, dtype=float)

    step = (xmax - xmin) / ((2 ** bits) - 1)
    return xmin + decoded_q * step


def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original = np.asarray(original, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    n = min(len(original), len(reconstructed))
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean((original[:n] - reconstructed[:n]) ** 2)))


def plot_quantization(original: np.ndarray, quantized_recon: np.ndarray, title: str, fname: str) -> None:
    original = np.asarray(original, dtype=float)
    quantized_recon = np.asarray(quantized_recon, dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original")
    plt.step(np.arange(len(quantized_recon)), quantized_recon, where="mid", label="Quantized/Reconstructed")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


def plot_quant_error(original: np.ndarray, recon: np.ndarray, title: str, fname: str) -> None:
    original = np.asarray(original, dtype=float)
    recon = np.asarray(recon, dtype=float)

    n = min(len(original), len(recon))
    err = original[:n] - recon[:n]

    plt.figure(figsize=(10, 4))
    plt.plot(err)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


def plot_ber_curve(snr_values: list[float], ber_values: list[float], fname: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.semilogy(snr_values, ber_values, marker="o")
    plt.title("BER vs SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# 5. MONITORING / FEATURE EXTRACTION SECTION
# ------------------------------------------------------------
def extract_features(df: pd.DataFrame) -> dict:
    return {
        "avg_temperature": float(df["temperature"].mean()),
        "avg_pH": float(df["pH"].mean()),
        "avg_dissolved_oxygen": float(df["dissolved_oxygen"].mean()),
        "min_dissolved_oxygen": float(df["dissolved_oxygen"].min()),
        "max_temperature": float(df["temperature"].max()),
        "low_do_events": int((df["dissolved_oxygen"] < 4.0).sum()),
    }


def generate_alerts(df: pd.DataFrame) -> list[str]:
    alerts = []

    if (df["dissolved_oxygen"] < 4.0).any():
        alerts.append("WARNING: Dissolved oxygen dropped below 4 mg/L.")
    if (df["pH"] < 6.5).any() or (df["pH"] > 8.5).any():
        alerts.append("WARNING: pH exceeded safe monitoring range.")
    if (df["temperature"] > 30).any():
        alerts.append("WARNING: High water temperature detected.")

    if not alerts:
        alerts.append("System status normal. No critical threshold exceeded.")

    return alerts


def plot_monitoring_dashboard(df: pd.DataFrame, fname: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["temperature"], label="Temperature")
    plt.plot(df["timestamp"], df["pH"], label="pH")
    plt.plot(df["timestamp"], df["dissolved_oxygen"], label="Dissolved Oxygen")
    plt.axhline(4.0, linestyle="--", label="DO Alert Threshold")
    plt.title("Monitoring Output")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=300)
    plt.close()


# ------------------------------------------------------------
# 6. FULL CONNECTED PIPELINE
# ------------------------------------------------------------
def main() -> None:
    print("\n=== TELE 523 END-TO-END SMART WATER QUALITY TELEMETRY SIMULATION ===")

    # Stage 1: Dataset
    raw_df = load_local_or_demo_data(DATA_FILE)

    # Stage 2: Signal Processing
    clean_df = clean_data(raw_df)
    proc_df = resample_uniform(clean_df, "30min")
    proc_df = normalize_columns(proc_df, ["temperature", "pH", "dissolved_oxygen"])
    normal_segment, event_segment = segment_conditions(proc_df)

    # Signal processing plots
    plot_before_after(raw_df, proc_df, "temperature", "before_after_temperature.png")
    plot_before_after(raw_df, proc_df, "pH", "before_after_pH.png")
    plot_before_after(raw_df, proc_df, "dissolved_oxygen", "before_after_do.png")
    plot_psd(proc_df, "temperature", "psd_temperature.png")
    plot_psd(proc_df, "dissolved_oxygen", "psd_do.png")

    # Use dissolved oxygen as the main demonstration channel
    tx_signal = proc_df["dissolved_oxygen_norm"].to_numpy(dtype=float)

    # Stage 3: Quantization + PCM
    q_levels, q_recon, step, xmin, xmax = linear_quantize(tx_signal, bits=8)
    bitstream = pcm_encode(q_levels, bits=8)
    linecoded = nrz_linecode(bitstream)

    plot_quantization(
        tx_signal,
        q_recon,
        "Original vs Quantized Signal (Normalized DO)",
        "original_vs_quantized_do.png",
    )
    plot_quant_error(
        tx_signal,
        q_recon,
        "Quantization Error (Normalized DO)",
        "quantization_error_do.png",
    )

    # Stage 4: Modulation + AWGN + Demodulation
    snr_values = [0, 2, 4, 6, 8, 10, 12]
    ber_values = []
    best_received_bits = None
    best_snr = None
    best_ber = np.inf

    modulated = bpsk_modulate(bitstream)

    for snr in snr_values:
        rx_channel = add_awgn(modulated, snr)
        rx_bits = bpsk_demodulate(rx_channel)
        cur_ber = ber(bitstream, rx_bits)
        ber_values.append(cur_ber)

        if cur_ber < best_ber:
            best_ber = cur_ber
            best_received_bits = rx_bits.copy()
            best_snr = snr

    plot_ber_curve(snr_values, ber_values, "ber_vs_snr.png")

    # Stage 5: Decode + Reconstruct
    decoded_q = pcm_decode(best_received_bits, bits=8)
    decoded_q = decoded_q[:len(q_levels)]
    reconstructed_signal = reconstruct_from_quantized(decoded_q, xmin, xmax, bits=8)
    rmse_value = calculate_rmse(tx_signal, reconstructed_signal)

    plt.figure(figsize=(10, 4))
    plt.plot(tx_signal, label="Original Signal")
    plt.plot(reconstructed_signal, label=f"Reconstructed Signal (Best SNR={best_snr} dB)")
    plt.title("Reconstructed vs Original Signal")
    plt.xlabel("Sample")
    plt.ylabel("Normalized DO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reconstructed_vs_original.png", dpi=300)
    plt.close()

    # Stage 6: Monitoring
    final_df = proc_df.copy()
    final_df["recovered_dissolved_oxygen_norm"] = np.nan
    final_df.loc[:len(reconstructed_signal) - 1, "recovered_dissolved_oxygen_norm"] = reconstructed_signal

    monitoring_df = proc_df[["timestamp", "temperature", "pH", "dissolved_oxygen"]].copy()
    features = extract_features(monitoring_df)
    alerts = generate_alerts(monitoring_df)
    plot_monitoring_dashboard(monitoring_df, "monitoring_dashboard.png")

    # Save outputs
    proc_df.to_csv(RESULTS_DIR / "processed_data.csv", index=False)
    final_df.to_csv(RESULTS_DIR / "final_monitored_data.csv", index=False)
    normal_segment.to_csv(RESULTS_DIR / "normal_segment.csv", index=False)
    event_segment.to_csv(RESULTS_DIR / "event_segment.csv", index=False)
    pd.DataFrame({"bitstream": bitstream}).to_csv(RESULTS_DIR / "pcm_bitstream.csv", index=False)
    pd.DataFrame({"nrz_linecoded": linecoded}).to_csv(RESULTS_DIR / "linecoded_signal.csv", index=False)
    pd.DataFrame({"snr_db": snr_values, "ber": ber_values}).to_csv(RESULTS_DIR / "ber_results.csv", index=False)

    with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("TELE 523 End-to-End Simulation Summary\n")
        f.write("====================================\n")
        f.write(f"Input samples: {len(proc_df)}\n")
        f.write(f"Normal segment samples: {len(normal_segment)}\n")
        f.write(f"Event segment samples: {len(event_segment)}\n")
        f.write(f"Quantization step size: {step}\n")
        f.write(f"Best SNR: {best_snr} dB\n")
        f.write(f"Final BER at best case: {best_ber:.6f}\n")
        f.write(f"Reconstruction RMSE: {rmse_value:.6f}\n\n")
        f.write("Extracted Features:\n")
        for k, v in features.items():
            f.write(f"- {k}: {v}\n")
        f.write("\nAlerts:\n")
        for a in alerts:
            f.write(f"- {a}\n")

    print("\nSimulation complete.")
    print(f"Results saved in: {RESULTS_DIR.resolve()}")
    print(f"Best SNR: {best_snr} dB")
    print(f"Best BER: {best_ber:.6f}")
    print(f"RMSE: {rmse_value:.6f}")
    print("\nExtracted Features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    print("\nAlerts:")
    for a in alerts:
        print(f"  {a}")


if __name__ == "__main__":
    main()