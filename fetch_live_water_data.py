import zipfile
import urllib.request
from pathlib import Path
import pandas as pd

OUTPUT_FILE = "water_quality_data.csv"
DOWNLOAD_DIR = Path("downloads")
EXTRACT_DIR = Path("extracted")

DOWNLOAD_DIR.mkdir(exist_ok=True)
EXTRACT_DIR.mkdir(exist_ok=True)

RAW_DATA_URLS = {
    "jan_mar_2024": "https://www.epa.nsw.gov.au/sites/default/files/raw-data-jan-mar-2024.zip",
    "apr_jun_2024": "https://www.epa.nsw.gov.au/sites/default/files/raw-data-apr-jun-2024.zip",
    "jul_sep_2024": "https://www.epa.nsw.gov.au/sites/default/files/raw-data-july-sep-2024.zip",
}

SELECTED_QUARTER = "jan_mar_2024"
START_DATE = "2024-01-01"
END_DATE = "2024-03-31"


def download_zip(url: str, out_path: Path) -> None:
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    out_path.write_bytes(data)
    print(f"Saved ZIP: {out_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted ZIP to: {extract_dir}")


def find_data_files(root: Path) -> list[Path]:
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(root.rglob(ext))
    return files


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    rename_map = {}

    for c in df.columns:
        c_low = str(c).strip().lower()

        if "time" in c_low or "date" in c_low:
            rename_map[c] = "timestamp"
        elif "temp" in c_low:
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
    if not all(c in df.columns for c in required):
        return None

    out = df[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
    out["pH"] = pd.to_numeric(out["pH"], errors="coerce")
    out["dissolved_oxygen"] = pd.to_numeric(out["dissolved_oxygen"], errors="coerce")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out if not out.empty else None


def load_candidate_file(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            attempts = [
                {"sep": ",", "engine": "python"},
                {"sep": ";", "engine": "python"},
                {"sep": "\t", "engine": "python"},
                {"sep": None, "engine": "python"},
            ]
            for opts in attempts:
                try:
                    df = pd.read_csv(path, **opts)
                    std = standardize_columns(df)
                    if std is not None:
                        print(f"Matched CSV file: {path.name}")
                        return std
                except Exception:
                    continue

        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
            std = standardize_columns(df)
            if std is not None:
                print(f"Matched Excel file: {path.name}")
                return std

    except Exception as e:
        print(f"Skipping {path.name}: {e}")

    return None


def filter_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    out = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    if SELECTED_QUARTER not in RAW_DATA_URLS:
        raise ValueError(f"Unknown quarter key: {SELECTED_QUARTER}")

    url = RAW_DATA_URLS[SELECTED_QUARTER]
    zip_path = DOWNLOAD_DIR / f"{SELECTED_QUARTER}.zip"

    download_zip(url, zip_path)
    extract_zip(zip_path, EXTRACT_DIR)

    candidates = find_data_files(EXTRACT_DIR)
    if not candidates:
        raise FileNotFoundError("No CSV/XLSX files found after extraction.")

    matched_df = None
    for file_path in candidates:
        matched_df = load_candidate_file(file_path)
        if matched_df is not None:
            break

    if matched_df is None:
        raise ValueError(
            "Could not find a usable file with timestamp, temperature, pH, and dissolved oxygen columns."
        )

    filtered_df = filter_date_range(matched_df, START_DATE, END_DATE)

    if filtered_df.empty:
        raise ValueError(
            f"No rows found between {START_DATE} and {END_DATE}. "
            "Try widening the date range or choosing another quarter."
        )

    filtered_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved filtered data to: {OUTPUT_FILE}")
    print(filtered_df.head())
    print(f"Rows saved: {len(filtered_df)}")


if __name__ == "__main__":
    main()