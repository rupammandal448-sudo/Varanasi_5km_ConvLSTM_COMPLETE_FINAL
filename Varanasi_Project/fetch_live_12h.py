
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import time

import requests
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
GRID_FILE = ROOT / "master_grid_alignment.csv"
OUT_DIR = ROOT / "live_12h"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://api.open-meteo.com/v1/forecast"
TZ_NAME = "Asia/Kolkata"
LOCAL_TZ = ZoneInfo(TZ_NAME)

# Request variables supported by Open-Meteo's 15-minute API.
API_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
]

# Standardized output names.
FEATURE_MAP = {
    "temperature_2m": "temperature_2m_C",
    "relative_humidity_2m": "relative_humidity_pct",
    "dew_point_2m": "dewpoint_temperature_2m_C",
    "surface_pressure": "surface_pressure_hPa",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_10m_kmh",
    "wind_direction_10m": "wind_direction_10m_deg",
    "cloud_cover": "total_cloud_cover_pct",
    "shortwave_radiation": "shortwave_radiation_W_m2",
}

FEATURES = list(FEATURE_MAP.values())

# The previous 12 hours: 48 x 15-minute intervals.
INTERVAL_MINUTES = 15
SEQUENCE_LENGTH = 48

# Set a small delay between requests to reduce request bursts.
REQUEST_DELAY_SECONDS = 0.15
REQUEST_TIMEOUT_SECONDS = 45
MAX_RETRIES = 3


# ============================================================
# 2. VALIDATE MASTER GRID
# ============================================================

if not GRID_FILE.exists():
    raise FileNotFoundError(
        f"Master grid not found: {GRID_FILE}\n"
        "Place master_grid_alignment.csv beside this script."
    )

grid = pd.read_csv(GRID_FILE)

required_columns = {
    "grid_id",
    "grid_index",
    "grid_row",
    "grid_col",
    "latitude",
    "longitude",
}

missing_columns = required_columns - set(grid.columns)

if missing_columns:
    raise ValueError(
        f"Master grid is missing columns: {missing_columns}"
    )

grid["grid_id"] = grid["grid_id"].astype(str)
grid["latitude"] = pd.to_numeric(
    grid["latitude"], errors="raise"
)
grid["longitude"] = pd.to_numeric(
    grid["longitude"], errors="raise"
)
grid["grid_index"] = pd.to_numeric(
    grid["grid_index"], errors="raise"
).astype(int)
grid["grid_row"] = pd.to_numeric(
    grid["grid_row"], errors="raise"
).astype(int)
grid["grid_col"] = pd.to_numeric(
    grid["grid_col"], errors="raise"
).astype(int)

grid = grid.sort_values("grid_index").reset_index(drop=True)

if grid["grid_id"].duplicated().any():
    raise ValueError("Duplicate grid IDs found.")

if len(grid) != 77:
    raise ValueError(
        f"Expected 77 grid points, found {len(grid)}."
    )

if not (
    grid["grid_row"].between(0, 6).all()
    and grid["grid_col"].between(0, 10).all()
):
    raise ValueError(
        "Grid row/column indices must fit a 7x11 grid."
    )

if grid.duplicated(["grid_row", "grid_col"]).any():
    raise ValueError("Duplicate row/column locations found.")

print("=" * 65)
print("VARANASI 15-MINUTE WEATHER DATA EXTRACTOR")
print("=" * 65)
print("Grid points:", len(grid))


# ============================================================
# 3. DEFINE THE PREVIOUS 12-HOUR WINDOW
# ============================================================

now_utc = datetime.now(timezone.utc)

# End at the most recent completed 15-minute boundary.
end_utc = now_utc.replace(
    minute=(now_utc.minute // 15) * 15,
    second=0,
    microsecond=0,
)

start_utc = end_utc - timedelta(hours=12)

start_local = start_utc.astimezone(LOCAL_TZ).replace(
    tzinfo=None
)
end_local = end_utc.astimezone(LOCAL_TZ).replace(
    tzinfo=None
)

EXPECTED_TIMES = pd.date_range(
    start=pd.Timestamp(start_local),
    end=pd.Timestamp(end_local),
    freq="15min",
    inclusive="left",
)

if len(EXPECTED_TIMES) != SEQUENCE_LENGTH:
    raise RuntimeError(
        f"Expected 48 time slots, got {len(EXPECTED_TIMES)}."
    )

print("Timezone:", TZ_NAME)
print("Window start:", start_local)
print("Window end (exclusive):", end_local)
print("Expected timestamps per grid point:", len(EXPECTED_TIMES))


# ============================================================
# 4. FETCH 15-MINUTE DATA
# ============================================================

def fetch_point(latitude, longitude, grid_id):
    """
    Request the previous 12 hours of minutely_15 data.

    The API may not provide past 15-minute values for every
    variable/model/region. Missing timestamps are retained
    as NaN. No interpolation is performed.
    """

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "minutely_15": ",".join(API_VARIABLES),
        "timezone": TZ_NAME,
        "past_minutely_15": 48,
        "forecast_minutely_15": 0,
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                wait = 2 ** attempt
                print(
                    f"Rate limit for {grid_id}; "
                    f"retrying in {wait}s."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            payload = response.json()

            minute_data = payload.get("minutely_15", {})

            if not minute_data or "time" not in minute_data:
                raise ValueError(
                    "Response has no minutely_15 time series. "
                    f"Response keys: {list(payload.keys())}"
                )

            frame = pd.DataFrame(minute_data)

            frame["timestamp"] = pd.to_datetime(
                frame.pop("time"),
                errors="coerce",
            )

            # Select exactly the requested 12-hour window.
            frame = frame[
                (frame["timestamp"] >= pd.Timestamp(start_local))
                & (frame["timestamp"] < pd.Timestamp(end_local))
            ].copy()

            # Rename variables to standardized names.
            frame = frame.rename(columns=FEATURE_MAP)

            # Add grid metadata.
            frame["grid_id"] = str(grid_id)
            frame["latitude"] = float(latitude)
            frame["longitude"] = float(longitude)

            return frame

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as exc:
            last_error = exc

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed "
                f"for {grid_id}: {exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    print(f"Giving up on {grid_id}: {last_error}")
    return pd.DataFrame()


# ============================================================
# 5. DOWNLOAD ALL GRID POINTS
# ============================================================

all_frames = []

for position, point in grid.iterrows():
    grid_id = point["grid_id"]

    print(
        f"[{position + 1}/{len(grid)}] "
        f"Fetching {grid_id}..."
    )

    frame = fetch_point(
        latitude=point["latitude"],
        longitude=point["longitude"],
        grid_id=grid_id,
    )

    if not frame.empty:
        frame["grid_index"] = int(point["grid_index"])
        frame["grid_row"] = int(point["grid_row"])
        frame["grid_col"] = int(point["grid_col"])
        all_frames.append(frame)

    time.sleep(REQUEST_DELAY_SECONDS)


if not all_frames:
    raise RuntimeError(
        "No 15-minute records were retrieved. "
        "Check API support, internet access, and the response errors."
    )

data = pd.concat(all_frames, ignore_index=True)

data = data.drop_duplicates(
    subset=["timestamp", "grid_id"],
    keep="last",
)

data = data.sort_values(
    ["timestamp", "grid_index"]
).reset_index(drop=True)

data["source"] = "Open-Meteo minutely_15"

# Preserve the original API values and timestamps.
raw_csv = OUT_DIR / "operational_weather_last_12h.csv"
data.to_csv(raw_csv, index=False)

print("\nRaw observations saved:", raw_csv)
print("Rows received:", len(data))


# ============================================================
# 6. REINDEX TO A COMPLETE 48 x 77 GRID
# ============================================================

# Expected grid order is explicitly controlled by master_grid_alignment.csv.
grid_ids = grid["grid_id"].tolist()

full_index = pd.MultiIndex.from_product(
    [EXPECTED_TIMES, grid_ids],
    names=["timestamp", "grid_id"],
)

indexed = data.set_index(
    ["timestamp", "grid_id"]
)

# Reindex creates explicit NaNs for missing observations.
aligned = indexed.reindex(full_index).reset_index()

# Attach spatial metadata from the master grid.
metadata = grid.set_index("grid_id")

for column in [
    "grid_index",
    "grid_row",
    "grid_col",
    "latitude",
    "longitude",
]:
    aligned[column] = aligned["grid_id"].map(metadata[column])

aligned["source"] = aligned["source"].fillna(
    "missing"
)

# Ensure feature columns exist even if an API variable was absent.
for feature in FEATURES:
    if feature not in aligned.columns:
        aligned[feature] = np.nan

for feature in FEATURES:
    aligned[feature] = pd.to_numeric(
        aligned[feature], errors="coerce"
    )

aligned = aligned.sort_values(
    ["timestamp", "grid_index"]
).reset_index(drop=True)

aligned_csv = OUT_DIR / "operational_weather_grid_15min.csv"
aligned.to_csv(aligned_csv, index=False)

print("Aligned grid CSV:", aligned_csv)


# ============================================================
# 7. CREATE ARRAY AND VALIDITY MASK
# ============================================================

# Shape: [48 time steps, 77 grid points, 9 features].
array = np.full(
    (SEQUENCE_LENGTH, len(grid), len(FEATURES)),
    np.nan,
    dtype=np.float32,
)

valid_mask = np.zeros(
    array.shape,
    dtype=bool,
)

time_lookup = {
    timestamp: i
    for i, timestamp in enumerate(EXPECTED_TIMES)
}

grid_lookup = {
    grid_id: i
    for i, grid_id in enumerate(grid_ids)
}

for _, row in aligned.iterrows():
    ti = time_lookup.get(row["timestamp"])
    gi = grid_lookup.get(row["grid_id"])

    if ti is None or gi is None:
        continue

    for ci, feature in enumerate(FEATURES):
        value = row[feature]

        if pd.notna(value):
            array[ti, gi, ci] = float(value)
            valid_mask[ti, gi, ci] = True


# Save native point order plus spatial row/column metadata.
npz_path = OUT_DIR / "operational_weather_last_12h.npz"

np.savez_compressed(
    npz_path,
    data=array,
    valid_mask=valid_mask,
    timestamps=np.array(
        [str(t) for t in EXPECTED_TIMES]
    ),
    grid_ids=np.array(grid_ids),
    grid_index=grid["grid_index"].to_numpy(),
    grid_row=grid["grid_row"].to_numpy(),
    grid_col=grid["grid_col"].to_numpy(),
    latitude=grid["latitude"].to_numpy(),
    longitude=grid["longitude"].to_numpy(),
    channels=np.array(FEATURES),
)

print("\nSequence saved:", npz_path)
print("Sequence shape [time, grid point, channel]:", array.shape)
print("Observed feature values:", int(valid_mask.sum()))
print("Missing feature values:", int((~valid_mask).sum()))


# ============================================================
# 8. CREATE 7 x 11 SPATIAL ARRAYS FOR EACH TIMESTAMP
# ============================================================

# Each feature becomes [48, 7, 11].
# NaNs are preserved; this does not fabricate a complete map.

grid_arrays = {}

for ci, feature in enumerate(FEATURES):
    feature_array = np.full(
        (SEQUENCE_LENGTH, 7, 11),
        np.nan,
        dtype=np.float32,
    )

    for gi, point in grid.iterrows():
        row_index = int(point["grid_row"])
        col_index = int(point["grid_col"])

        feature_array[:, row_index, col_index] = array[
            :, gi, ci
        ]

    grid_arrays[feature] = feature_array

grid_npz = OUT_DIR / "operational_weather_maps_7x11.npz"

np.savez_compressed(
    grid_npz,
    timestamps=np.array(
        [str(t) for t in EXPECTED_TIMES]
    ),
    **grid_arrays,
)

print("7x11 maps saved:", grid_npz)


# ============================================================
# 9. DATA COMPLETENESS REPORT
# ============================================================

report = []

for ci, feature in enumerate(FEATURES):
    observed = int(valid_mask[:, :, ci].sum())
    total = SEQUENCE_LENGTH * len(grid)

    report.append({
        "feature": feature,
        "expected_values": total,
        "available_values": observed,
        "missing_values": total - observed,
        "coverage_pct": round(100 * observed / total, 2),
    })

report_df = pd.DataFrame(report)

report_path = OUT_DIR / "data_completeness_report.csv"
report_df.to_csv(report_path, index=False)

print("\nData completeness report:")
print(report_df.to_string(index=False))
print("\nSaved:", report_path)

print("\nProcessing finished.")
print("NOTE: This is operational weather, not exact ERA5 + IMERG.")
print("Do not pass it directly to the trained ConvLSTM.")
print("Check feature coverage and source compatibility first.")
