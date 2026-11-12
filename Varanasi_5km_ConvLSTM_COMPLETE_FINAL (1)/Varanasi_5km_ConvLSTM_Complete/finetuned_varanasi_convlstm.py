"""
VARANASI 5-km STATIC-CONDITIONED CONVLSTM
==========================================

Complete local/Google-Colab training pipeline.

DATA
----
Dynamic:
    ERA5 2024 Q1-Q4
    ERA5 2025 Q1-Q4
    IMERG 2024 Q1-Q4
    IMERG 2025 Q1-Q4

Static:
    WorldCover
    GHSL
    DEM

GRID
----
77 spatial cells arranged as a 7 x 11 grid.

INPUT
-----
Previous 12 hours:
    24 x 30-minute timesteps
    10 ERA5 variables
    + lagged/current IMERG precipitation

TARGET
------
IMERG precipitation 30 minutes ahead.

SPLIT
-----
Training   : 2024-01-01 to 2024-09-30
Validation : 2024-10-01 to 2024-12-31
Test       : 2025-01-01 to 2025-12-31

IMPORTANT
---------
The CSV files live in:

    project/
        train_varanasi_convlstm.py
        data/
            *.csv

This script automatically uses the data/ folder.

OUTPUTS
-------
outputs/
    00_model_architecture.png
    01_spatial_grid.png
    02_static_feature_maps.png
    03_training_curves.png
    04_rain_event_f1.png
    05_actual_vs_predicted_map.png
    06_actual_vs_predicted_scatter.png
    training_history.csv
    final_2025_test_metrics.csv
    best_varanasi_convlstm.pt
    dynamic_scaler.npz
    static_scaler.npz
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

GRID_H = 7
GRID_W = 11
N_CELLS = GRID_H * GRID_W

INPUT_STEPS = 24          # 12 hours at 30-min resolution
FORECAST_HORIZON = 1      # 30 minutes

# Fine-tuned training configuration.
# The data paths and file addresses below are intentionally unchanged.
BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN1 = 64
HIDDEN2 = 64
STATIC_HIDDEN = 32

# Early stopping prevents over-training once validation performance
# stops improving.
EARLY_STOPPING_PATIENCE = 10

NUM_WORKERS = 0           # safest on Windows
PIN_MEMORY = torch.cuda.is_available()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = torch.cuda.is_available()


# ============================================================
# 2. DATASET FILES
# ============================================================

ERA5_FILES = {
    2024: [
        "ERA5_Varanasi_5km_2024_Q1.csv",
        "ERA5_Varanasi_5km_2024_Q2.csv",
        "ERA5_Varanasi_5km_2024_Q3.csv",
        "ERA5_Varanasi_5km_2024_Q4.csv",
    ],

    2025: [
        "ERA5_Varanasi_5km_2025_Q1(1).csv",
        "ERA5_Varanasi_5km_2025_Q2(1).csv",
        "ERA5_Varanasi_5km_2025_Q3(1).csv",
        "ERA5_Varanasi_5km_2025_Q4(1).csv",
    ],
}


IMERG_FILES = {
    2024: [
        "IMERG_Varanasi_5km_2024_Q1_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2024_Q2_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2024_Q3_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2024_Q4_ERA5_aligned.csv",
    ],

    2025: [
        "IMERG_Varanasi_5km_2025_Q1_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2025_Q2_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2025_Q3_ERA5_aligned.csv",
        "IMERG_Varanasi_5km_2025_Q4_ERA5_aligned.csv",
    ],
}


STATIC_FILES = {
    "worldcover":
        "WorldCover_5km_IMERG_grid_aligned.csv",

    "ghsl":
        "GHSL_5km_IMERG_grid_aligned.csv",

    "dem":
        "DEM_5km_IMERG_grid_aligned.csv",
}


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. ERA5 VARIABLE NORMALIZATION
# ============================================================

ERA5_ALIASES = {

    "boundary_layer_height": [
        "boundary_layer_height_m",
        "boundary_layer_height",
    ],

    "dewpoint_temperature_2m": [
        "dewpoint_temperature_2m_K",
        "dewpoint_temperature_2m",
    ],

    "skin_temperature": [
        "skin_temperature_K",
        "skin_temperature",
    ],

    "surface_pressure": [
        "surface_pressure_Pa",
        "surface_pressure",
    ],

    "surface_solar_radiation": [
        "surface_solar_radiation_J_m2",
        "surface_solar_radiation_downwards",
    ],

    "temperature_2m": [
        "temperature_2m_K",
        "temperature_2m",
    ],

    "total_cloud_cover": [
        "total_cloud_cover",
    ],

    "total_precipitation": [
        "total_precipitation_m",
        "total_precipitation",
    ],

    "u_wind_10m": [
        "u_wind_10m_ms",
        "u_component_of_wind_10m",
    ],

    "v_wind_10m": [
        "v_wind_10m_ms",
        "v_component_of_wind_10m",
    ],
}


ERA5_FEATURES = list(
    ERA5_ALIASES.keys()
)


DYNAMIC_FEATURES = (
    ERA5_FEATURES +
    ["imerg_precip_log1p"]
)


# ============================================================
# 5. FILE VALIDATION
# ============================================================

def validate_files():

    required = []

    for year in [2024, 2025]:

        required.extend(
            ERA5_FILES[year]
        )

        required.extend(
            IMERG_FILES[year]
        )

    required.extend(
        STATIC_FILES.values()
    )

    missing = [
        f for f in required
        if not (DATA_DIR / f).exists()
    ]

    if missing:

        print("\nMissing files:")

        for f in missing:
            print("  ", f)

        raise FileNotFoundError(
            "\nPut all CSV files inside:\n"
            f"{DATA_DIR}\n"
        )

    print(
        f"Found all {len(required)} required CSV files."
    )


# ============================================================
# 6. GRID CONSTRUCTION
# ============================================================

def numeric_grid_id(value):

    match = re.search(
        r"(\d+)",
        str(value)
    )

    if match is None:
        raise ValueError(
            f"Invalid grid_id: {value}"
        )

    return int(
        match.group(1)
    )


def make_grid_order():

    """
    Build the Varanasi 5-km spatial grid.

    The dataset contains exactly 77 cells:
        G0000 ... G0076

    The grid is stored in COLUMN-MAJOR order:

        G0000 G0007 G0014 ... G0070
        G0001 G0008 G0015 ... G0071
        G0002 G0009 G0016 ... G0072
        G0003 G0010 G0017 ... G0073
        G0004 G0011 G0018 ... G0074
        G0005 G0012 G0019 ... G0075
        G0006 G0013 G0020 ... G0076

    Therefore:

        row = grid_number % 7
        col = grid_number // 7
    """

    # ---------------------------------------------------------
    # 1. Read one ERA5 file as the reference grid
    # ---------------------------------------------------------

    reference_file = DATA_DIR / ERA5_FILES[2024][0]

    print("\nReading reference grid:")
    print(reference_file)

    df = pd.read_csv(
        reference_file,
        usecols=[
            "grid_id",
            "latitude",
            "longitude"
        ]
    )

    print(
        f"\nTotal rows in reference file: {len(df)}"
    )

    # ---------------------------------------------------------
    # 2. Keep only one record for each spatial cell
    # ---------------------------------------------------------

    grid = (
        df[
            [
                "grid_id",
                "latitude",
                "longitude"
            ]
        ]
        .drop_duplicates(
            subset="grid_id"
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique grid cells found: {len(grid)}"
    )

    # ---------------------------------------------------------
    # 3. We need exactly 77 cells
    # ---------------------------------------------------------

    if len(grid) != 77:

        raise ValueError(
            f"Expected 77 unique grid cells, "
            f"but found {len(grid)}."
        )

    # ---------------------------------------------------------
    # 4. Convert G0000 -> 0, G0001 -> 1, etc.
    # ---------------------------------------------------------

    grid["grid_num"] = (
        grid["grid_id"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )

    # ---------------------------------------------------------
    # 5. Verify IDs are exactly 0 ... 76
    # ---------------------------------------------------------

    expected_ids = set(range(77))

    actual_ids = set(
        grid["grid_num"].tolist()
    )

    if actual_ids != expected_ids:

        missing = sorted(
            expected_ids - actual_ids
        )

        extra = sorted(
            actual_ids - expected_ids
        )

        raise ValueError(
            "Grid IDs are incorrect.\n"
            f"Missing IDs: {missing}\n"
            f"Extra IDs: {extra}"
        )

    # ---------------------------------------------------------
    # 6. Construct 7 x 11 grid
    #
    # COLUMN-MAJOR:
    #
    # row = grid_num % 7
    # col = grid_num // 7
    # ---------------------------------------------------------

    grid["grid_row"] = (
        grid["grid_num"] % GRID_H
    )

    grid["grid_col"] = (
        grid["grid_num"] // GRID_H
    )

    # ---------------------------------------------------------
    # 7. Verify dimensions
    # ---------------------------------------------------------

    if (
        grid["grid_row"].min() != 0
        or
        grid["grid_row"].max() != 6
    ):

        raise ValueError(
            "Invalid grid rows."
        )

    if (
        grid["grid_col"].min() != 0
        or
        grid["grid_col"].max() != 10
    ):

        raise ValueError(
            "Invalid grid columns."
        )

    # ---------------------------------------------------------
    # 8. Verify every position occurs exactly once
    # ---------------------------------------------------------

    positions = list(
        zip(
            grid["grid_row"],
            grid["grid_col"]
        )
    )

    if len(set(positions)) != 77:

        raise ValueError(
            "Duplicate grid positions detected."
        )

    # ---------------------------------------------------------
    # 9. Sort in row-major order
    # ---------------------------------------------------------

    grid = (
        grid
        .sort_values(
            ["grid_row", "grid_col"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 10. Print resulting grid
    # ---------------------------------------------------------

    print("\n========================================")
    print("VARANASI SPATIAL GRID")
    print("========================================")

    print(f"Rows  : {GRID_H}")
    print(f"Cols  : {GRID_W}")
    print(f"Cells : {len(grid)}")

    print("\nGrid layout:\n")

    grid_matrix = np.empty(
        (GRID_H, GRID_W),
        dtype=object
    )

    for _, item in grid.iterrows():

        r = int(item["grid_row"])
        c = int(item["grid_col"])

        grid_matrix[r, c] = item["grid_id"]

    for r in range(GRID_H):

        print(
            " | ".join(
                str(grid_matrix[r, c])
                for c in range(GRID_W)
            )
        )

    print(
        "\nGrid construction SUCCESS."
    )

    return grid


# ============================================================
# 7. STATIC FEATURES
# ============================================================

def load_static_features(grid):

    wc = pd.read_csv(
        DATA_DIR /
        STATIC_FILES["worldcover"]
    )

    ghsl = pd.read_csv(
        DATA_DIR /
        STATIC_FILES["ghsl"]
    )

    dem = pd.read_csv(
        DATA_DIR /
        STATIC_FILES["dem"]
    )

    wc_cols = [
        "bare_fraction",
        "builtup_fraction",
        "cropland_fraction",
        "grass_fraction",
        "mangrove_fraction",
        "moss_lichen_fraction",
        "shrub_fraction",
        "snow_ice_fraction",
        "tree_fraction",
        "water_fraction",
        "wetland_fraction",
    ]

    ghsl_cols = [
        "builtup_fraction_mean",
        "max_building_height_m",
        "mean_building_height_m",
        "nres_fraction_mean",
    ]

    dem_cols = [
        "elevation_mean",
        "elevation_std",
        "max_slope",
        "mean_slope",
    ]

    def prepare(
        df,
        columns,
        name
    ):

        missing = [
            c for c in columns
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{name} missing columns: {missing}"
            )

        out = df[
            ["grid_id"] + columns
        ].drop_duplicates(
            "grid_id"
        )

        if len(out) != N_CELLS:

            raise ValueError(
                f"{name}: expected {N_CELLS} "
                f"unique grid cells, found {len(out)}."
            )

        # Align using grid_id, NOT file row order.
        out = (
            grid[["grid_id"]]
            .merge(
                out,
                on="grid_id",
                how="left",
                validate="one_to_one"
            )
        )

        if out[columns].isna().any().any():

            raise ValueError(
                f"{name} has missing grid cells."
            )

        return out

    wc = prepare(
        wc,
        wc_cols,
        "WorldCover"
    )

    ghsl = prepare(
        ghsl,
        ghsl_cols,
        "GHSL"
    )

    dem = prepare(
        dem,
        dem_cols,
        "DEM"
    )

    static_df = pd.DataFrame(
        {
            "grid_id":
                grid.grid_id
        }
    )

    for c in wc_cols:
        static_df[c] = wc[c]

    for c in ghsl_cols:
        static_df[c] = ghsl[c]

    for c in dem_cols:
        static_df[c] = dem[c]

    static_cols = [
        c for c in static_df.columns
        if c != "grid_id"
    ]

    # Drop spatially constant channels.
    static_cols = [
        c for c in static_cols
        if static_df[c].nunique() > 1
    ]

    static = static_df[
        static_cols
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(static).all():

        raise ValueError(
            "Static data contains NaN or Inf."
        )

    print("\nStatic channels:")

    for c in static_cols:
        print("  ", c)

    print(
        f"\nNumber of static channels: "
        f"{len(static_cols)}"
    )

    return static, static_cols


# ============================================================
# 8. ERA5 COLUMN NORMALIZATION
# ============================================================

def normalize_era5_columns(df):

    rename = {}

    for canonical, aliases in ERA5_ALIASES.items():

        found = None

        for col in aliases:

            if col in df.columns:

                found = col
                break

        if found is None:

            raise ValueError(
                f"Could not find ERA5 variable "
                f"{canonical}.\n"
                f"Accepted names: {aliases}"
            )

        rename[found] = canonical

    df = df.rename(
        columns=rename
    )

    # ERA5 precipitation is stored in metres.
    df["total_precipitation"] = (
        pd.to_numeric(
            df["total_precipitation"],
            errors="coerce"
        ) * 1000.0
    )

    for c in ERA5_FEATURES:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return df


# ============================================================
# 9. LOAD ONE YEAR
# ============================================================

def load_year(
    year,
    grid
):

    print(
        f"\n{'=' * 20}"
        f" Loading {year} "
        f"{'=' * 20}"
    )

    # --------------------------------------------------------
    # ERA5
    # --------------------------------------------------------

    era_parts = []

    for fname in ERA5_FILES[year]:

        path = DATA_DIR / fname

        print(
            "ERA5:",
            fname
        )

        df = pd.read_csv(
            path
        )

        df = normalize_era5_columns(
            df
        )

        df = df[
            [
                "grid_id",
                "timestamp",
            ] + ERA5_FEATURES
        ]

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        era_parts.append(df)

    era = pd.concat(
        era_parts,
        ignore_index=True
    )

    era = era.dropna(
        subset=[
            "grid_id",
            "timestamp"
        ]
    )

    # There should be one ERA5 row per
    # timestamp + grid cell.
    era = era.drop_duplicates(
        subset=[
            "timestamp",
            "grid_id"
        ],
        keep="first"
    )

    print(
        "ERA5 rows:",
        len(era)
    )

    print(
        "ERA5 unique grid cells:",
        era.grid_id.nunique()
    )

    # --------------------------------------------------------
    # IMERG
    # --------------------------------------------------------

    imerg_parts = []

    for fname in IMERG_FILES[year]:

        path = DATA_DIR / fname

        print(
            "IMERG:",
            fname
        )

        df = pd.read_csv(
            path,
            usecols=[
                "grid_id",
                "timestamp",
                "precipitation_mm_hr",
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df["precipitation_mm_hr"] = pd.to_numeric(
            df["precipitation_mm_hr"],
            errors="coerce"
        )

        imerg_parts.append(df)

    imerg = pd.concat(
        imerg_parts,
        ignore_index=True
    )

    imerg = imerg.dropna(
        subset=[
            "grid_id",
            "timestamp"
        ]
    )

    imerg = imerg.drop_duplicates(
        subset=[
            "timestamp",
            "grid_id"
        ],
        keep="first"
    )

    print(
        "IMERG rows:",
        len(imerg)
    )

    print(
        "IMERG unique grid cells:",
        imerg.grid_id.nunique()
    )

    # --------------------------------------------------------
    # Check spatial grid
    # --------------------------------------------------------

    expected_ids = set(
        grid.grid_id
    )

    era_ids = set(
        era.grid_id
    )

    imerg_ids = set(
        imerg.grid_id
    )

    if era_ids != expected_ids:

        raise ValueError(
            f"{year} ERA5 grid IDs do not match "
            "the reference grid."
        )

    if imerg_ids != expected_ids:

        raise ValueError(
            f"{year} IMERG grid IDs do not match "
            "the reference grid."
        )

    # --------------------------------------------------------
    # IMERG timestamps become the common timeline.
    # --------------------------------------------------------

    times = (
        pd.DatetimeIndex(
            sorted(
                imerg.timestamp.unique()
            )
        )
    )

    print(
        "IMERG time steps:",
        len(times)
    )

    # Verify 30-minute spacing.
    diffs = (
        pd.Series(times)
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )

    unusual = diffs[
        diffs != 30
    ]

    if len(unusual):

        print(
            "Warning:",
            len(unusual),
            "non-30-minute intervals found."
        )

    # --------------------------------------------------------
    # Align ERA5 hourly data to IMERG 30-min timeline.
    #
    # We use the most recent ERA5 observation at or before
    # each IMERG timestamp. This avoids using future ERA5 data.
    #
    # A 12-hour tolerance handles known short missing ERA5
    # blocks while still preventing arbitrarily stale values.
    # --------------------------------------------------------

    aligned_parts = []

    for gid in grid.grid_id:

        left = imerg[
            imerg.grid_id == gid
        ].sort_values(
            "timestamp"
        )

        right = era[
            era.grid_id == gid
        ].sort_values(
            "timestamp"
        )

        merged = pd.merge_asof(
            left,
            right[
                ["timestamp"] + ERA5_FEATURES
            ],
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(
                "12h"
            )
        )

        aligned_parts.append(
            merged
        )

    merged = pd.concat(
        aligned_parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Validate dynamic features.
    # --------------------------------------------------------

    missing_era = (
        merged[ERA5_FEATURES]
        .isna()
        .any(axis=1)
    )

    if missing_era.any():

        bad_times = (
            merged.loc[
                missing_era,
                "timestamp"
            ]
            .drop_duplicates()
            .sort_values()
        )

        raise ValueError(
            f"{year}: ERA5 alignment still has "
            f"{len(bad_times)} timestamps with missing "
            "features.\n"
            f"First problematic times:\n"
            f"{bad_times.head(10).tolist()}"
        )

    if merged[
        "precipitation_mm_hr"
    ].isna().any():

        raise ValueError(
            f"{year}: IMERG target contains missing values."
        )

    # --------------------------------------------------------
    # Reconstruct exact spatial tensor order.
    # --------------------------------------------------------

    grid_order = {
        gid: i
        for i, gid in enumerate(
            grid.grid_id
        )
    }

    merged["grid_position"] = (
        merged.grid_id.map(
            grid_order
        )
    )

    merged = merged.sort_values(
        [
            "timestamp",
            "grid_position"
        ]
    )

    # --------------------------------------------------------
    # Check every timestamp has 77 cells.
    # --------------------------------------------------------

    counts = (
        merged
        .groupby("timestamp")
        .grid_id
        .nunique()
    )

    bad_counts = counts[
        counts != N_CELLS
    ]

    if len(bad_counts):

        raise ValueError(
            f"{year}: {len(bad_counts)} timestamps "
            "do not contain all 77 cells."
        )

    # --------------------------------------------------------
    # Feature transformation
    # --------------------------------------------------------

    merged[
        "imerg_precip_log1p"
    ] = np.log1p(
        np.clip(
            merged[
                "precipitation_mm_hr"
            ].to_numpy(),
            a_min=0.0,
            a_max=None
        )
    )

    # --------------------------------------------------------
    # Tensor conversion
    #
    # T,H,W,C -> T,C,H,W
    # --------------------------------------------------------

    n_time = len(times)

    feature_values = merged[
        DYNAMIC_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    feature_values = feature_values.reshape(
        n_time,
        GRID_H,
        GRID_W,
        len(DYNAMIC_FEATURES)
    )

    X = np.transpose(
        feature_values,
        (0, 3, 1, 2)
    ).copy()

    rain = merged[
        "precipitation_mm_hr"
    ].to_numpy(
        dtype=np.float32
    )

    rain = rain.reshape(
        n_time,
        GRID_H,
        GRID_W
    )

    # Target is one 30-minute step ahead.
    y = np.roll(
        rain,
        -FORECAST_HORIZON,
        axis=0
    )

    valid_n = (
        n_time -
        FORECAST_HORIZON
    )

    X = X[:valid_n]

    y = y[:valid_n]

    times = times[:valid_n]

    print(
        f"{year}: X shape = {X.shape}"
    )

    print(
        f"{year}: y shape = {y.shape}"
    )

    return X, y, times


# ============================================================
# 10. SCALERS
# ============================================================

class DynamicScaler:

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, X):

        self.mean = np.mean(
            X,
            axis=(0, 2, 3),
            keepdims=True
        ).astype(
            np.float32
        )

        self.std = np.std(
            X,
            axis=(0, 2, 3),
            keepdims=True
        ).astype(
            np.float32
        )

        self.std[
            self.std < 1e-6
        ] = 1.0

    def transform(self, X):

        return (
            (
                X -
                self.mean
            ) /
            self.std
        ).astype(
            np.float32
        )

    def save(self, path):

        np.savez(
            path,
            mean=self.mean,
            std=self.std
        )


class StaticScaler:

    def __init__(self):

        self.mean = None
        self.std = None

    def fit(self, X):

        self.mean = X.mean(
            axis=0,
            keepdims=True
        ).astype(
            np.float32
        )

        self.std = X.std(
            axis=0,
            keepdims=True
        ).astype(
            np.float32
        )

        self.std[
            self.std < 1e-6
        ] = 1.0

    def transform(self, X):

        return (
            (
                X -
                self.mean
            ) /
            self.std
        ).astype(
            np.float32
        )

    def save(self, path):

        np.savez(
            path,
            mean=self.mean,
            std=self.std
        )


# ============================================================
# 11. SEQUENCE DATASET
# ============================================================

class SequenceDataset(
    Dataset
):

    def __init__(
        self,
        X,
        y,
        timestamps,
        static,
        sample_indices,
        input_steps=24
    ):

        self.X = torch.from_numpy(
            X
        )

        self.y = torch.from_numpy(
            y
        )

        self.timestamps = timestamps

        self.sample_indices = np.asarray(
            sample_indices,
            dtype=np.int64
        )

        self.input_steps = input_steps

        self.static = torch.from_numpy(
            static.T.reshape(
                static.shape[1],
                GRID_H,
                GRID_W
            )
        )

    def __len__(self):

        return len(
            self.sample_indices
        )

    def __getitem__(
        self,
        dataset_index
    ):

        target_index = int(
            self.sample_indices[
                dataset_index
            ]
        )

        start = (
            target_index -
            self.input_steps
        )

        x = self.X[
            start:target_index
        ]

        # T,C,H,W -> C,T,H,W
        x = x.permute(
            1, 0, 2, 3
        )

        target = self.y[
            target_index
        ]

        timestamp = str(
            self.timestamps[
                target_index
            ]
        )

        return (
            x,
            self.static,
            target,
            timestamp
        )


# ============================================================
# 12. CONVLSTM CELL
# ============================================================

class ConvLSTMCell(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        padding = (
            kernel_size // 2
        )

        self.conv = nn.Conv2d(
            input_dim +
            hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x,
        state
    ):

        h, c = state

        combined = torch.cat(
            [x, h],
            dim=1
        )

        gates = self.conv(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = (
            f * c +
            i * g
        )

        h_next = (
            o *
            torch.tanh(
                c_next
            )
        )

        return (
            h_next,
            c_next
        )

    def init_state(
        self,
        batch,
        height,
        width,
        device
    ):

        h = torch.zeros(
            batch,
            self.hidden_dim,
            height,
            width,
            device=device
        )

        c = torch.zeros(
            batch,
            self.hidden_dim,
            height,
            width,
            device=device
        )

        return h, c


class ConvLSTMLayer(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3
    ):

        super().__init__()

        self.cell = ConvLSTMCell(
            input_dim,
            hidden_dim,
            kernel_size
        )

    def forward(
        self,
        x
    ):
        """
        x:
            B,T,C,H,W

        returns:
            B,T,HIDDEN,H,W
        """

        B, T, C, H, W = x.shape

        h, c = (
            self.cell.init_state(
                B,
                H,
                W,
                x.device
            )
        )

        outputs = []

        for t in range(T):

            h, c = self.cell(
                x[:, t],
                (h, c)
            )

            outputs.append(
                h
            )

        return torch.stack(
            outputs,
            dim=1
        )


# ============================================================
# 13. STATIC-CONDITIONED CONVLSTM MODEL
# ============================================================

class StaticConditionedConvLSTM(
    nn.Module
):

    def __init__(
        self,
        dynamic_channels,
        static_channels,
        hidden1=64,
        hidden2=64,
        static_hidden=32
    ):

        super().__init__()

        # Per-time-step CNN encoder.
        self.input_encoder = nn.Sequential(

            nn.Conv2d(
                dynamic_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # Temporal encoder.
        self.convlstm1 = ConvLSTMLayer(
            32,
            hidden1
        )

        self.convlstm2 = ConvLSTMLayer(
            hidden1,
            hidden2
        )

        # Static spatial encoder.
        self.static_encoder = nn.Sequential(

            nn.Conv2d(
                static_channels,
                static_hidden,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                static_hidden
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                static_hidden,
                static_hidden,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # Dynamic + static fusion.
        self.fusion = nn.Sequential(

            nn.Conv2d(
                hidden2 +
                static_hidden,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # Output rainfall.
        self.output_head = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )
        )

    def forward(
        self,
        x,
        static
    ):
        """
        x:
            B,C,T,H,W

        static:
            B,S,H,W

        output:
            B,H,W
            log1p rainfall
        """

        B, C, T, H, W = x.shape

        # Rearrange for efficient per-frame CNN.
        z = x.permute(
            0,
            2,
            1,
            3,
            4
        )

        z = z.reshape(
            B * T,
            C,
            H,
            W
        )

        z = self.input_encoder(
            z
        )

        z = z.reshape(
            B,
            T,
            32,
            H,
            W
        )

        # First ConvLSTM.
        z = self.convlstm1(
            z
        )

        # Second ConvLSTM.
        z = self.convlstm2(
            z
        )

        # Last temporal representation.
        temporal = z[:, -1]

        # Static representation.
        static_features = (
            self.static_encoder(
                static
            )
        )

        # Fuse.
        fused = torch.cat(
            [
                temporal,
                static_features
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        output = self.output_head(
            fused
        )

        return output[:, 0]


# ============================================================
# 14. RAINFALL LOSS
# ============================================================

class RainfallLoss(
    nn.Module
):

    def __init__(
        self,
        wet_threshold=0.1,
        wet_weight=5.0
    ):

        super().__init__()

        self.wet_threshold = (
            wet_threshold
        )

        self.wet_weight = (
            wet_weight
        )

    def forward(
        self,
        pred_log,
        target
    ):

        target = torch.clamp(
            target,
            min=0.0
        )

        target_log = torch.log1p(
            target
        )

        wet = (
            target >=
            self.wet_threshold
        ).float()

        weight = (
            1.0 +
            self.wet_weight *
            wet
        )

        loss = F.smooth_l1_loss(
            pred_log,
            target_log,
            reduction="none"
        )

        return (
            loss *
            weight
        ).mean()


# ============================================================
# 15. TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for (
        x,
        static,
        target,
        _
    ) in progress:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        static = static.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                pred_log = model(
                    x,
                    static
                )

                loss = criterion(
                    pred_log,
                    target
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            pred_log = model(
                x,
                static
            )

            loss = criterion(
                pred_log,
                target
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

        total_loss += (
            loss.item() *
            x.size(0)
        )

        total_samples += (
            x.size(0)
        )

        progress.set_postfix(
            loss=f"{loss.item():.5f}"
        )

    return (
        total_loss /
        max(total_samples, 1)
    )


# ============================================================
# 16. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    event_threshold=0.1
):
    model.eval()

    total_abs = 0.0
    total_sq = 0.0
    total_count = 0

    # For R²
    total_target = 0.0
    total_target_sq = 0.0

    tp = 0
    fp = 0
    fn = 0

    for (
        x,
        static,
        target,
        _
    ) in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        static = static.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        pred_log = model(
            x,
            static
        )

        # Convert log prediction back to mm/hr
        pred = torch.expm1(
            pred_log
        )

        pred = torch.clamp(
            pred,
            min=0.0
        )

        # ----------------------------------------------------
        # Regression errors
        # ----------------------------------------------------

        diff = (
            pred -
            target
        )

        total_abs += (
            diff.abs()
            .sum()
            .item()
        )

        total_sq += (
            diff.pow(2)
            .sum()
            .item()
        )

        total_count += (
            target.numel()
        )

        # ----------------------------------------------------
        # R² statistics
        # ----------------------------------------------------

        total_target += (
            target.sum()
            .item()
        )

        total_target_sq += (
            target.pow(2)
            .sum()
            .item()
        )

        # ----------------------------------------------------
        # Rainfall event detection
        # ----------------------------------------------------

        pred_event = (
            pred >= event_threshold
        )

        true_event = (
            target >= event_threshold
        )

        tp += (
            pred_event &
            true_event
        ).sum().item()

        fp += (
            pred_event &
            ~true_event
        ).sum().item()

        fn += (
            ~pred_event &
            true_event
        ).sum().item()

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = (
        total_abs /
        max(total_count, 1)
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    mse = (
        total_sq /
        max(total_count, 1)
    )

    rmse = np.sqrt(mse)

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    # SST = Σ(y - mean(y))²
    mean_target = (
        total_target /
        max(total_count, 1)
    )

    sst = (
        total_target_sq -
        total_count * mean_target ** 2
    )

    # SSE = Σ(y - prediction)²
    sse = total_sq

    if sst > 1e-12:
        r2 = 1.0 - (
            sse / sst
        )
    else:
        r2 = float("nan")

    # --------------------------------------------------------
    # Rain event metrics
    # --------------------------------------------------------

    precision = (
        tp /
        max(tp + fp, 1)
    )

    recall = (
        tp /
        max(tp + fn, 1)
    )

    f1 = (
        2 *
        precision *
        recall /
        max(
            precision + recall,
            1e-8
        )
    )

    return {
        "MAE_mm_hr": mae,
        "RMSE_mm_hr": rmse,
        "R2": r2,
        "RainEvent_Precision": precision,
        "RainEvent_Recall": recall,
        "RainEvent_F1": f1
    }

# ============================================================
# 17. VISUALIZATION
# ============================================================

def save_grid_visualization(
    grid
):

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    x = grid.grid_col.to_numpy()
    y = grid.grid_row.to_numpy()

    ax.scatter(
        x,
        y,
        s=260,
        edgecolors="black"
    )

    for _, row in grid.iterrows():

        ax.text(
            row.grid_col,
            row.grid_row,
            str(row.grid_id),
            ha="center",
            va="center",
            fontsize=7
        )

    ax.set_title(
        "Varanasi 5-km Spatial Grid (7 × 11)"
    )

    ax.set_xlabel(
        "Grid column"
    )

    ax.set_ylabel(
        "Grid row"
    )

    ax.set_xticks(
        range(GRID_W)
    )

    ax.set_yticks(
        range(GRID_H)
    )

    ax.invert_yaxis()

    ax.set_aspect(
        "equal"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "01_spatial_grid.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_static_maps(
    static,
    names
):

    n = len(names)

    cols = 4

    rows = int(
        np.ceil(
            n / cols
        )
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            14,
            3.3 * rows
        )
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for i, name in enumerate(names):

        arr = static[
            :,
            i
        ].reshape(
            GRID_H,
            GRID_W
        )

        im = axes[i].imshow(
            arr
        )

        axes[i].set_title(
            name
        )

        axes[i].set_xticks([])
        axes[i].set_yticks([])

        fig.colorbar(
            im,
            ax=axes[i],
            fraction=0.046,
            pad=0.04
        )

    for i in range(
        n,
        len(axes)
    ):

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Static Spatial Features",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "02_static_feature_maps.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_architecture_diagram():

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.set_xlim(
        0,
        15
    )

    ax.set_ylim(
        0,
        7
    )

    ax.axis(
        "off"
    )

    boxes = [

        (
            0.3,
            4.5,
            2.2,
            1.2,
            "ERA5 + IMERG\n24 × C × 7 × 11"
        ),

        (
            3.1,
            4.5,
            2.2,
            1.2,
            "CNN Feature\nEncoder"
        ),

        (
            5.9,
            4.5,
            2.2,
            1.2,
            "ConvLSTM × 2\nTemporal Encoder"
        ),

        (
            8.7,
            4.5,
            2.2,
            1.2,
            "Static Fusion"
        ),

        (
            11.5,
            4.5,
            2.2,
            1.2,
            "CNN Decoder"
        ),

        (
            11.5,
            1.7,
            2.2,
            1.2,
            "30-min Rainfall\n7 × 11"
        ),

        (
            6.0,
            1.7,
            2.2,
            1.2,
            "WorldCover\nGHSL + DEM"
        )
    ]

    for (
        x,
        y,
        w,
        h,
        label
    ) in boxes:

        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            fill=True,
            alpha=0.12
        )

        ax.add_patch(
            rect
        )

        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10
        )

    arrows = [

        ((2.5, 5.1), (3.1, 5.1)),
        ((5.3, 5.1), (5.9, 5.1)),
        ((8.1, 5.1), (8.7, 5.1)),
        ((10.9, 5.1), (11.5, 5.1)),
        ((12.6, 4.5), (12.6, 2.9)),
        ((8.2, 2.3), (8.7, 4.5)),
    ]

    for start, end in arrows:

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 2
            }
        )

    ax.text(
        7.5,
        6.35,
        "Static-Conditioned ConvLSTM Encoder–Decoder",
        ha="center",
        fontsize=16,
        fontweight="bold"
    )

    ax.text(
        7.5,
        0.55,
        "12-hour history → 30-minute-ahead 5-km rainfall prediction",
        ha="center",
        fontsize=11
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "00_model_architecture.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_training_curves(
    history
):

    if not history:
        return

    df = pd.DataFrame(
        history
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        df.epoch,
        df.train_loss,
        marker="o",
        label="Training loss"
    )

    ax.plot(
        df.epoch,
        df.RMSE_mm_hr,
        marker="s",
        label="Validation RMSE"
    )

    ax.set_title(
        "Training and Validation"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss / RMSE"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "03_training_curves.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        df.epoch,
        df.RainEvent_F1,
        marker="o"
    )

    ax.set_title(
        "Validation Rain Event F1"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "F1"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "04_rain_event_f1.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


@torch.no_grad()
def save_prediction_map(
    model,
    dataset,
    sample_index=0
):

    model.eval()

    (
        x,
        static,
        target,
        timestamp
    ) = dataset[
        sample_index
    ]

    x = x.unsqueeze(
        0
    ).to(DEVICE)

    static = static.unsqueeze(
        0
    ).to(DEVICE)

    pred_log = model(
        x,
        static
    )

    prediction = torch.expm1(
        pred_log
    )

    prediction = torch.clamp(
        prediction,
        min=0.0
    )

    prediction = (
        prediction[0]
        .cpu()
        .numpy()
    )

    target = target.numpy()

    imerg_channel = (
        DYNAMIC_FEATURES.index(
            "imerg_precip_log1p"
        )
    )

    previous = (
        torch.expm1(
            dataset.X[
                int(
                    dataset.sample_indices[
                        sample_index
                    ]
                ) -
                1,
                imerg_channel
            ]
        )
        .numpy()
    )

    error = np.abs(
        prediction -
        target
    )

    vmax = max(
        np.percentile(
            target,
            99
        ),
        np.percentile(
            prediction,
            99
        ),
        0.1
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17, 4)
    )

    maps = [
        (
            previous,
            "Previous IMERG\n(t)"
        ),
        (
            target,
            "Actual IMERG\n(t + 30 min)"
        ),
        (
            prediction,
            "ConvLSTM prediction\n(t + 30 min)"
        ),
        (
            error,
            "Absolute error"
        )
    ]

    for i, (
        arr,
        title
    ) in enumerate(maps):

        if i < 3:

            im = axes[i].imshow(
                arr,
                vmin=0,
                vmax=vmax
            )

        else:

            im = axes[i].imshow(
                arr
            )

        axes[i].set_title(
            title
        )

        axes[i].set_xlabel(
            "Grid column"
        )

        axes[i].set_ylabel(
            "Grid row"
        )

        fig.colorbar(
            im,
            ax=axes[i],
            fraction=0.046,
            pad=0.04,
            label="mm/hr"
        )

    fig.suptitle(
        f"30-minute Rainfall Prediction\nTarget: {timestamp}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "05_actual_vs_predicted_map.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


@torch.no_grad()
def save_prediction_scatter(
    model,
    loader,
    max_points=30000
):

    model.eval()

    actual_values = []
    predicted_values = []

    collected = 0

    for (
        x,
        static,
        target,
        _
    ) in loader:

        x = x.to(
            DEVICE
        )

        static = static.to(
            DEVICE
        )

        pred_log = model(
            x,
            static
        )

        prediction = torch.expm1(
            pred_log
        )

        prediction = torch.clamp(
            prediction,
            min=0.0
        )

        actual_values.append(
            target.numpy()
            .reshape(-1)
        )

        predicted_values.append(
            prediction.cpu()
            .numpy()
            .reshape(-1)
        )

        collected += target.numel()

        if collected >= max_points:
            break

    actual = np.concatenate(
        actual_values
    )[:max_points]

    predicted = np.concatenate(
        predicted_values
    )[:max_points]

    upper = max(
        np.percentile(
            actual,
            99.5
        ),
        np.percentile(
            predicted,
            99.5
        ),
        0.1
    )

    mask = (
        (actual <= upper) &
        (predicted <= upper)
    )

    actual = actual[
        mask
    ]

    predicted = predicted[
        mask
    ]

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        actual,
        predicted,
        s=7,
        alpha=0.2
    )

    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction"
    )

    ax.set_xlim(
        0,
        upper
    )

    ax.set_ylim(
        0,
        upper
    )

    ax.set_xlabel(
        "Actual IMERG rainfall (mm/hr)"
    )

    ax.set_ylabel(
        "Predicted rainfall (mm/hr)"
    )

    ax.set_title(
        "Actual vs Predicted Rainfall — 2025 Test"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        "06_actual_vs_predicted_scatter.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print(
        "\n" +
        "=" * 60
    )

    print(
        "VARANASI 5-KM CONVLSTM TRAINING"
    )

    print(
        "=" * 60
    )

    print(
        "\nProject directory:"
    )

    print(
        PROJECT_DIR
    )

    print(
        "\nData directory:"
    )

    print(
        DATA_DIR
    )

    print(
        "\nDevice:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    validate_files()

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid = make_grid_order()

    save_grid_visualization(
        grid
    )

    # --------------------------------------------------------
    # Static data
    # --------------------------------------------------------

    static, static_names = (
        load_static_features(
            grid
        )
    )

    save_static_maps(
        static,
        static_names
    )

    # --------------------------------------------------------
    # Dynamic data
    # --------------------------------------------------------

    X2024, y2024, t2024 = (
        load_year(
            2024,
            grid
        )
    )

    X2025, y2025, t2025 = (
        load_year(
            2025,
            grid
        )
    )

    # --------------------------------------------------------
    # Combine 2024 + 2025.
    #
    # This allows the first 2025 test windows to use late-2024
    # observations as historical context, which is realistic.
    # --------------------------------------------------------

    X_all = np.concatenate(
        [
            X2024,
            X2025
        ],
        axis=0
    )

    y_all = np.concatenate(
        [
            y2024,
            y2025
        ],
        axis=0
    )

    t_all = pd.DatetimeIndex(
        np.concatenate(
            [
                t2024,
                t2025
            ]
        )
    )

    print(
        "\nCombined dynamic tensor:"
    )

    print(
        "X:",
        X_all.shape
    )

    print(
        "y:",
        y_all.shape
    )

    # --------------------------------------------------------
    # Every training sample needs 24 historical steps.
    #
    # target index = i
    # input = i-24 ... i-1
    # --------------------------------------------------------

    first_valid_target = (
        INPUT_STEPS
    )

    candidate_indices = np.arange(
        first_valid_target,
        len(t_all)
    )

    target_times = (
        t_all[
            candidate_indices
        ]
    )

    train_indices = (
        candidate_indices[
            target_times <
            pd.Timestamp(
                "2024-10-01"
            )
        ]
    )

    val_indices = (
        candidate_indices[
            (
                target_times >=
                pd.Timestamp(
                    "2024-10-01"
                )
            ) &
            (
                target_times <
                pd.Timestamp(
                    "2025-01-01"
                )
            )
        ]
    )

    test_indices = (
        candidate_indices[
            (
                target_times >=
                pd.Timestamp(
                    "2025-01-01"
                )
            )
        ]
    )

    print(
        "\nWindow counts:"
    )

    print(
        "Train:",
        len(train_indices)
    )

    print(
        "Validation:",
        len(val_indices)
    )

    print(
        "Test:",
        len(test_indices)
    )

    # --------------------------------------------------------
    # Fit scalers ONLY on training period.
    # --------------------------------------------------------

    train_time_mask = (
        t_all <
        pd.Timestamp(
            "2024-10-01"
        )
    )

    dynamic_scaler = (
        DynamicScaler()
    )

    dynamic_scaler.fit(
        X_all[
            train_time_mask
        ]
    )

    X_all = (
        dynamic_scaler.transform(
            X_all
        )
    )

    dynamic_scaler.save(
        OUTPUT_DIR /
        "dynamic_scaler.npz"
    )

    static_scaler = (
        StaticScaler()
    )

    static_scaler.fit(
        static
    )

    static_scaled = (
        static_scaler.transform(
            static
        )
    )

    static_scaler.save(
        OUTPUT_DIR /
        "static_scaler.npz"
    )

    np.save(
        OUTPUT_DIR /
        "static_feature_names.npy",
        np.array(
            static_names
        )
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = SequenceDataset(
        X_all,
        y_all,
        t_all,
        static_scaled,
        train_indices,
        INPUT_STEPS
    )

    val_dataset = SequenceDataset(
        X_all,
        y_all,
        t_all,
        static_scaled,
        val_indices,
        INPUT_STEPS
    )

    test_dataset = SequenceDataset(
        X_all,
        y_all,
        t_all,
        static_scaled,
        test_indices,
        INPUT_STEPS
    )

    print(
        "\nDataset sizes:"
    )

    print(
        "Train:",
        len(train_dataset)
    )

    print(
        "Validation:",
        len(val_dataset)
    )

    print(
        "Test:",
        len(test_dataset)
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        StaticConditionedConvLSTM(
            dynamic_channels=len(
                DYNAMIC_FEATURES
            ),
            static_channels=static_scaled.shape[1],
            hidden1=HIDDEN1,
            hidden2=HIDDEN2,
            static_hidden=STATIC_HIDDEN
        )
        .to(DEVICE)
    )

    save_architecture_diagram()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = RainfallLoss(
        # Increased wet-event emphasis because the original model was
        # conservative and missed a substantial number of rain events.
        wet_threshold=0.1,
        wet_weight=5.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    if USE_AMP:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_rmse = float(
        "inf"
    )

    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler
            )
        )

        metrics = evaluate(
            model,
            val_loader
        )

        scheduler.step(
            metrics[
                "RMSE_mm_hr"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": current_lr,
            **metrics
        }

        history.append(
            record
        )

        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Val MAE: "
            f"{metrics['MAE_mm_hr']:.6f} mm/hr"
        )

        print(
            f"Val RMSE: "
            f"{metrics['RMSE_mm_hr']:.6f} mm/hr"
        )

        print(
            f"Val R²: "
            f"{metrics['R2']:.6f}"
        )

        print(
            f"Val Rain F1: "
            f"{metrics['RainEvent_F1']:.4f}"
        )

        if (
            metrics["RMSE_mm_hr"]
            < best_rmse
        ):

            best_rmse = (
                metrics[
                    "RMSE_mm_hr"
                ]
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_val_rmse":
                        best_rmse,
                    "dynamic_features":
                        DYNAMIC_FEATURES,
                    "static_features":
                        static_names,
                    "grid_height":
                        GRID_H,
                    "grid_width":
                        GRID_W,
                    "input_steps":
                        INPUT_STEPS,
                    "forecast_horizon":
                        FORECAST_HORIZON,
                },
                OUTPUT_DIR /
                "best_varanasi_convlstm.pt"
            )

            print(
                "✓ Saved best model"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"No validation RMSE improvement for "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epoch(s)."
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "\nEarly stopping triggered."
                )
                break

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            OUTPUT_DIR /
            "training_history.csv",
            index=False
        )

        save_training_curves(
            history
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        OUTPUT_DIR /
        "best_varanasi_convlstm.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Final 2025 test
    # --------------------------------------------------------

    test_metrics = evaluate(
        model,
        test_loader,
        event_threshold=0.1
    )

    # Additional rainfall-event diagnostics at multiple thresholds.
    # These do not affect training/checkpoint selection.
    threshold_metrics = {}
    for threshold in [0.1, 0.2, 0.5, 1.0]:
        m = evaluate(
            model,
            test_loader,
            event_threshold=threshold
        )
        threshold_metrics[threshold] = m

    print(
        "\n" +
        "=" * 60
    )

    print(
        "FINAL 2025 TEST RESULTS"
    )

    print(
        "=" * 60
    )

    for key, value in (
        test_metrics.items()
    ):

        print(
            f"{key}: {value:.6f}"
        )

    print("\nRain-event threshold diagnostics:")
    for threshold, metrics_at_threshold in threshold_metrics.items():
        print(
            f"  {threshold:.1f} mm/hr -> "
            f"Precision={metrics_at_threshold['RainEvent_Precision']:.4f}, "
            f"Recall={metrics_at_threshold['RainEvent_Recall']:.4f}, "
            f"F1={metrics_at_threshold['RainEvent_F1']:.4f}"
        )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(
        OUTPUT_DIR /
        "final_2025_test_metrics.csv",
        index=False
    )

    # Save event-threshold diagnostics separately.
    threshold_rows = []
    for threshold, metrics_at_threshold in threshold_metrics.items():
        threshold_rows.append(
            {
                "Event_Threshold_mm_hr": threshold,
                "Precision": metrics_at_threshold["RainEvent_Precision"],
                "Recall": metrics_at_threshold["RainEvent_Recall"],
                "F1": metrics_at_threshold["RainEvent_F1"],
            }
        )

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        OUTPUT_DIR /
        "2025_event_threshold_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final visualizations
    # --------------------------------------------------------

    if len(test_dataset) > 0:

        save_prediction_map(
            model,
            test_dataset,
            sample_index=0
        )

        save_prediction_scatter(
            model,
            test_loader
        )

    save_training_curves(
        history
    )

    print(
        "\nBest epoch:",
        best_epoch
    )

    print(
        "Best validation RMSE:",
        best_rmse
    )
    

    print(
        "\nAll outputs saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nTRAINING COMPLETE."
    )


# ============================================================
# 19. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
