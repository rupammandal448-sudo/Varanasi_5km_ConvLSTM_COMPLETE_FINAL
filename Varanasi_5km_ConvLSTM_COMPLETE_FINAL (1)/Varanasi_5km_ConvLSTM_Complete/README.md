# Varanasi 5-km Nowcast and ConvLSTM Research Project

This repository contains two separate pipelines:

1. **Live website nowcast** (`web/`, `server.py`, `fetch_imd.py`): an IMD-station-driven 30-minute rain-event model served in the browser.
2. **Offline gridded ConvLSTM research pipeline** (`train_varanasi_convlstm.py`, `data/`, `outputs/`): a 5-km, 7×11 grid model using ERA5, IMERG and static geographic data.

These are not the same model. The ConvLSTM is not used by the live website because the live IMD feed does not provide the required 12-hour gridded sequence.

## IMPORTANT: extract the ZIP first

In Windows, right-click the ZIP and choose **Extract All**. Open the extracted folder. Do not open `index.html` from inside the compressed ZIP preview: the browser cannot reliably resolve the neighbouring CSS, JavaScript and image files there.

## Quick start (complete website)

1. Install Python 3.10+ (the `py` launcher should work).
2. Open PowerShell in the extracted project folder.
3. Install dependencies once:
   ```powershell
   py -m pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and enter your own `IMD_API_KEY`, `IMD_EMAIL`, and `IMD_PASSWORD`. Do not share `.env`. Without credentials the site still serves, but live IMD updates will not arrive.
5. Double-click `run_local.bat`. The browser opens at `http://127.0.0.1:8000`. Keep the terminal window open.

### Opening `index.html` directly

After extraction, opening the root `index.html` redirects to `web/index.html`. The Leaflet map interface can initialize without the model JSON requests, but browser security restrictions block local `fetch()` of the model and live JSON files. Therefore direct-file mode is for viewing the map only; use `run_local.bat` for forecasts. Basemap tiles require an internet connection to CARTO/OSM. If your network blocks map tile servers, the basemap may remain dark even though the local map interface is loaded.

## Live-data pipeline

`run_local.bat` → `server.py` → `fetch_imd.py` → `web/live.json` → `web/app.js` + `web/model.js` + `web/models.json` + `web/grid_static.json`.

- `fetch_imd.py --probe`: inspect current IMD payload field names and units (requires credentials/network).
- `fetch_imd.py --sample`: parse saved sample responses.
- `fetch_imd.py --mock`: generate synthetic demo data for UI testing only.
- `fetch_imd.py --loop 15`: poll IMD every 15 minutes.
- `http://127.0.0.1:8000/healthz`: server fetch health status.

IMD field timestamps, wind units and cumulative rainfall semantics are configurable assumptions documented near the top of `fetch_imd.py`. Verify them with `--probe` against the current API before relying on operational forecasts. The website labels forecasts as unofficial.

## Training the website ML model (optional)

```powershell
py -m pip install -r requirements.txt
py training/train_nowcast.py
```

This reads `data/` and `master_grid_alignment.csv`, holds out 2025 for testing, then exports `web/models.json` and `web/grid_static.json` plus reports under `training/reports/`. Training is computationally intensive and is not performed by `verify_project.py`.

## Training the gridded ConvLSTM (optional)

```powershell
py -m pip install -r requirements-research.txt
py train_varanasi_convlstm.py
```

This uses PyTorch and the ERA5/IMERG/static CSVs in `data/`. It writes the checkpoint, scalers, plots and metrics into `outputs/`. The supplied `outputs/best_varanasi_convlstm.pt` is the existing trained checkpoint. Full retraining can take substantial time and memory.

The separate `finetuned_varanasi_convlstm.py` is an experimental fine-tuning variant. `fetch_live_12h.py` is a separate Open-Meteo 12-hour operational-grid extractor and is not currently connected to the live website.

## Tests / audit

Run:
```powershell
py verify_project.py
py tests/test_fetch.py
node tests/test_nowcast.js
```

These validate required assets, JSON and grid structure, JavaScript syntax/model parity, and the IMD parser using mocked responses. They do **not** prove live IMD credentials/network access or reproduce a complete multi-hour model training run.

## Security

The archive intentionally does not include `.env`. Create it locally from `.env.example`. Never publish API credentials or upload `.env` to GitHub.

## Main directories

- `web/`: website, browser inference, JSON inputs, Leaflet runtime.
- `training/`: website model training script, notebook and reports.
- `data/`: original ERA5, IMERG, DEM, GHSL and WorldCover datasets. Retain for retraining.
- `outputs/`: ConvLSTM checkpoint, scalers, metrics, plots and training history.
- `live_12h/`: previously extracted 12-hour operational grid products.
- `imd_live_data/`: saved IMD sample payloads for parser checks.
- `tests/`: fetcher and JavaScript model tests.
