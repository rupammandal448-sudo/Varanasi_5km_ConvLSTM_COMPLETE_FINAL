# Project file guide

This project contains two related parts:
1. A browser-based 30-minute Varanasi nowcast website (`web/`).
2. The offline ConvLSTM research/training pipeline (`train_varanasi_convlstm.py`, `data/`, `outputs/`).

## Start here

**Extract the ZIP first.** Do not open HTML from inside the Windows compressed-folder preview. Direct `file://` mode is for the map interface only; run `run_local.bat` for model forecasts.

| File | Purpose |
|---|---|
| `index.html` | Convenience entry point that redirects to `web/index.html`. |
| `run_local.bat` | Checks core Python packages, starts the Flask server and opens the website. |
| `verify_project.py` | Lightweight structural, JSON, grid, JavaScript and model-parity audit. Does not retrain models or contact IMD. |
| `requirements-research.txt` | Adds PyTorch to the normal dependencies for ConvLSTM retraining. |
| `server.py` | Serves the website at `http://127.0.0.1:8000` and periodically refreshes IMD observations. |
| `README.md` | Setup, run, deployment, model description, and evaluation notes. |
| `FILE_GUIDE.md` | This file: what the main project files and folders are for. |
| `requirements.txt` | Python dependencies for local development and research scripts. |
| `requirements-deploy.txt` | Smaller dependency list used by the deployment Docker image. |
| `.env.example` | Template for IMD credentials. Copy to `.env` and fill it in locally. Do not share `.env`. |
| `Dockerfile` | Container build instructions. |
| `Procfile` | Gunicorn start command for platforms that use a Procfile. |
| `.dockerignore`, `.gitignore` | Exclude local secrets, caches, and generated files from builds/version control. |

## Website: `web/`

| File/folder | Purpose |
|---|---|
| `web/index.html` | Website page structure. |
| `web/app.js` | Browser interface: initializes Leaflet, handles map clicks, loads data, and displays the forecast. |
| `web/model.js` | JavaScript implementation of the trained forecast models. Runs inference in the browser. |
| `web/style.css` | Website layout and visual styling. |
| `web/models.json` | Trained model parameters exported from scikit-learn for browser inference. Large but required. |
| `web/grid_static.json` | Coordinates and static geographic features for the 77 model grid cells. Required by the forecast. |
| `web/live.json` | Latest IMD station observations read by the website. |
| `web/stations.json` | Configured reference IMD station IDs, names, and coordinates. |
| `web/vendor/leaflet/` | Bundled Leaflet map library, CSS, marker images, and its license. Required for the map; source maps are for debugging only. |

## Model training: `training/`

| File/folder | Purpose |
|---|---|
| `training/train_nowcast.py` | Trains the website's gradient-boosting, random-forest, neural-network, and logistic-stacking models; exports `web/models.json`. |
| `training/train_nowcast.ipynb` | Notebook version for Jupyter/Colab workflows. |
| `training/reports/test_metrics.csv`, `test_metrics.json` | Evaluation metrics from the held-out test period. |
| `training/reports/train_log.txt` | Training run log. |
| `training/reports/roc_reliability.png` | ROC/reliability diagnostic plot. |
| `training/reports/js_parity_fixture.json` | Fixed test inputs used to check that JavaScript inference matches Python inference. |

## Offline ConvLSTM research pipeline

| File/folder | Purpose |
|---|---|
| `Varanasi_5km_Static_Conditioned_ConvLSTM.ipynb` | Main notebook for the static-conditioned ConvLSTM experiment. |
| `train_varanasi_convlstm.py` | Main ConvLSTM training/evaluation script. |
| `finetuned_varanasi_convlstm.py` | Fine-tuning experiment variant. |
| `data/` | ERA5, IMERG, DEM, GHSL, and WorldCover gridded input CSVs. These are large source datasets; retain them if retraining. |
| `outputs/best_varanasi_convlstm.pt` | Saved ConvLSTM model checkpoint. |
| `outputs/dynamic_scaler.npz`, `static_scaler.npz` | Fitted preprocessing scalers needed when reproducing inference. |
| `outputs/static_feature_names.npy` | Names/order of static input features. |
| `outputs/training_history.csv` | Per-epoch training history. |
| `outputs/*metrics*.csv` | Test and threshold evaluation metrics. |
| `outputs/*.png` | Model architecture, grid, feature maps, training curves, and evaluation plots. |
| `master_grid_alignment.csv` | Reference alignment of the 77 spatial grid points. |
| `training_feature_schema.csv` | Feature names and schema used by the operational/training data. |

## Live-data utilities and tests

| File/folder | Purpose |
|---|---|
| `fetch_imd.py` | Current website's IMD fetcher. Updates `web/live.json` and its local history. |
| `imd_live_data/` | Saved IMD response samples used by the fetcher's sample/test mode. |
| `fetch_live_12h.py` | Separate extractor for the older 12-hour, gridded ConvLSTM operational-data experiment; not used by the website. |
| `live_12h/` | Output from that 12-hour extractor: gridded CSV/NPZ data and completeness report. Not needed to display the website. |
| `tests/test_nowcast.js` | Checks JavaScript model parity and forecast scenarios. |
| `tests/test_fetch.py` | Tests IMD data parsing/fetch behavior. |
| `.github/workflows/live.yml` | GitHub Actions workflow for scheduled IMD refresh and Pages deployment. |

## Files deliberately removed from the supplied archive

- `nowcast/`: old hourly prototype, including a Google Maps page with a placeholder `YOUR_GOOGLE_MAPS_KEY`; superseded by the Leaflet-based `web/` site.
- `check.py`: older duplicate ConvLSTM training script; use `train_varanasi_convlstm.py` instead.
- `convlstm_new.ipynb`: empty notebook with no code cells.
- `imd_live_data.py`: older IMD fetcher superseded by `fetch_imd.py`.
- `.env`: contained populated IMD credentials. It is deliberately excluded for security.

## What can be removed later if you only want to run the website?

For a website-only deployment, the ConvLSTM notebook/scripts, `data/`, `outputs/`, `live_12h/`, `fetch_live_12h.py`, and the training notebooks/reports are not needed at runtime. Keep `web/`, `server.py`, `fetch_imd.py`, `requirements-deploy.txt`, `Dockerfile`, and the deployment configuration. Keep `.env.example` as a credentials template.
