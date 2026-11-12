# Project completion and validation report

## Archive / file integrity
- Final archive opens successfully and passes ZIP CRC integrity validation.
- 90 project files are included.
- No `.env` file or populated credentials are included.
- Required web assets, model JSON, 77-cell grid, IMD fetcher, server, training scripts, source datasets and existing ConvLSTM checkpoint are present.
- Generated Python bytecode and pytest cache directories were removed.
- Unused Leaflet source bundles and source maps were removed; the runtime JS, CSS, marker images and license remain.
- Obsolete Google Maps `nowcast/` prototype was not carried forward; the active site is the Leaflet site in `web/`.

## Checks actually run
- Python syntax compilation: passed for all top-level and training Python scripts.
- JSON parsing: passed for `web/models.json`, `web/grid_static.json`, `web/live.json`, and `web/stations.json`.
- Grid validation: passed; `master_grid_alignment.csv` and `web/grid_static.json` each contain 77 grid cells.
- HTML asset-reference validation: passed; all local CSS/JS assets referenced by `web/index.html` exist.
- JavaScript syntax checks: passed for `web/app.js` and `web/model.js`.
- `node tests/test_nowcast.js`: passed; full and lite JavaScript model parity errors were approximately 4.41e-8 and 4.92e-8, and all listed forecast scenarios passed.
- `python tests/test_fetch.py`: passed using mocked IMD API responses; history, rain differencing, stale-station filtering and radius filtering passed.
- ZIP integrity and secret exclusion: passed.

## Checks not run / limitations
- A complete website HTTP smoke test could not be run in this audit environment because Flask is not installed here. Install `requirements.txt` before launching `run_local.bat` on Windows.
- No live IMD API call was made: credentials and network access are user-specific. Run `py fetch_imd.py --probe` after configuring `.env` to validate current IMD field names and units.
- Neither training pipeline was fully retrained. Full training is compute-intensive; this audit checks source syntax, required data/artifacts and the already-exported JavaScript model behavior, not reproducibility of a full training run.
- A visual browser test was not performed. Direct `file://` mode depends on extracting the ZIP first and having network access to the CARTO/OSM tile provider. Browser restrictions mean local model/live JSON fetches require `run_local.bat`.

## Correct startup sequence
1. Right-click the ZIP and choose **Extract All**. Do not open HTML from inside the ZIP preview.
2. Install dependencies: `py -m pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and add your own IMD credentials if live IMD updates are required.
4. Double-click `run_local.bat` and keep its terminal window open.
5. Use `http://127.0.0.1:8000` for the full website and forecasts.
