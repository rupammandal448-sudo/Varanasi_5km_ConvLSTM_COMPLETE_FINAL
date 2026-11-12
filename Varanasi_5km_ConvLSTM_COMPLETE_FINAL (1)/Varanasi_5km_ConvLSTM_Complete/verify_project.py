"""Run a structural and lightweight runtime audit of the Varanasi project.

Usage: py verify_project.py
This does not retrain either model or make an external IMD API request.
"""
from pathlib import Path
import json, csv, sys, subprocess
ROOT = Path(__file__).resolve().parent
required = [
    'index.html','run_local.bat','server.py','fetch_imd.py','requirements.txt',
    'master_grid_alignment.csv','web/index.html','web/app.js','web/model.js',
    'web/style.css','web/models.json','web/grid_static.json','web/live.json',
    'web/stations.json','web/vendor/leaflet/leaflet.js',
    'web/vendor/leaflet/leaflet.css','training/train_nowcast.py',
    'train_varanasi_convlstm.py','outputs/best_varanasi_convlstm.pt',
]
missing=[x for x in required if not (ROOT/x).is_file()]
if missing:
    print('FAIL missing required files:', *missing, sep='\n  '); sys.exit(1)
for name in ['web/models.json','web/grid_static.json','web/live.json','web/stations.json']:
    with (ROOT/name).open(encoding='utf-8') as f: json.load(f)
with (ROOT/'master_grid_alignment.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
    assert len(rows)==77, f'Expected 77 grid rows, got {len(rows)}'
    assert {'grid_id','grid_index','grid_row','grid_col','latitude','longitude'} <= set(rows[0]), 'Grid schema mismatch'
with (ROOT/'web/grid_static.json').open(encoding='utf-8') as f: grid=json.load(f)
assert len(grid['cells'])==77, f"Expected 77 web grid cells, got {len(grid['cells'])}"
html=(ROOT/'web/index.html').read_text(encoding='utf-8')
for asset in ['vendor/leaflet/leaflet.css','style.css','vendor/leaflet/leaflet.js','model.js','app.js']:
    assert asset in html, f'HTML does not reference {asset}'
    assert (ROOT/'web'/asset).is_file(), f'Missing web asset {asset}'
print('PASS: required files, JSON, 77-cell grid, and website asset references')
# Optional JavaScript syntax + model parity checks.
node=subprocess.run(['node','--check','web/app.js'],cwd=ROOT,capture_output=True,text=True)
if node.returncode: print(node.stderr); sys.exit(node.returncode)
node=subprocess.run(['node','--check','web/model.js'],cwd=ROOT,capture_output=True,text=True)
if node.returncode: print(node.stderr); sys.exit(node.returncode)
node=subprocess.run(['node','tests/test_nowcast.js'],cwd=ROOT,text=True)
if node.returncode: sys.exit(node.returncode)
print('PASS: JavaScript syntax and model parity/scenario tests')
print('NOTE: IMD network access and full model retraining are intentionally not run by this audit.')
