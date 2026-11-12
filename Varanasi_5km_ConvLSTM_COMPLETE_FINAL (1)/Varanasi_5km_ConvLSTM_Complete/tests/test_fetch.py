"""python tests/test_fetch.py  - simulates 4 hourly IMD fetches (no network) and checks live.json."""
import json, shutil, sys, tempfile, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import fetch_imd as F

tmp = Path(tempfile.mkdtemp())
shutil.copy(ROOT / "web" / "stations.json", tmp / "stations.json")
F.WEB = tmp
F.get_jwt = lambda: "x"
now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
cum = [0, 0, 1.0, 3.5]                      # Varanasi gauge: 24h-cumulative rain, oldest -> newest
temps = [30, 29, 27, 25]

def fake_api_get(endpoint, jwt, params=None):
    j = fake_api_get.k
    t = now - dt.timedelta(hours=3 - j)
    if endpoint == "current_wx":
        base = dict(Date_=t.strftime("%Y-%m-%d"))
        return [
            {"Station Id": "42479", "Station": "Varanasi", "Date of Observation": t.strftime("%Y-%m-%d"), "Time": str(t.hour),
             "Mean Sea Level Pressure": str(1010 - j), "Wind Direction": "90", "Wind Speed KMPH": "10.8", "Temperature": str(temps[j]),
             "Humidity": str(70 + 8 * j), "Last 24 hrs Rainfall": str(cum[j]), "WEATHER_MESSAGE": "Moderate Rain" if j == 3 else "Cloudy"},
            {"Station Id": "42483", "Station": "Varanasi BHU", "Date of Observation": "2020-01-01", "Time": "3",       # stale
             "Mean Sea Level Pressure": "NA", "Wind Direction": "140", "Wind Speed KMPH": "3.7", "Temperature": "24", "Humidity": "92",
             "Last 24 hrs Rainfall": "0", "WEATHER_MESSAGE": "Haze"}]
    return [{"ID": "UPX", "STATION": "FARAWAY", "DATE": t.strftime("%Y-%m-%d"), "TIME": t.strftime("%H:%M:%S"), "CURR_TEMP": "30",
             "RH": "60", "WIND_DIRECTION": "10", "WIND_SPEED": 9.3, "MSLP": "1010", "RAINFALL": "0", "Latitude": 28.0, "Longitude": 77.0},
            {"ID": "UPNEAR", "STATION": "NEAR", "DATE": t.strftime("%Y-%m-%d"), "TIME": t.strftime("%H:%M:%S"), "CURR_TEMP": str(temps[j] + 1),
             "RH": "65", "WIND_DIRECTION": "45", "WIND_SPEED": 9.3, "MSLP": str(1010 - j), "RAINFALL": "0.0", "Latitude": 25.1, "Longitude": 82.8}]
F.api_get = fake_api_get
for j in range(4):
    fake_api_get.k = j
    F.run_once([])
live = json.load(open(tmp / "live.json"))
names = {s["name"]: s for s in live["stations"]}
print(list(names))
v = names["Varanasi (Babatpur)"]
assert "Varanasi BHU" not in names, "stale station must be dropped"
assert not any("Faraway" in n for n in names), "station outside radius must be ignored"
assert v["n_obs"] == 4 and v["temp_c"] == [25, 27, 29, 30], v["temp_c"]
assert v["rain_mm"] == [2.5, 1.0, 0.0], v["rain_mm"]
assert abs(v["wind_ms"][0] - 3.0) < 1e-6
assert live["cadence_min"] == 60
print("OK: history, lags, rain differencing, stale/radius filters")
