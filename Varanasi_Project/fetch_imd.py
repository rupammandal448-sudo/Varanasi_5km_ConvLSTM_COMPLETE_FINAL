"""
IMD live-data fetcher  ->  web/live.json  (+ web/history.json)

Merges the authentication / endpoints from imd_live_data.py (JWT + X-API-KEY) with the
station-history logic the website needs.

Sources (both from https://api.imd.gov.in/api/v1):
    current_wx        hourly synoptic observations, ~420 stations, no lat/lon in payload
                      -> only the stations listed in web/stations.json are used (they carry lat/lon)
    aws_data?sid=5    Uttar Pradesh AWS/ARG stations, lat/lon in payload
                      -> every station within AWS_RADIUS_KM of Varanasi is used

Credentials (never hard-coded): IMD_API_KEY, IMD_EMAIL, IMD_PASSWORD
    read from the environment or from a local .env file.

Usage
    python fetch_imd.py                 one fetch, update web/live.json
    python fetch_imd.py --loop 15       keep fetching every 15 minutes
    python fetch_imd.py --probe         print the raw Varanasi records (verify field names / units)
    python fetch_imd.py --sample        parse the saved files in imd_live_data/ instead of calling IMD
    python fetch_imd.py --mock          write synthetic demo data (UI testing only)
"""
import json, math, os, sys, time, glob, statistics
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
WEB.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

API_KEY = os.getenv("IMD_API_KEY", "").strip()
IMD_EMAIL = os.getenv("IMD_EMAIL", "").strip()
IMD_PASSWORD = os.getenv("IMD_PASSWORD", "")
TOKEN_URL = os.getenv("IMD_TOKEN_URL", "https://api.imd.gov.in/api/oauth/token.php")
API_BASE = os.getenv("IMD_API_BASE", "https://api.imd.gov.in/api/v1")
UP_STATE_ID = int(os.getenv("IMD_STATE_ID", "5"))
TIMEOUT = 30

CENTER = (25.3176, 82.9739)                 # Varanasi
AWS_RADIUS_KM = float(os.getenv("AWS_RADIUS_KM", "100"))
# IMD hour stamps are UTC (verified against retrieval time in imd_live_data/). "auto" = UTC unless that puts the
# observation in the future, then IST.
OBS_TZ = os.getenv("IMD_OBS_TZ", "auto").upper()          # auto | UTC | IST
WIND_UNIT = os.getenv("IMD_WIND_UNIT", "kmph")            # kmph | ms | knots   (both IMD endpoints: kmph)
RAIN_FLAG_MM = 0.5     # hourly rain assumed when IMD's weather text says it is raining but the gauge diff is 0
RESET_UTC_HOURS = (3, 4)   # rainfall day boundary 03 UTC (08:30 IST): cumulative gauge resets here
MAX_OBS_AGE_H = 3.0    # drop a station whose latest observation is older than this
KEEP_H = 8
HOUR = 3600 * 1000
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
RAIN_WORDS = ("rain", "drizzle", "shower", "thunder", "storm")


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None


def km(a, b, c, d):
    p = math.pi / 180
    x = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(x))


def to_ms(t):
    return int(t.timestamp() * 1000)


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ IMD API
def get_jwt():
    import requests
    if not (API_KEY and IMD_EMAIL and IMD_PASSWORD):
        raise SystemExit("Missing IMD_API_KEY / IMD_EMAIL / IMD_PASSWORD (set them in .env or the environment).")
    r = requests.post(TOKEN_URL, json={"email": IMD_EMAIL, "password": IMD_PASSWORD},
                      headers={"Content-Type": "application/json", "Accept": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    tok = j.get("access_token") or (j.get("data") or {}).get("access_token") if isinstance(j, dict) else None
    if not tok:
        raise RuntimeError("IMD token response had no access_token (fields: %s)" % list(j))
    return tok


def api_get(endpoint, jwt, params=None):
    import requests
    r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=TIMEOUT,
                     headers={"X-API-KEY": API_KEY, "Authorization": f"Bearer {jwt}", "Accept": "application/json",
                              "User-Agent": "Varanasi-Nowcast/1.0"})
    r.raise_for_status()
    return records(r.json())


def records(p):
    if p is None:
        return []
    if isinstance(p, list):
        return p
    if isinstance(p, dict):
        d = p.get("data")
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    return v
            return [d]
        if d is None:
            return [p]
    return []


# ------------------------------------------------------------------ parsing
def obs_time(date, time_s, now):
    """date 'YYYY-MM-DD', time_s '10' | '10:00:00' -> aware datetime (UTC)."""
    try:
        parts = str(time_s).strip().split(":")
        hh = int(float(parts[0])); mm = int(parts[1]) if len(parts) > 1 else 0
        base = dt.datetime.strptime(str(date).strip()[:10], "%Y-%m-%d").replace(hour=hh, minute=mm)
    except Exception:
        return None
    if OBS_TZ == "IST":
        return base.replace(tzinfo=IST).astimezone(dt.timezone.utc)
    t = base.replace(tzinfo=dt.timezone.utc)
    if OBS_TZ == "AUTO" and t > now + dt.timedelta(minutes=10):
        t = base.replace(tzinfo=IST).astimezone(dt.timezone.utc)
    return t


def wind_ms(v):
    v = num(v)
    if v is None:
        return None
    return v / 3.6 if WIND_UNIT == "kmph" else v * 0.514444 if WIND_UNIT == "knots" else v


def wet(msg):
    m = str(msg or "").lower()
    return any(w in m for w in RAIN_WORDS)


def parse_current(rec, now):
    t = obs_time(rec.get("Date of Observation"), rec.get("Time"), now)
    if not t:
        return None
    return {"ms": to_ms(t), "temp_c": num(rec.get("Temperature")), "rh": num(rec.get("Humidity")),
            "pres_hpa": num(rec.get("Mean Sea Level Pressure")), "wind_ms": wind_ms(rec.get("Wind Speed KMPH")),
            "wind_dir": num(rec.get("Wind Direction")), "rain_cum": num(rec.get("Last 24 hrs Rainfall")),
            "wx_rain": wet(rec.get("WEATHER_MESSAGE"))}


def parse_aws(rec, now):
    t = obs_time(rec.get("DATE"), rec.get("TIME"), now)
    if not t:
        return None
    return {"ms": to_ms(t), "temp_c": num(rec.get("CURR_TEMP")), "rh": num(rec.get("RH")),
            "pres_hpa": num(rec.get("MSLP")), "wind_ms": wind_ms(rec.get("WIND_SPEED")),
            "wind_dir": num(rec.get("WIND_DIRECTION")), "rain_cum": num(rec.get("RAINFALL")),
            "wx_rain": wet(rec.get("WEATHER_MESSAGE"))}


# ------------------------------------------------------------------ history -> live.json
def nearest(h, ms, tol=25 * 60000):
    b = min(h, key=lambda o: abs(o["ms"] - ms), default=None)
    return b if b and abs(b["ms"] - ms) <= tol else None


def hourly_rain(obs):
    """obs: observations for lags 0..3 (newest first). Returns hourly mm list [r0,r1,r2] or None."""
    inc = []
    for k in range(3):
        a, b = obs[k], obs[k + 1]
        if a is None or b is None or a.get("rain_cum") is None or b.get("rain_cum") is None:
            inc.append(None)
            continue
        d = a["rain_cum"] - b["rain_cum"]
        if d < 0:                                    # gauge reset or window roll-off
            hr = dt.datetime.fromtimestamp(a["ms"] / 1000, dt.timezone.utc).hour
            d = a["rain_cum"] if hr in RESET_UTC_HOURS else 0.0
        inc.append(d)
    if inc[0] is None:
        return None
    inc = [0.0 if x is None else x for x in inc]
    if obs[0].get("wx_rain") and inc[0] < RAIN_FLAG_MM:
        inc[0] = RAIN_FLAG_MM
    return [round(x, 2) for x in inc]


def build(st, h, now_ms):
    if not h:
        return None
    L = max(o["ms"] for o in h)
    if (now_ms - L) / HOUR > MAX_OBS_AGE_H:
        return None
    ob = [nearest(h, L - k * HOUR) for k in range(4)]
    n_obs = sum(o is not None for o in ob)

    def ser(key):
        v = [None if o is None else o.get(key) for o in ob]
        if v[0] is None:
            return None
        last = v[0]
        for i in range(1, 4):                        # warm-up: carry the oldest known value back
            if v[i] is None:
                v[i] = last
            else:
                last = v[i]
        return v

    out = {"id": st["id"], "name": st["name"], "lat": st["lat"], "lon": st["lon"], "obs_time": iso(L),
           "n_obs": n_obs, "temp_c": ser("temp_c"), "rh": ser("rh"), "pres_hpa": ser("pres_hpa"),
           "wind_ms": ser("wind_ms"), "wind_dir": ser("wind_dir"), "rain_mm": hourly_rain(ob) if n_obs >= 2 else None}
    return out


def cadence_of(hist):
    gaps = []
    for h in hist.values():
        ms = sorted(o["ms"] for o in h)
        gaps += [(b - a) / 60000 for a, b in zip(ms, ms[1:]) if 5 <= (b - a) / 60000 <= 120]
    if not gaps:
        return 60
    g = statistics.median(gaps)
    return min((15, 30, 60), key=lambda c: abs(c - g))


# ------------------------------------------------------------------ sources
def load_stations():
    return [s for s in json.load(open(WEB / "stations.json")) if not str(s["id"]).startswith("REPLACE")]


def collect_live(now, probe=False, sample=False):
    """Returns {station_id: {'meta':{...}, 'obs': obs}} for the latest observation of every usable station."""
    found = {}
    stations = {str(s["id"]): s for s in load_stations()}
    if sample:
        d = ROOT / "imd_live_data"
        cw = json.load(open(sorted(glob.glob(str(d / "varanasi_current_weather_*.json")))[-1]))
        aw = json.load(open(sorted(glob.glob(str(d / "varanasi_aws_arg_*.json")))[-1]))
    else:
        jwt = get_jwt()
        cw = api_get("current_wx", jwt)
        try:
            aw = api_get("aws_data", jwt, {"sid": UP_STATE_ID})
        except Exception as e:
            print("aws_data failed:", e)
            aw = []
    for r in cw:
        sid = str(r.get("Station Id"))
        if sid in stations:
            if probe:
                print(json.dumps(r, indent=1))
            o = parse_current(r, now)
            if o:
                found[sid] = {"meta": stations[sid], "obs": o}
    for r in aw:
        la, lo = num(r.get("Latitude")), num(r.get("Longitude"))
        if la is None or lo is None or km(CENTER[0], CENTER[1], la, lo) > AWS_RADIUS_KM:
            continue
        if probe:
            print(json.dumps(r, indent=1))
        o = parse_aws(r, now)
        if o:
            sid = "AWS_" + str(r.get("ID"))
            found[sid] = {"meta": {"id": sid, "name": str(r.get("STATION", sid)).title() + " (AWS)", "lat": la, "lon": lo}, "obs": o}
    return found


def mock(now):
    base = to_ms(now.replace(minute=0, second=0, microsecond=0))
    out = {}
    for i, s in enumerate(load_stations()[:3]):
        hist = []
        for k in range(5):
            ms = base - k * HOUR
            hh = ((ms / HOUR) + 5.5) % 24
            hist.append({"ms": ms, "temp_c": round(30 + 4 * math.sin((hh - 9) / 24 * 2 * math.pi) - .3 * i, 1),
                         "rh": round(70 - 15 * math.sin((hh - 9) / 24 * 2 * math.pi) + i, 1),
                         "pres_hpa": round(1000 - 0.4 * k, 1), "wind_ms": round(2.2 + .3 * i, 1), "wind_dir": 250 + 10 * i,
                         "rain_cum": 0.0, "wx_rain": False})
        out[str(s["id"])] = (s, hist)
    return out


# ------------------------------------------------------------------ main
def run_once(args):
    now = dt.datetime.now(dt.timezone.utc)
    now_ms = to_ms(now)
    hp = WEB / "history.json"
    hist = json.load(open(hp)) if hp.exists() else {}
    meta = {}
    demo = "--mock" in args
    if demo:
        hist = {}
        for sid, (s, h) in mock(now).items():
            hist[sid] = h
            meta[sid] = s
    else:
        found = collect_live(now, probe="--probe" in args, sample="--sample" in args)
        if "--probe" in args:
            return
        if not found:
            print("No usable stations returned.")
            sys.exit(1)
        for sid, v in found.items():
            meta[sid] = v["meta"]
            h = hist.setdefault(sid, [])
            if all(x["ms"] != v["obs"]["ms"] for x in h):
                h.append(v["obs"])
            else:                                     # same hour re-fetched: keep the newest values
                for i, x in enumerate(h):
                    if x["ms"] == v["obs"]["ms"]:
                        h[i] = v["obs"]
    horizon_ms = (now_ms if "--sample" not in args else max(o["ms"] for h in hist.values() for o in h)) - KEEP_H * HOUR
    for k in list(hist):
        hist[k] = sorted([o for o in hist[k] if o["ms"] > horizon_ms], key=lambda o: o["ms"])
        if not hist[k]:
            del hist[k]
    ref_ms = now_ms if "--sample" not in args else max(o["ms"] for h in hist.values() for o in h) + 20 * 60000
    if not demo:
        # stations known from earlier runs but not in this response keep their meta from stations.json / history
        for s in load_stations():
            meta.setdefault(str(s["id"]), s)
    sts = []
    for sid, h in hist.items():
        m = meta.get(sid)
        if not m:
            continue
        b = build(m, h, ref_ms)
        if b:
            sts.append(b)
    live = {"updated": iso(now_ms), "cadence_min": cadence_of(hist), "demo": demo,
            "obs_timezone_assumed": "UTC", "stations": sts}
    json.dump(hist, open(hp, "w"))
    tmp = WEB / "live.json.tmp"
    json.dump(live, open(tmp, "w"), indent=1)
    os.replace(tmp, WEB / "live.json")
    print(time.strftime("%H:%M:%S"), "wrote live.json with", len(sts), "station(s):",
          ", ".join(f"{s['name']}@{s['obs_time'][11:16]}Z(n={s['n_obs']})" for s in sts))


def main():
    a = sys.argv[1:]
    if "--loop" in a:
        mins = float(a[a.index("--loop") + 1]) if len(a) > a.index("--loop") + 1 else 15
        while True:
            try:
                run_once([x for x in a if x != "--loop"])
            except Exception as e:
                print("fetch failed:", e)
            time.sleep(mins * 60)
    else:
        run_once(a)


if __name__ == "__main__":
    main()
