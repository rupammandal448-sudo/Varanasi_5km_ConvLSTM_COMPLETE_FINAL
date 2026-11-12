"""
Production server: serves web/ and keeps web/live.json fresh from the IMD API.

    python server.py                       (dev, http://localhost:8000)
    gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT server:app     (production; keep ONE worker)

Env: IMD_API_KEY, IMD_EMAIL, IMD_PASSWORD   (required for live data)
     FETCH_EVERY_MIN (default 15)           how often to call IMD
"""
import os, threading, time
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

import fetch_imd

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
EVERY = float(os.getenv("FETCH_EVERY_MIN", "15"))
state = {"last_ok": None, "last_error": None}

app = Flask(__name__, static_folder=None)


def worker():
    while True:
        try:
            fetch_imd.run_once([])
            state["last_ok"], state["last_error"] = time.time(), None
        except BaseException as e:                     # SystemExit from missing credentials too
            state["last_error"] = repr(e)
            print("IMD fetch failed:", e, flush=True)
        time.sleep(EVERY * 60)


_started = False


def start_worker():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=worker, daemon=True).start()


start_worker()


@app.route("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.route("/live.json")
def live():
    r = send_from_directory(WEB, "live.json")
    r.headers["Cache-Control"] = "no-store"
    return r


@app.route("/healthz")
def health():
    age = None if not state["last_ok"] else round((time.time() - state["last_ok"]) / 60, 1)
    return jsonify(ok=state["last_ok"] is not None, minutes_since_last_imd_fetch=age, last_error=state["last_error"])


@app.route("/<path:p>")
def static_files(p):
    if p in ("history.json", "stations.json.tmp"):
        return ("", 404)
    return send_from_directory(WEB, p)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
