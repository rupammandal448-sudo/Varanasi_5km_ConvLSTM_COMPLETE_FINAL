"""
VARANASI 30-MINUTE NOWCAST - STACKED ENSEMBLE TRAINING
=======================================================

Base learners (3):
    1. HistGradientBoostingClassifier   - strongest single learner on tabular weather features
    2. RandomForestClassifier           - bagged trees, different bias/variance profile
    3. MLPClassifier (small neural net) - smooth decision surface, captures interactions
Meta learner:
    LogisticRegression on the logits of the 3 base probabilities
    (fitted on out-of-fold predictions so it never sees a base model's own training rows).

Target : IMERG rain >= 0.1 mm/hr in the 30-minute window that starts at the observation time.
Inputs : only variables that IMD stations report live (T, RH, wind, pressure tendency, rain
         history) plus 4 static 5-km descriptors (elevation, built-up, water, tree cover).
         ERA5 precipitation is deliberately NOT used as a feature.

Data   : <project>/data/ERA5_Varanasi_5km_<year>_Q<n>*.csv, IMERG_Varanasi_5km_<year>_Q<n>_ERA5_aligned.csv
         Any extra years you drop into data/ (e.g. 2020, 2021) are picked up automatically.
Split  : test = 2025 (never touched during fitting) ; everything earlier = train/OOF.
Output : <project>/web/models.json, <project>/web/grid_static.json,
         <project>/training/reports/*

Run    : python training/train_nowcast.py
"""
import glob, json, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent          # project root
DATA = ROOT / "data"
WEB = ROOT / "web"
REPORTS = ROOT / "training" / "reports"
WEB.mkdir(exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- config
TEST_YEAR = 2025
THR = 0.1                 # mm/hr, rain / no-rain
NPOS_NEG_RATIO = 5        # negatives kept per positive in each base-model fit
MAX_FIT_ROWS = 160_000
N_FOLDS_BY_QUARTER = True # OOF folds = calendar quarters (all years pooled)
SEED = 0
N_GRID = 77

ERA5_RENAME = {
    "temperature_2m": "temperature_2m_K",
    "dewpoint_temperature_2m": "dewpoint_temperature_2m_K",
    "surface_pressure": "surface_pressure_Pa",
    "u_component_of_wind_10m": "u_wind_10m_ms",
    "v_component_of_wind_10m": "v_wind_10m_ms",
}
ERA5_NEED = ["temperature_2m_K", "dewpoint_temperature_2m_K", "surface_pressure_Pa",
             "u_wind_10m_ms", "v_wind_10m_ms"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- grid + static
grid = pd.read_csv(ROOT / "master_grid_alignment.csv").sort_values("grid_index").reset_index(drop=True)
assert len(grid) == N_GRID
gid2idx = dict(zip(grid.grid_id, grid.grid_index))

dem = pd.read_csv(DATA / "DEM_5km_IMERG_grid_aligned.csv")[["grid_id", "elevation_mean"]]
wc = pd.read_csv(DATA / "WorldCover_5km_IMERG_grid_aligned.csv")[
    ["grid_id", "builtup_fraction", "water_fraction", "tree_fraction"]]
static = (grid[["grid_id", "grid_index", "latitude", "longitude"]]
          .merge(dem, on="grid_id", how="left").merge(wc, on="grid_id", how="left")
          .rename(columns={"elevation_mean": "elev"})
          .rename(columns={"builtup_fraction": "builtup", "water_fraction": "water", "tree_fraction": "tree"}))
static = static.sort_values("grid_index").reset_index(drop=True)
assert static[["elev", "builtup", "water", "tree"]].notna().all().all()
STATIC_NAMES = ["elev", "builtup", "water", "tree"]
S = static[STATIC_NAMES].to_numpy("float32")           # (77, 4)

# --------------------------------------------------------------------------- load ERA5 + IMERG
era_files = sorted(glob.glob(str(DATA / "ERA5_Varanasi_5km_*.csv")))
img_files = sorted(glob.glob(str(DATA / "IMERG_Varanasi_5km_*_ERA5_aligned.csv")))
years = sorted({int(re.search(r"_(\d{4})_Q", os.path.basename(f)).group(1)) for f in era_files})
log("years found:", years, "| ERA5 files:", len(era_files), "| IMERG files:", len(img_files))
START = pd.Timestamp(f"{years[0]}-01-01")
END = pd.Timestamp(f"{years[-1] + 1}-01-01")
H = int((END - START) / pd.Timedelta(hours=1))

A = {k: np.full((H, N_GRID), np.nan, "float32") for k in ERA5_NEED}
for f in era_files:
    d = pd.read_csv(f, usecols=lambda c: ERA5_RENAME.get(c, c) in ERA5_NEED + ["grid_id", "timestamp"]
                    ).rename(columns=ERA5_RENAME)
    hi = ((pd.to_datetime(d.timestamp) - START) / pd.Timedelta(hours=1)).astype(int).values
    pi = d.grid_id.map(gid2idx).values
    ok = (hi >= 0) & (hi < H) & ~pd.isna(pi)
    for k in ERA5_NEED:
        A[k][hi[ok], pi[ok].astype(int)] = d[k].values[ok]
    del d
IM = np.full((2 * H, N_GRID), np.nan, "float32")          # 30-minute IMERG
for f in img_files:
    d = pd.read_csv(f, usecols=["grid_id", "timestamp", "precipitation_mm_hr"])
    hi = ((pd.to_datetime(d.timestamp) - START) / pd.Timedelta(minutes=30)).astype(int).values
    pi = d.grid_id.map(gid2idx).values
    ok = (hi >= 0) & (hi < 2 * H) & ~pd.isna(pi)
    IM[hi[ok], pi[ok].astype(int)] = d.precipitation_mm_hr.values[ok]
    del d
log("loaded. ERA5 coverage %.1f%%, IMERG coverage %.1f%%" %
    (100 * np.isfinite(A["temperature_2m_K"]).mean(), 100 * np.isfinite(IM).mean()))

# --------------------------------------------------------------------------- physical variables
T = A["temperature_2m_K"] - 273.15
Td = A["dewpoint_temperature_2m_K"] - 273.15
es = lambda x: 6.112 * np.exp(17.62 * x / (243.12 + x))
RH = np.clip(100 * es(Td) / es(T), 1, 100)
P = A["surface_pressure_Pa"] / 100
U, V = A["u_wind_10m_ms"], A["v_wind_10m_ms"]
W = np.hypot(U, V)


def lag(a, k):
    o = np.full_like(a, np.nan)
    o[k:] = a[:-k]
    return o


def lead(a, k):
    o = np.full_like(a, np.nan)
    o[:-k] = a[k:]
    return o


# IMERG timestamp = start of the half-hour. At observation time t (hour h) the rain that has
# already happened is IM[2h-1] ([t-30,t)), IM[2h-2], ...  The target is IM[2h] = [t, t+30).
def im_at(offset):                                         # IM[2h + offset] for every hour h
    o = np.full((H, N_GRID), np.nan, "float32")
    idx = 2 * np.arange(H) + offset
    ok = (idx >= 0) & (idx < 2 * H)
    o[ok] = IM[idx[ok]]
    return o


rain0 = 0.5 * (im_at(-1) + im_at(-2))      # mm in the hour before t   (what a gauge would have reported)
rain1 = 0.5 * (im_at(-3) + im_at(-4))
rain2 = 0.5 * (im_at(-5) + im_at(-6))
y_rain = im_at(0)                          # target intensity

g = np.log(RH / 100) + 17.62 * T / (243.12 + T)
DD = T - 243.12 * g / (17.62 - g)          # dewpoint depression
ts = START + pd.to_timedelta(np.arange(H), "h")
bc = lambda a: np.broadcast_to(np.asarray(a, dtype="float32")[:, None], (H, N_GRID))
hr, doy = bc(ts.hour.values), bc(ts.dayofyear.values)
yr, qt = bc(ts.year.values), bc(ts.quarter.values)

FE = {
    "temp_c": T, "rh": RH, "dewdep": DD, "u": U, "v": V, "wspd": W,
    "dT1": T - lag(T, 1), "dT3": T - lag(T, 3), "dRH1": RH - lag(RH, 1), "dRH3": RH - lag(RH, 3),
    "dws1": W - lag(W, 1),
    "hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
    "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    "dp1": P - lag(P, 1), "dp3": P - lag(P, 3),
    "rain0": rain0, "rain1": rain1, "rain2": rain2, "rain3sum": rain0 + rain1 + rain2,
}
for j, n in enumerate(STATIC_NAMES):
    FE[n] = np.broadcast_to(S[:, j][None, :], (H, N_GRID))

LITE = ["temp_c", "rh", "dewdep", "u", "v", "wspd", "dT1", "dT3", "dRH1", "dRH3", "dws1",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos"] + STATIC_NAMES
FULL = LITE + ["dp1", "dp3", "rain0", "rain1", "rain2", "rain3sum"]

Z = np.stack([FE[k].astype("float32") for k in FULL], -1)
ok = np.isfinite(Z).all(-1) & np.isfinite(y_rain)
for a in (T, RH, U, V):
    ok &= np.isfinite(lead(a, 1))
Xall = Z[ok]
del Z
y_int = y_rain[ok]
yall = (y_int >= THR)
yrs, qts = yr[ok], qt[ok]
r0 = rain0[ok]
DT = (lead(T, 1) - T)[ok]; DR = (lead(RH, 1) - RH)[ok]
DU = (lead(U, 1) - U)[ok]; DV = (lead(V, 1) - V)[ok]
del FE, A, T, Td, RH, P, U, V, W, g, DD, IM

tr = np.where(yrs < TEST_YEAR)[0]
te = np.where(yrs == TEST_YEAR)[0]
if len(te) == 0:
    raise SystemExit(f"No rows for test year {TEST_YEAR} - check the data folder.")
log("rows: train/OOF pool %d | test %d | rain rate train %.3f test %.3f" %
    (len(tr), len(te), yall[tr].mean(), yall[te].mean()))

# --------------------------------------------------------------------------- helpers
rs = np.random.RandomState(SEED)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def balanced_sample(idx):
    pos = idx[yall[idx]]
    neg = idx[~yall[idx]]
    neg = rs.choice(neg, min(len(neg), NPOS_NEG_RATIO * max(len(pos), 1)), replace=False)
    s = np.concatenate([pos, neg])
    if len(s) > MAX_FIT_ROWS:
        s = rs.choice(s, MAX_FIT_ROWS, replace=False)
    return rs.permutation(s)


def fit_base(cols, idx):
    ci = [FULL.index(c) for c in cols]
    s = balanced_sample(idx)
    Xs, ys = Xall[s][:, ci], yall[s]
    hg = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6,
                                        min_samples_leaf=100, l2_regularization=1.0,
                                        random_state=SEED).fit(Xs, ys)
    rf = RandomForestClassifier(n_estimators=30, max_depth=10, min_samples_leaf=100,
                                max_features="sqrt", max_samples=0.6, n_jobs=-1,
                                random_state=SEED).fit(Xs, ys)
    sc = StandardScaler().fit(Xs)
    nn = MLPClassifier(hidden_layer_sizes=(48, 24), early_stopping=True, n_iter_no_change=4,
                       max_iter=40, batch_size=512, random_state=SEED).fit(sc.transform(Xs), ys)
    return dict(ci=ci, hg=hg, rf=rf, sc=sc, nn=nn)


def base_probs(M, X):
    X = X[:, M["ci"]]
    return np.stack([M["hg"].predict_proba(X)[:, 1],
                     M["rf"].predict_proba(X)[:, 1],
                     M["nn"].predict_proba(M["sc"].transform(X))[:, 1]], 1)


def skill(y, p, thr=0.5):
    h = p >= thr
    hit, miss, fa = (h & y).sum(), (~h & y).sum(), (h & ~y).sum()
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), brier=brier_score_loss(y, p),
                pod=hit / max(hit + miss, 1), far=fa / max(hit + fa, 1), csi=hit / max(hit + miss + fa, 1),
                f1=2 * hit / max(2 * hit + miss + fa, 1))


def best_threshold(y, p):
    best, bt = -1, 0.5
    for t in np.arange(0.05, 0.95, 0.025):
        c = skill(y, p, t)["csi"]
        if c > best:
            best, bt = c, float(t)
    return bt


# --------------------------------------------------------------------------- train variants
MODELS, RES, OOF = {}, {}, {}
for name, cols in (("full", FULL), ("lite", LITE)):
    log(f"=== variant {name}: {len(cols)} features")
    oof = np.full((len(tr), 3), np.nan)
    for q in (1, 2, 3, 4):                                  # blocked-by-season out-of-fold
        hold = qts[tr] == q
        if hold.sum() == 0 or (~hold).sum() == 0:
            continue
        M = fit_base(cols, tr[~hold])
        oof[hold] = base_probs(M, Xall[tr[hold]])
        log(f"  OOF quarter {q} done ({hold.sum()} rows held out)")
    good = np.isfinite(oof).all(1)
    stack = LogisticRegression(C=1.0).fit(logit(oof[good]), yall[tr][good])
    log("  stacker weights (GB, RF, NN):", stack.coef_.round(3), "intercept", stack.intercept_.round(3))
    M = fit_base(cols, tr)                                   # final base models on all pre-test data
    M["st"] = stack
    MODELS[name] = M
    p_oof = stack.predict_proba(logit(oof[good]))[:, 1]
    thr = best_threshold(yall[tr][good], p_oof)
    M["thr"] = thr
    OOF[name] = (oof[good], yall[tr][good])
    log(f"  decision threshold (max CSI on OOF): {thr:.3f}")

    B = base_probs(M, Xall[te])
    pe = stack.predict_proba(logit(B))[:, 1]
    yt, rt = yall[te], r0[te]
    R = {"gradient_boosting": skill(yt, B[:, 0], thr), "random_forest": skill(yt, B[:, 1], thr),
         "neural_net": skill(yt, B[:, 2], thr), "stacked_logistic": skill(yt, pe, thr)}
    # base-model probabilities are un-calibrated (balanced sampling) -> report AUC/AP only for them
    for k in ("gradient_boosting", "random_forest", "neural_net"):
        R[k]["note"] = "uncalibrated base learner; Brier not comparable"
    R["stacked_logistic"]["onset_auc"] = roc_auc_score(yt[rt < THR], pe[rt < THR])
    R["stacked_logistic"]["cessation_auc"] = roc_auc_score(yt[rt >= THR], pe[rt >= THR])
    RES[name] = R
    RES[name]["_pe"] = pe
    log(f"  TEST {TEST_YEAR} stacked: AUC {R['stacked_logistic']['auc']:.3f} AP {R['stacked_logistic']['ap']:.3f} "
        f"Brier {R['stacked_logistic']['brier']:.4f} POD {R['stacked_logistic']['pod']:.3f} "
        f"FAR {R['stacked_logistic']['far']:.3f} CSI {R['stacked_logistic']['csi']:.3f}")

yt, rt = yall[te], r0[te]
persist = (rt >= THR).astype(float)
RES["persistence"] = skill(yt, persist, 0.5)
RES["climatology_brier"] = brier_score_loss(yt, np.full(len(yt), yall[tr].mean()))

# reliability table (full variant)
pe = RES["full"]["_pe"]
rel = []
q = np.minimum((pe * 10).astype(int), 9)
for i in range(10):
    m = q == i
    if m.sum() > 200:
        rel.append(dict(bin=f"{i/10:.1f}-{(i+1)/10:.1f}", predicted=float(pe[m].mean()),
                        observed=float(yt[m].mean()), n=int(m.sum())))
RES["lite"].pop("_pe", None)
RES["full"].pop("_pe", None)

# --------------------------------------------------------------------------- trend (T / RH / wind) ridge
LI = [FULL.index(c) for c in LITE]
XL = Xall[:, LI]
sub = rs.choice(tr, min(300_000, len(tr)), replace=False)
tsc = StandardScaler().fit(XL[sub])
TR, TRM = {}, {}
for k, tg in (("temp_c", DT), ("rh", DR), ("u", DU), ("v", DV)):
    rg = Ridge(alpha=10).fit(tsc.transform(XL[sub]), tg[sub])
    pr = rg.predict(tsc.transform(XL[te]))
    TRM[k] = dict(mae_model_1h=float(np.abs(pr - tg[te]).mean()), mae_persistence_1h=float(np.abs(tg[te]).mean()))
    TR[k] = {"coef": [float("%.7g" % x) for x in rg.coef_], "intercept": float("%.7g" % rg.intercept_)}
    log("trend %-7s MAE model %.4f vs persistence %.4f (per hour)" %
        (k, TRM[k]["mae_model_1h"], TRM[k]["mae_persistence_1h"]))

# --------------------------------------------------------------------------- export to JSON for the browser
r7 = lambda x: float("%.7g" % x)


def hgb_json(m):
    out = []
    for it in m._predictors:
        n = it[0].nodes
        out.append({"f": np.where(n["is_leaf"], -1, n["feature_idx"]).astype(int).tolist(),
                    "t": [float(x) for x in n["num_threshold"]],
                    "l": n["left"].astype(int).tolist(), "r": n["right"].astype(int).tolist(),
                    "v": [r7(x) for x in n["value"]]})
    return {"base": float(np.ravel(m._baseline_prediction)[0]), "trees": out}


def rf_json(m):
    out = []
    for e in m.estimators_:
        a = e.tree_
        v = a.value[:, 0, :]
        out.append({"f": a.feature.astype(int).tolist(), "t": [float(x) for x in a.threshold],
                    "l": a.children_left.astype(int).tolist(), "r": a.children_right.astype(int).tolist(),
                    "v": [r7(x) for x in (v[:, 1] / v.sum(1))]})
    return out


def nn_json(M):
    return {"mu": [r7(x) for x in M["sc"].mean_], "sd": [r7(x) for x in M["sc"].scale_],
            "W": [[[r7(x) for x in row] for row in w] for w in M["nn"].coefs_],
            "b": [[r7(x) for x in b] for b in M["nn"].intercepts_]}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


OUTJ = {
    "meta": {
        "horizon_min": 30, "rain_threshold_mm_hr": THR,
        "target": f"IMERG rain >= {THR} mm/hr during the 30 minutes after the observation time",
        "train_years": [int(y) for y in years if y < TEST_YEAR], "test_year": TEST_YEAR,
        "base_models": ["HistGradientBoosting", "RandomForest", "MLP"], "meta_model": "LogisticRegression",
        "metrics": clean({k: v for k, v in RES.items()}), "reliability_full": rel,
        "trend_metrics": TRM,
    },
    "trend": {"features": LITE, "mu": [r7(x) for x in tsc.mean_], "sd": [r7(x) for x in tsc.scale_], "targets": TR},
}
for name, cols in (("full", FULL), ("lite", LITE)):
    M = MODELS[name]
    OUTJ[name] = {"features": cols, "gb": hgb_json(M["hg"]), "rf": rf_json(M["rf"]), "nn": nn_json(M),
                  "stack": {"coef": [r7(x) for x in M["st"].coef_[0]], "intercept": r7(M["st"].intercept_[0])},
                  "threshold": r7(M["thr"])}
with open(WEB / "models.json", "w") as f:
    json.dump(OUTJ, f, separators=(",", ":"))
log("models.json", round(os.path.getsize(WEB / "models.json") / 1e6, 2), "MB")

# grid_static.json for the browser: nearest 5-km cell -> static features
lat_min, lat_max = static.latitude.min(), static.latitude.max()
lon_min, lon_max = static.longitude.min(), static.longitude.max()
pad = 0.03
gs = {"cells": [{"id": r.grid_id, "lat": round(float(r.latitude), 6), "lon": round(float(r.longitude), 6),
                 **{n: round(float(getattr(r, n)), 5) for n in STATIC_NAMES}} for r in static.itertuples()],
      "cover": {"n": round(float(lat_max + pad), 4), "s": round(float(lat_min - pad), 4),
                "e": round(float(lon_max + pad), 4), "w": round(float(lon_min - pad), 4)},
      "max_cell_km": 6.0}
json.dump(gs, open(WEB / "grid_static.json", "w"), separators=(",", ":"))

# --------------------------------------------------------------------------- parity check Python vs exported JSON
def walk(t, x):
    n = 0
    while t["f"][n] >= 0:
        n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
    return t["v"][n]


sig = lambda z: 1 / (1 + np.exp(-z))


def py_eval(J, X):
    X = np.asarray(X, "float64")
    gb = np.array([sig(J["gb"]["base"] + sum(walk(t, x) for t in J["gb"]["trees"])) for x in X])
    rf = np.array([np.mean([walk(t, x) for t in J["rf"]]) for x in X])
    Zs = (X - np.array(J["nn"]["mu"])) / np.array(J["nn"]["sd"])
    for i, (w, b) in enumerate(zip(J["nn"]["W"], J["nn"]["b"])):
        Zs = Zs @ np.array(w) + np.array(b)
        if i < len(J["nn"]["W"]) - 1:
            Zs = np.maximum(Zs, 0)
    nn = sig(Zs[:, 0])
    s = J["stack"]
    return sig(s["intercept"] + s["coef"][0] * logit(gb) + s["coef"][1] * logit(rf) + s["coef"][2] * logit(nn))


ix = np.random.RandomState(1).choice(len(te), 300, replace=False)
parity = {}
for name in ("full", "lite"):
    M = MODELS[name]
    X = Xall[te][ix][:, M["ci"]]
    ref = M["st"].predict_proba(logit(base_probs(M, Xall[te][ix])))[:, 1]
    parity[name] = float(np.abs(py_eval(OUTJ[name], X) - ref).max())
    log(f"exported JSON vs sklearn, max |diff| ({name}):", parity[name])
    assert parity[name] < 1e-4, "JSON export does not match sklearn"

# parity fixtures for the JS test (node tests/test_model_js.js)
fx = {"full": {"X": Xall[te][ix][:20][:, MODELS["full"]["ci"]].tolist(),
               "p": MODELS["full"]["st"].predict_proba(logit(base_probs(MODELS["full"], Xall[te][ix][:20])))[:, 1].tolist()},
      "lite": {"X": Xall[te][ix][:20][:, MODELS["lite"]["ci"]].tolist(),
               "p": MODELS["lite"]["st"].predict_proba(logit(base_probs(MODELS["lite"], Xall[te][ix][:20])))[:, 1].tolist()}}
json.dump(fx, open(REPORTS / "js_parity_fixture.json", "w"))

# --------------------------------------------------------------------------- reports
def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, pre + k + ".")
        else:
            yield pre + k, v


rows = []
for name in ("full", "lite"):
    for k in ("gradient_boosting", "random_forest", "neural_net", "stacked_logistic"):
        rows.append({"variant": name, "model": k, **{m: v for m, v in RES[name][k].items() if m != "note"}})
rows.append({"variant": "baseline", "model": "persistence", **RES["persistence"]})
pd.DataFrame(rows).to_csv(REPORTS / "test_metrics.csv", index=False)
json.dump(clean({"metrics": RES, "reliability_full": rel, "trend": TRM, "parity": parity,
                 "stack_weights": {n: MODELS[n]["st"].coef_[0].tolist() for n in MODELS},
                 "thresholds": {n: MODELS[n]["thr"] for n in MODELS}}),
          open(REPORTS / "test_metrics.json", "w"), indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, col in (("full", "tab:blue"), ("lite", "tab:orange")):
        M = MODELS[name]
        p = M["st"].predict_proba(logit(base_probs(M, Xall[te])))[:, 1]
        fpr, tpr, _ = roc_curve(yt, p)
        ax[0].plot(fpr, tpr, color=col, label=f"{name} stack (AUC {RES[name]['stacked_logistic']['auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[0].set(xlabel="False-positive rate", ylabel="Hit rate", title=f"ROC, {TEST_YEAR} test, +30 min")
    ax[0].legend()
    ax[1].plot([r["predicted"] for r in rel], [r["observed"] for r in rel], "o-")
    ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
    ax[1].set(xlabel="Predicted probability", ylabel="Observed frequency", title="Reliability (full stack)")
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_reliability.png", dpi=120)
except Exception as e:
    log("plot skipped:", e)
log("DONE")
