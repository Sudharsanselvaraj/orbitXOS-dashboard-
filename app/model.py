import os
import pandas as pd
import joblib
from datetime import datetime, timezone, timedelta
from math import floor
from typing import Dict, Any, List

from app.utils import load_tle_map, find_tle_block, risk_bucket, normalize_probabilities

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")
TLE_FILE = os.path.join(BASE_DIR, "active_satellites_tle.txt")
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
_tle_map = load_tle_map(TLE_FILE)

def time_to_impact_str(tca_iso: str) -> str:
    try:
        tca = datetime.fromisoformat(tca_iso)
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = max(0, (tca - now).total_seconds())
        days = int(delta // 86400)
        hours = int((delta % 86400) // 3600)
        minutes = int((delta % 3600) // 60)
        return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m"
    except Exception:
        return "N/A"

def fetch_tle(name: str) -> str:
    return find_tle_block(_tle_map, name)

def predict_top_events(top_n: int = 6) -> Dict[str, Any]:
    """
    Build the dynamic list of conjunction events from CSV + optional ML model.
    Always returns up to top_n events across ALL risk levels (not only critical).
    """
    try:
        df = pd.read_csv(CSV_PATH)

        # Required columns: i_name, j_name, tca, miss_km, vrel_kms
        if not {"i_name","j_name","tca"}.issubset(df.columns):
            return {"critical_events": [], "status": "error",
                    "message": "CSV missing necessary columns"}

        df["EPOCH_dt"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)

        # Probability source: model or heuristic 1/(1+miss)
        if model:
            try:
                if hasattr(model, "feature_names_in_"):
                    feats = list(model.feature_names_in_)
                else:
                    feats = df.columns[:model.n_features_in_]
                X = df[feats].fillna(0.0)
                raw = model.predict_proba(X)[:, 1]
                df["raw_prob"] = raw
            except Exception:
                df["raw_prob"] = 1.0 / (1.0 + df["miss_km"].astype(float))
        else:
            df["raw_prob"] = 1.0 / (1.0 + df["miss_km"].astype(float))

        # Normalize to 0..1
        df["probability"] = normalize_probabilities(df["raw_prob"].tolist())

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=3)   # show a bit wider window
        view_df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)].copy()

        if view_df.empty:
            return {"critical_events": [], "status": "ok", "message": "No events in window."}

        # Sort primarily by probability desc, then soonest TCA
        view_df = view_df.sort_values(["probability", "EPOCH_dt"], ascending=[False, True]).head(top_n)

        results: List[Dict[str, Any]] = []
        for _, row in view_df.iterrows():
            prob = float(row["probability"])
            risk_level = risk_bucket(prob)
            sat_name = str(row["i_name"])
            deb_name = str(row["j_name"])
            tca_iso = str(row["tca"])

            results.append({
                "satellite": sat_name,
                "satellite_tle": fetch_tle(sat_name),
                "debris": deb_name,
                "debris_tle": fetch_tle(deb_name),
                "tca": tca_iso,
                "time_to_impact": time_to_impact_str(tca_iso),
                "miss_km": float(row.get("miss_km", 0.0)),
                "vrel_kms": float(row.get("vrel_kms", 0.0)),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_level,
                "maneuver_suggestion": (
                    "No action needed" if prob < 0.3 else
                    "Monitor, prepare retrograde burn" if prob < 0.6 else
                    "Plan radial maneuver" if prob < 0.8 else
                    "Execute immediate retrograde burn"
                ),
                "confidence": f"{prob*100:.1f}%"
            })

        return {"critical_events": results, "status": "ok"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def build_dashboard() -> Dict[str, Any]:
    """
    Compose the dashboard payload from the local TLE map + events.
    """
    events = predict_top_events(top_n=12)  # get a bigger pool for cards/tables
    items = events.get("critical_events", [])

    # objects_tracked = count of 3-line TLE blocks we parsed
    objects_tracked = len(_tle_map)

    # threats = High + Critical
    active_threats = sum(1 for e in items if e["risk_level"] in ("High","Critical"))

    # predictions_made/today: derive from the slice we just produced
    now = datetime.now(timezone.utc)
    _24h = now - timedelta(hours=24)
    predictions_today = sum(1 for e in items
                            if pd.to_datetime(e["tca"], utc=True, errors="coerce") >= _24h)
    predictions_made = len(items)  # you can replace with DB count if you persist

    # satellites_protected: those with high/critical -> we “recommended” maneuver
    satellites_protected = len({e["satellite"] for e in items if e["risk_level"] in ("High","Critical")})
    satellites_protected_today = sum(1 for e in items
                                     if e["risk_level"] in ("High","Critical") and
                                     pd.to_datetime(e["tca"], utc=True, errors="coerce") >= _24h)

    # recent alerts (most recent by TCA within the next 3 days)
    recent_sorted = sorted(
        (e for e in items),
        key=lambda x: pd.to_datetime(x["tca"], utc=True, errors="coerce"),
        reverse=True
    )[:5]

    recent_alerts = []
    for e in recent_sorted:
        recent_alerts.append({
            "type": "conjunction" if e["risk_level"] in ("High","Critical") else "notice",
            "message": f"Conjunction detected: {e['satellite']}",
            "time": e["tca"]
        })

    # high-priority table (top by risk then nearest TCA)
    hp_sorted = sorted(
        items,
        key=lambda x: (
            {"Critical":3,"High":2,"Medium":1,"Low":0}[x["risk_level"]],
            float(x["probability"].rstrip("%"))
        ),
        reverse=True
    )[:5]

    from app.utils import quick_altitude_velocity_from_tle
    high_priority_tracking = []
    for e in hp_sorted:
        alt_km, vel_kms = quick_altitude_velocity_from_tle(e["satellite_tle"])
        high_priority_tracking.append({
            "object": e["satellite"],
            "type": "Conjunction",
            "altitude_km": round(alt_km, 1),
            "velocity_kms": round(vel_kms, 2),
            "size": "Unknown",
            "risk": e["risk_level"]
        })

    return {
        "stats": {
            "objects_tracked": objects_tracked,
            "objects_tracked_today": max(0, predictions_today // 2),  # placeholder derivation
            "active_threats": active_threats,
            "active_threats_change": 0,  # set 0 unless you persist yesterday’s snapshot
            "predictions_made": predictions_made,
            "predictions_today": predictions_today,
            "satellites_protected": satellites_protected,
            "satellites_protected_today": satellites_protected_today
        },
        "recent_alerts": recent_alerts,
        "high_priority_tracking": high_priority_tracking
    }
