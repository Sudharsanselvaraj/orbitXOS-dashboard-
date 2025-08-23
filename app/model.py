import os
import pandas as pd
import joblib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from app.utils import load_tle_map, find_tle_block, risk_bucket, normalize_probabilities, quick_altitude_velocity_from_tle

# File paths
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "reduced_file2.csv")
TLE_FILE = os.path.join(BASE_DIR, "active_satellites_tle.txt")
MODEL_PATH = os.path.join(BASE_DIR, "prop_risk_model_resaved.joblib")

# Load model and TLE map
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
_tle_map = load_tle_map(TLE_FILE)

def time_to_impact_str(tca_iso: str) -> str:
    """Return time to impact as human-readable string."""
    try:
        tca = datetime.fromisoformat(tca_iso)
        if tca.tzinfo is None:
            tca = tca.replace(tzinfo=timezone.utc)
        delta = max(0, (tca - datetime.now(timezone.utc)).total_seconds())
        days, remainder = divmod(delta, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes = remainder // 60
        return f"{int(days)}d {int(hours)}h {int(minutes)}m" if days else f"{int(hours)}h {int(minutes)}m"
    except Exception:
        return "N/A"

def fetch_tle(name: str) -> str:
    """Return TLE string for a satellite or debris."""
    return find_tle_block(_tle_map, name)

def predict_top_events(top_n: int = 6) -> Dict[str, Any]:
    """Return top N ranked conjunction events."""
    try:
        df = pd.read_csv(CSV_PATH)
        if not {"i_name","j_name","tca"}.issubset(df.columns):
            return {"critical_events": [], "status": "error", "message": "CSV missing necessary columns"}

        df["EPOCH_dt"] = pd.to_datetime(df["tca"], errors="coerce", utc=True)
        # Probability
        if model:
            try:
                feats = list(getattr(model, "feature_names_in_", df.columns[:model.n_features_in_]))
                X = df[feats].fillna(0.0)
                df["raw_prob"] = model.predict_proba(X)[:, 1]
            except Exception:
                df["raw_prob"] = 1.0 / (1.0 + df["miss_km"].astype(float))
        else:
            df["raw_prob"] = 1.0 / (1.0 + df["miss_km"].astype(float))

        df["probability"] = normalize_probabilities(df["raw_prob"].tolist())

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=3)
        view_df = df[(df["EPOCH_dt"] >= now) & (df["EPOCH_dt"] <= cutoff)].copy()
        if view_df.empty:
            return {"critical_events": [], "status": "ok", "message": "No events in window."}

        view_df = view_df.sort_values(["probability", "EPOCH_dt"], ascending=[False, True]).head(top_n)

        results: List[Dict[str, Any]] = []
        for _, row in view_df.iterrows():
            prob = float(row["probability"])
            results.append({
                "satellite": row["i_name"],
                "satellite_tle": fetch_tle(row["i_name"]),
                "debris": row["j_name"],
                "debris_tle": fetch_tle(row["j_name"]),
                "tca": str(row["tca"]),
                "time_to_impact": time_to_impact_str(str(row["tca"])),
                "miss_km": float(row.get("miss_km", 0.0)),
                "vrel_kms": float(row.get("vrel_kms", 0.0)),
                "probability": f"{prob*100:.1f}%",
                "risk_level": risk_bucket(prob),
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
    """Return dashboard payload for homepage."""
    items = predict_top_events(top_n=12).get("critical_events", [])

    stats = {
        "objects_tracked": len(_tle_map),
        "active_threats": sum(1 for e in items if e["risk_level"] in ("High","Critical")),
        "predictions_made": len(items),
        "satellites_protected": len({e["satellite"] for e in items if e["risk_level"] in ("High","Critical")})
    }

    recent_alerts = [{
        "type": "conjunction" if e["risk_level"] in ("High","Critical") else "notice",
        "message": f"Conjunction detected: {e['satellite']}",
        "time": e["tca"]
    } for e in sorted(items, key=lambda x: x["tca"], reverse=True)[:5]]

    high_priority_tracking = []
    for e in sorted(items, key=lambda x: ({"Critical":3,"High":2,"Medium":1,"Low":0}[x["risk_level"]],
                                         float(x["probability"].rstrip("%"))), reverse=True)[:5]:
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
        "stats": stats,
        "recent_alerts": recent_alerts,
        "high_priority_tracking": high_priority_tracking
    }
