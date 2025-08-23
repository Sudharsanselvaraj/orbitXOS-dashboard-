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
