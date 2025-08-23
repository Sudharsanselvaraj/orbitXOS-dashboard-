import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from sgp4.api import Satrec, jday

LEO_CA_THRESHOLD_KM = 5.0
GEO_CA_THRESHOLD_KM = 25.0

def _sanitize(vec: List[float]) -> List[float]:
    return [0.0 if math.isinf(x) or math.isnan(x) else x for x in vec]

def _normalize_tle_block(tle_text: str) -> Tuple[str, str, str]:
    lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
    if len(lines) == 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
        return lines[0], lines[1], lines[2]
    if len(lines) == 2 and lines[0].startswith("1 ") and lines[1].startswith("2 "):
        return "UNKNOWN", lines[0], lines[1]
    raise ValueError("Invalid TLE format")

def _validate_tle(tle_text: str) -> Tuple[str, str, str]:
    name, L1, L2 = _normalize_tle_block(tle_text)
    if len(L1) < 68 or len(L2) < 68:
        raise ValueError("TLE lines too short")
    return name, L1, L2

def _regime_from_mean_motion(mm_rev_per_day: float) -> str:
    if mm_rev_per_day > 10: return "LEO"
    if mm_rev_per_day < 2: return "GEO"
    return "MEO"

def _propagate_positions(tle_text: str, minutes: int, step_s: int) -> List[Dict]:
    _, L1, L2 = _normalize_tle_block(tle_text)
    sat = Satrec.twoline2rv(L1, L2)
    t0 = datetime.utcnow()
    out = []
    for k in range(0, minutes*60 + 1, step_s):
        t = t0 + timedelta(seconds=k)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond/1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            out.append({"t": t.isoformat()+"Z", "r": _sanitize(r), "v": _sanitize(v)})
    return out

def _nearest_approach_km(path_a: List[Dict], path_b: List[Dict]) -> Tuple[float, Dict]:
    n = min(len(path_a), len(path_b))
    dmin = float("inf")
    kmin = -1
    for i in range(n):
        ax, ay, az = path_a[i]["r"]
        bx, by, bz = path_b[i]["r"]
        d = math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)
        if d < dmin:
            dmin = d
            kmin = i
    meta = {}
    if kmin >= 0:
        meta = {"time": path_a[kmin]["t"], "index": kmin}
    return dmin, meta

def _adjust_mean_motion_l2(line2: str, delta_rev_per_day: float) -> str:
    current_mm = float(line2[52:63])
    new_mm = current_mm + delta_rev_per_day
    mm_str = f"{new_mm:11.8f}"
    l2 = line2[:52] + mm_str + line2[63:]
    if len(l2) < 69:  # rebuild checksum roughly
        l2 = l2.ljust(68) + "0"
    return l2

def _generate_safe_tle(original_tle: str, dv_mps: float) -> str:
    name, L1, L2 = _normalize_tle_block(original_tle)
    delta_rev_per_day = dv_mps * 0.00005
    new_L2 = _adjust_mean_motion_l2(L2, delta_rev_per_day)
    return f"{name}\n{L1}\n{new_L2}"

def predict_safe_path(
    satellite_tle: str,
    debris_tle: str,
    horizon_minutes: int = 60,
    step_seconds: int = 30
) -> Dict[str, Any]:

    sat_name, sat_l1, sat_l2 = _validate_tle(satellite_tle)
    deb_name, deb_l1, deb_l2 = _validate_tle(debris_tle)

    try:
        mm_sat = float(sat_l2[52:63])
        regime = _regime_from_mean_motion(mm_sat)
    except Exception:
        regime = "UNKNOWN"

    step_s = step_seconds if regime != "GEO" else max(300, step_seconds)

    sat_path = _propagate_positions(f"{sat_name}\n{sat_l1}\n{sat_l2}", horizon_minutes, step_s)
    deb_path = _propagate_positions(f"{deb_name}\n{deb_l1}\n{deb_l2}", horizon_minutes, step_s)

    dmin_km, meta = _nearest_approach_km(sat_path, deb_path)
    threshold = LEO_CA_THRESHOLD_KM if regime == "LEO" else GEO_CA_THRESHOLD_KM
    risky = dmin_km <= threshold if dmin_km != float("inf") else False

    if risky:
        maneuver = {"type": "retrograde_burn", "recommended_dv_mps": 1.0,
                    "note": "Small along-track tweak to desynchronize TCA."}
        safe_tle = _generate_safe_tle(f"{sat_name}\n{sat_l1}\n{sat_l2}", maneuver["recommended_dv_mps"])
    else:
        maneuver = {"type": "no_action", "recommended_dv_mps": 0.0,
                    "note": "Separation above threshold."}
        safe_tle = f"{sat_name}\n{sat_l1}\n{sat_l2}"

    return {
        "risk": {
            "min_distance_km": None if dmin_km == float("inf") else round(dmin_km, 3),
            "tca": meta.get("time"),
            "regime": regime,
            "threshold_km": threshold,
            "risky": risky
        },
        "maneuver": maneuver,
        "tle_output": {
            "satellite_tle": f"{sat_name}\n{sat_l1}\n{sat_l2}",
            "debris_tle": f"{deb_name}\n{deb_l1}\n{deb_l2}",
            "predicted_safe_tle": safe_tle
        },
        "paths": {
            "satellite_xyz_km": [p["r"] for p in sat_path],
            "debris_xyz_km": [p["r"] for p in deb_path]
        }
    }
