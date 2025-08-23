import math
from typing import Dict, Tuple, List

MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137

def parse_tle_file(tle_text: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Parse a whole TLE text blob into {name: (name, L1, L2)}.
    Assumes classic 3-line blocks.
    """
    lines = [ln.rstrip("\n") for ln in tle_text.splitlines()]
    out: Dict[str, Tuple[str, str, str]] = {}
    i = 0
    while i < len(lines):
        name = lines[i].strip() if i < len(lines) else ""
        l1 = lines[i+1].strip() if i+1 < len(lines) else ""
        l2 = lines[i+2].strip() if i+2 < len(lines) else ""
        if name and l1.startswith("1 ") and l2.startswith("2 "):
            out[name] = (name, l1, l2)
            i += 3
        else:
            i += 1
    return out

def load_tle_map(tle_path: str) -> Dict[str, Tuple[str, str, str]]:
    with open(tle_path, "r") as f:
        return parse_tle_file(f.read())

def find_tle_block(tle_map: Dict[str, Tuple[str, str, str]], name: str) -> str:
    """
    Case-insensitive lookup by exact name; fallback to contains-substring match.
    """
    key_upper = name.strip().upper()
    for k, (nm, l1, l2) in tle_map.items():
        if k.upper() == key_upper:
            return f"{nm}\n{l1}\n{l2}"
    # fallback: substring search
    for k, (nm, l1, l2) in tle_map.items():
        if key_upper in k.upper():
            return f"{nm}\n{l1}\n{l2}"
    return f"TLE not found for '{name}' in local file"

def mean_motion_rev_per_day(line2: str) -> float:
    # TLE L2 columns 53–63 are mean motion (rev/day)
    return float(line2[52:63])

def semimajor_axis_km_from_mean_motion(n_rev_per_day: float) -> float:
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    a_km = (MU_EARTH_KM3_S2 / (n_rad_s ** 2)) ** (1.0 / 3.0)
    return a_km

def circular_velocity_kms(a_km: float) -> float:
    return math.sqrt(MU_EARTH_KM3_S2 / a_km)

def quick_altitude_velocity_from_tle(tle_block: str) -> Tuple[float, float]:
    """
    Estimate altitude (km) & circular velocity (km/s) from L2 mean motion.
    Good enough for dashboard cards.
    """
    lines = [ln.strip() for ln in tle_block.splitlines() if ln.strip()]
    if len(lines) < 3:
        return (0.0, 0.0)
    l2 = lines[2]
    try:
        n = mean_motion_rev_per_day(l2)
        a = semimajor_axis_km_from_mean_motion(n)
        alt = max(0.0, a - R_EARTH_KM)
        vel = circular_velocity_kms(a)
        return (alt, vel)
    except Exception:
        return (0.0, 0.0)

def risk_bucket(prob: float) -> str:
    if prob < 0.3: return "Low"
    if prob < 0.6: return "Medium"
    if prob < 0.8: return "High"
    return "Critical"

def normalize_probabilities(vals: List[float]) -> List[float]:
    m = max(vals) if vals else 0.0
    if m <= 0: return [0.0 for _ in vals]
    return [min(1.0, max(0.0, v / m)) for v in vals]
