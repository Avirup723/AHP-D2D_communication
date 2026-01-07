import numpy as np
from config import BS_TX_POWER_DBM, FREQ_TECH_HZ

def _path_loss(d_m: float, f_hz: float, environment: str = "urban") -> float:
   
    d = max(1.0, float(d_m))
    f_GHz = float(f_hz) / 1e9
    if environment == "urban":
        return 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(f_GHz)
    elif environment == "indoor":
        return 38.46 + 20.0 * np.log10(d) + 20.0 * np.log10(f_GHz)
    else:
        return 32.4 + 23.0 * np.log10(d) + 20.0 * np.log10(f_GHz)

def _nearest_distance(user_xy: np.ndarray, bs_locs: list[tuple[float, float]]) -> float:
    if not bs_locs:
        return 1e9
    arr = np.asarray(bs_locs, dtype=float)
    return float(np.linalg.norm(arr - user_xy[None, :], axis=1).min())

def calculate_rsrp_values(user: dict,
                          bs_5g_locs: list[tuple[float, float]],
                          bs_lte_locs: list[tuple[float, float]],
                          d2d_distance: float) -> dict:
  
    user_pos = np.array([user["x"], user["y"]], dtype=float)

    d5g  = _nearest_distance(user_pos, bs_5g_locs)
    dlte = _nearest_distance(user_pos, bs_lte_locs)
    dd2d = max(1.0, float(d2d_distance))

    pl_5g  = _path_loss(d5g,  FREQ_TECH_HZ["5G"],  environment="urban")
    pl_lte = _path_loss(dlte, FREQ_TECH_HZ["LTE"], environment="urban")
    pl_d2d = _path_loss(dd2d, FREQ_TECH_HZ["D2D"], environment="indoor")

    sh_5g_db  = np.random.normal(loc=0.0, scale=6.0)
    sh_lte_db = np.random.normal(loc=0.0, scale=6.0)
    sh_d2d_db = np.random.normal(loc=0.0, scale=4.0)

    rsrp_5g  = BS_TX_POWER_DBM["5G"]  - pl_5g  + sh_5g_db
    rsrp_lte = BS_TX_POWER_DBM["LTE"] - pl_lte + sh_lte_db
    rsrp_d2d = BS_TX_POWER_DBM["D2D"] - pl_d2d + sh_d2d_db

    return {"5G": rsrp_5g, "LTE": rsrp_lte, "D2D": rsrp_d2d}
