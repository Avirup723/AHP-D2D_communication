import random
import numpy as np

RNG_SEED = 123
if RNG_SEED is not None:
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

# ---------------- Simulation area (meters) ----------------
AREA_SIZE = (1600, 1600)  # (width, height)

# ---------------- User / slice selection ----------------
#USER_TYPE = ["eMBB", "URLLC", "mMTC"] # "eMBB", "URLLC", or "mMTC"
USER_TYPE = ["eMBB"] 
# ---------------- Population & time ----------------
NUM_USERS = 40
RUNS_PER_SPEED = 300    # timesteps per speed
SPEEDS = [2, 4, 6, 8, 10]      # m/s, plotted as discrete points
#PAIR_HOLD_STEPS = 2        

# ---------------- Base Station deployment ----------------
# Number of sites per RAT (randomly placed)
NUM_BS = {
    "5G": 4,
    "LTE": 1,
}

def _place_random_bs(n, area_size=AREA_SIZE, margin=30):
    """Uniform random BS placement inside AREA_SIZE, with a border margin."""
    w, h = area_size
    return [
        (random.uniform(margin, w - margin), random.uniform(margin, h - margin))
        for _ in range(n)
    ]

BASE_STATIONS = {
    "5G": _place_random_bs(NUM_BS["5G"]),
    "LTE": _place_random_bs(NUM_BS["LTE"]),
}

FREQ_TECH_HZ = {
    "5G":  6e9,   # Sub-6 NR
    "LTE": 2.1e9,   # LTE Band
    "D2D": 2.4e9,   # ISM for D2D
}

# Transmit powers (dBm)
BS_TX_POWER_DBM = {
    "5G":  15.0,    # macro NR
    "LTE": 15.0,    # macro LTE
    "D2D": 3.0,    # UE-to-UE
}

NOISE_FIGURE_DB = 7.0
THERMAL_NOISE_DBM_PER_HZ = -174.0


# These weights are applied to the (requested) inv_sigmoid(sigmoid(RSRP)) == RSRP score.
user_weights = {
    "eMBB": {"5G": 0.41,  "LTE": 0.199, "D2D": 0.24},
    "URLLC":{"5G": 0.47,  "LTE": 0.205, "D2D": 0.27},
   # "mMTC": {"5G": 0.224, "LTE": 0.262, "D2D": 0.495},
    "mMTC": {"5G": 0.424, "LTE": 0.262, "D2D": 0.295},
}

# ---------------- Handover thresholds PER TECHNOLOGY ----------------
# Increase to reduce HOs; decrease to make HOs easier.
HO_THRESHOLDS_TECH = {
    "5G":  10,
    "LTE": 10,
    "D2D": 10,
}


user_phy_profiles = {
    "5G": {
        "eMBB":  {"mod_order": 512, "freq": 6e9, "bw": 100e6, "ldpc_gain": 0.10, "pkt_size": 1500},
        "URLLC": {"mod_order":  128, "freq": 6e9, "bw":  40e6, "ldpc_gain": 0.08, "pkt_size":  1500},
        "mMTC":  {"mod_order":  64, "freq": 6e9, "bw":  10e6, "ldpc_gain": 0.10, "pkt_size":  1500},
    },
    "LTE": {
        "eMBB":  {"mod_order":  64, "freq": 2.1e9, "bw": 5e6, "ldpc_gain": 0.05, "pkt_size": 1500},
        "URLLC": {"mod_order":  16, "freq": 2.1e9, "bw":  3e6, "ldpc_gain": 0.04, "pkt_size":  1500},
        "mMTC":  {"mod_order":   8, "freq": 1.8e9, "bw":   1e6, "ldpc_gain": 0.05, "pkt_size":  1500},
    },
    "D2D": {
        "eMBB":  {"mod_order": 256, "freq": 2.4e9, "bw":  20e6, "ldpc_gain": 0.08, "pkt_size": 1500},
        "URLLC": {"mod_order":  64, "freq": 2.4e9, "bw":   10e6, "ldpc_gain": 0.02, "pkt_size":  1500},
        "mMTC":  {"mod_order":  16, "freq": 2.4e9, "bw":   3e6, "ldpc_gain": 0.08, "pkt_size":  1500},
    },
}

# ---------------- User placement ----------------
def assign_users(user_type: str = USER_TYPE, num_users: int = NUM_USERS):
    """
    Create 'num_users' random users in AREA_SIZE.
    Initial 'tech' is a placeholder; main_simulation seeds it properly before first run.
    """
    w, h = AREA_SIZE
    users = []
    for i in range(num_users):
        users.append({
            "id": i,
            "x": random.uniform(0.0, w),
            "y": random.uniform(0.0, h),
            "type": user_type,
            "tech": "5G",  
        })
    return users
