import numpy as np
import random
from collections import deque, defaultdict
from results_csv import save_results_to_csv


from config import (
    assign_users,         # default generator (uses NUM_USERS)
    BASE_STATIONS,
    user_phy_profiles,
    user_weights,
    SPEEDS, RUNS_PER_SPEED,
    AREA_SIZE, USER_TYPE  # for custom user-count sweep
)
try:
    from config import QOS_WEIGHT_BIAS_DB
except Exception:
    QOS_WEIGHT_BIAS_DB = 12.0
try:
    from config import PAIR_HOLD_STEPS
except Exception:
    PAIR_HOLD_STEPS = 30
try:
    from config import HO_THRESHOLDS_TECH
except Exception:
    HO_THRESHOLDS_TECH = {"5G": 0.6, "LTE": 0.6, "D2D": 0.6}

from mobility_model import move_users
from rsrp_model import calculate_rsrp_values
from phy_simulation import simulate_phy_link



# ---------------- inverse-sigmoid scoring ----------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: float) -> float:
    return np.log(p / (1.0 - p + 1e-12))

def inv_sigmoid_of_rsrp(rsrp_dbm: float) -> float:
    """logit(sigmoid(RSRP)) == RSRP (explicit; clipped for safety)."""
    x = float(np.clip(rsrp_dbm, -150.0, 10.0))
    p = _sigmoid(x)
    return float(_logit(p))


# ---------------- D2D pairing  ----------------
def _make_random_pairs(users):
    ids = [u["id"] for u in users]
    random.shuffle(ids)
    peer_of = {}
    for i in range(0, len(ids), 2):
        if i == len(ids) - 1:
            a, b = ids[i - 1], ids[i]
        else:
            a, b = ids[i], ids[i + 1]
        peer_of[a] = b
        peer_of[b] = a
    return peer_of

def _compute_pair_distances(users, peer_of):
    pos = {u["id"]: (u["x"], u["y"]) for u in users}
    d2d_dist = {}
    for uid, vid in peer_of.items():
        ax, ay = pos[uid]
        bx, by = pos[vid]
        d2d_dist[uid] = float(np.hypot(ax - bx, ay - by))
    return d2d_dist


def assign_users_n(num_users: int, user_type: str):
    return [
        {
            "id": i,
            "x": random.uniform(0, AREA_SIZE[0]),
            "y": random.uniform(0, AREA_SIZE[1]),
            "type": user_type,
            "tech": "5G"
        }
        for i in range(num_users)
    ]




# ---------------- Main simulation ----------------
slice_types = ["eMBB", "URLLC", "mMTC"]  # which slices to run
techs = ["5G", "LTE", "D2D"]
pair_labels = [f"{a}->{b}" for a in techs for b in techs if a != b]

USERS_LIST = [20, 40, 60, 80, 100]
FIXED_SPEED_FOR_USERS_SWEEP = 6.0  # m/s

results_speed = {"Our method": {}, "RSRP-based method": {}}
results_users = {"Our method": {}, "RSRP-based method": {}}

TOTAL_STEPS = len(slice_types) * (len(SPEEDS) * RUNS_PER_SPEED + len(USERS_LIST) * RUNS_PER_SPEED)
done_steps = 0
def _maybe_print_progress():
    global done_steps
    done_steps += 1
    pct = 100.0 * done_steps / max(1, TOTAL_STEPS)
    stride = max(1, TOTAL_STEPS // 20)   # ~5% granularity
    if (done_steps % stride == 0) or (done_steps == TOTAL_STEPS):
        print(f"[progress] {done_steps}/{TOTAL_STEPS} steps  ({pct:5.1f}%)", flush=True)


def run_setting(users, speed, user_type, steps_per_setting):
    peer_of = _make_random_pairs(users)
    d2d_dist = _compute_pair_distances(users, peer_of)

    prev_tech_qos, prev_tech_rsrp,  = [], []

   
    for u in users:
        rsrp0 = calculate_rsrp_values(
            u,
            bs_5g_locs=BASE_STATIONS["5G"],
            bs_lte_locs=BASE_STATIONS["LTE"],
            d2d_distance=d2d_dist.get(u["id"], 50.0)
        )
        qos_scores0 = {t: inv_sigmoid_of_rsrp(rsrp0[t]) + QOS_WEIGHT_BIAS_DB * user_weights[user_type][t]
                       for t in techs}
        seed_qos = max(qos_scores0, key=qos_scores0.get)
        u["tech"] = seed_qos
        prev_tech_qos.append(seed_qos)

        seed_rsrp = max(rsrp0, key=rsrp0.get)
        prev_tech_rsrp.append(seed_rsrp)

    qos_per_run, rsrp_per_run= [], []
    ho_qos_total, ho_rsrp_total= 0, 0
    ho_qos_pairs   = {p: 0 for p in pair_labels}
    ho_rsrp_pairs  = {p: 0 for p in pair_labels}
   

    for step_idx in range(steps_per_setting):
        if step_idx > 0 and (step_idx % PAIR_HOLD_STEPS == 0):
            peer_of = _make_random_pairs(users)

        users = move_users(users, speed)
        d2d_dist = _compute_pair_distances(users, peer_of)

        qos_this, rsrp_this = [], []

        for idx, user in enumerate(users):
            rsrp_map = calculate_rsrp_values(
                user,
                bs_5g_locs=BASE_STATIONS["5G"],
                bs_lte_locs=BASE_STATIONS["LTE"],
                d2d_distance=d2d_dist.get(user["id"], 50.0)
            )

        
            qos_scores = {t: inv_sigmoid_of_rsrp(rsrp_map[t]) + QOS_WEIGHT_BIAS_DB * user_weights[user_type][t]
                          for t in techs}
            best_qos_t = max(qos_scores, key=qos_scores.get)
            curr_qos_t = user["tech"]
            delta_qos  = qos_scores[best_qos_t] - qos_scores[curr_qos_t]

            chosen_qos = curr_qos_t
            if delta_qos > HO_THRESHOLDS_TECH.get(best_qos_t, 0.6):
                chosen_qos = best_qos_t

            if chosen_qos != prev_tech_qos[idx]:
                ho_qos_total += 1
                ho_qos_pairs[f"{prev_tech_qos[idx]}->{chosen_qos}"] += 1
                prev_tech_qos[idx] = chosen_qos

            prof_q = user_phy_profiles[chosen_qos][user_type]
            ber, thrpt, lat, jit = simulate_phy_link(
                rsrp_dbm=rsrp_map[chosen_qos],
                modulation_order=prof_q["mod_order"],
                bandwidth=prof_q["bw"],
                ldpc_gain=prof_q["ldpc_gain"],
                speed=speed,
                tech=chosen_qos,
                user_id=user["id"],
                packet_size_bytes=prof_q["pkt_size"],
                slice_type=user_type,
            )
            if not np.isfinite(ber):     ber = 1.0
            if not np.isfinite(thrpt):   thrpt = 0.0
            if not np.isfinite(lat):     lat = 1.0
            if not np.isfinite(jit):     jit = 0.0
            qos_this.append([thrpt, ber, lat, jit])
            user["tech"] = chosen_qos

            
            curr_rsrp_t = prev_tech_rsrp[idx]
            best_rsrp_t = max(rsrp_map, key=rsrp_map.get)
            delta_rsrp = rsrp_map[best_rsrp_t] - rsrp_map[curr_rsrp_t]

            chosen_rsrp = curr_rsrp_t
            if delta_rsrp > HO_THRESHOLDS_TECH.get(best_rsrp_t, 0.6):
                chosen_rsrp = best_rsrp_t

            if chosen_rsrp != prev_tech_rsrp[idx]:
                ho_rsrp_total += 1
                ho_rsrp_pairs[f"{prev_tech_rsrp[idx]}->{chosen_rsrp}"] += 1
                prev_tech_rsrp[idx] = chosen_rsrp

            prof_b = user_phy_profiles[chosen_rsrp][user_type]
            ber2, thrpt2, lat2, jit2 = simulate_phy_link(
                rsrp_dbm=rsrp_map[chosen_rsrp],
                modulation_order=prof_b["mod_order"],
                bandwidth=prof_b["bw"],
                ldpc_gain=prof_b["ldpc_gain"],
                speed=speed,
                tech=chosen_rsrp,
                user_id=user["id"],
                packet_size_bytes=prof_b["pkt_size"],
                slice_type=user_type,
            )
            if not np.isfinite(ber2):    ber2 = 1.0
            if not np.isfinite(thrpt2):  thrpt2 = 0.0
            if not np.isfinite(lat2):    lat2 = 1.0
            if not np.isfinite(jit2):    jit2 = 0.0
            rsrp_this.append([thrpt2, ber2, lat2, jit2])

            

        qos_per_run.append(np.mean(qos_this, axis=0))
        rsrp_per_run.append(np.mean(rsrp_this, axis=0))
       

        _maybe_print_progress()

    def ci95(arr):
        n = max(1, arr.shape[0])
    def ci95(arr):
        n = max(1, arr.shape[0])
        return (1.96 * arr.std(axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros(arr.shape[1])

    qos_arr, rsrp_arr = np.array(qos_per_run), np.array(rsrp_per_run)
    qos_mean   = qos_arr.mean(axis=0)   if qos_arr.size    else np.zeros(4)
    rsrp_mean  = rsrp_arr.mean(axis=0)  if rsrp_arr.size   else np.zeros(4)
    
    qos_ci   = ci95(qos_arr)    if qos_arr.size    else np.zeros(4)
    rsrp_ci  = ci95(rsrp_arr)   if rsrp_arr.size   else np.zeros(4)
   

    return (
        (qos_mean,   qos_ci,   ho_qos_total,   ho_qos_pairs),
        (rsrp_mean,  rsrp_ci,  ho_rsrp_total,  ho_rsrp_pairs),
        
    )

# ===================== DIFF. SPEED  =====================
for user_type in slice_types:
    def empty_metrics():
        return {
            "speed": [],
            "throughput": [], "ber": [], "latency": [], "jitter": [],
            "throughput_ci": [], "ber_ci": [], "latency_ci": [], "jitter_ci": [],
            "handover": [], "ho_pairs": []
        }
    for m in ["Our method", "RSRP-based method"]:
        results_speed[m][user_type] = empty_metrics()

    for speed in SPEEDS:
        users = assign_users()
        (q_mean, q_ci, q_ho, q_pairs), (r_mean, r_ci, r_ho, r_pairs) = run_setting(
            users, speed, user_type, steps_per_setting=RUNS_PER_SPEED
        )

        for method, mean, ci, ho_total, ho_pairs in [
            ("Our method", q_mean, q_ci, q_ho, q_pairs),
            ("RSRP-based method", r_mean, r_ci, r_ho, r_pairs),
            
        ]:
            res = results_speed[method][user_type]
            res["speed"].append(speed)
            res["throughput"].append(float(mean[0]))
            res["ber"].append(float(max(mean[1], 1e-12)))
            res["latency"].append(float(mean[2]))
            res["jitter"].append(float(mean[3]))
            res["throughput_ci"].append(float(ci[0]))
            res["ber_ci"].append(float(ci[1]))
            res["latency_ci"].append(float(ci[2]))
            res["jitter_ci"].append(float(ci[3]))
            res["handover"].append(int(ho_total))
            res["ho_pairs"].append(dict(ho_pairs))


# ===================== DIFF. USERS =====================
USERS_LIST = [20, 40, 60, 80, 100]
FIXED_SPEED_FOR_USERS_SWEEP = 6.0  # m/s

for user_type in slice_types:

    def empty_umetrics():
        return {
            "users": [],
            "throughput": [], "ber": [], "latency": [], "jitter": [],
            "throughput_ci": [], "ber_ci": [], "latency_ci": [], "jitter_ci": [],
            "handover": [], "ho_pairs": []
        }

    for m in ["Our method", "RSRP-based method"]:
        results_users[m][user_type] = empty_umetrics()

    for n_users in USERS_LIST:
        users = assign_users_n(n_users, user_type if USER_TYPE is None else USER_TYPE)
        (q_mean, q_ci, q_ho, q_pairs),         (r_mean, r_ci, r_ho, r_pairs) = run_setting(
            users, FIXED_SPEED_FOR_USERS_SWEEP, user_type, steps_per_setting=RUNS_PER_SPEED
        )

        for method, mean, ci, ho_total, ho_pairs in [
            ("Our method",        q_mean, q_ci, q_ho, q_pairs),
            ("RSRP-based method", r_mean, r_ci, r_ho, r_pairs),
           
        ]:
            res = results_users[method][user_type]
            res["users"].append(int(n_users))
            res["throughput"].append(float(mean[0]))
            res["ber"].append(float(max(mean[1], 1e-12)))
            res["latency"].append(float(mean[2]))
            res["jitter"].append(float(mean[3]))
            res["throughput_ci"].append(float(ci[0]))
            res["ber_ci"].append(float(ci[1]))
            res["latency_ci"].append(float(ci[2]))
            res["jitter_ci"].append(float(ci[3]))
            res["handover"].append(int(ho_total))
            res["ho_pairs"].append(dict(ho_pairs))

print("[done] Simulation complete.")


save_results_to_csv(results_speed, results_users, out_dir="results_csv")


#plot_qos_results(results_speed, results_users)


