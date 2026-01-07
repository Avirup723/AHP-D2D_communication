import os, json, csv
from typing import Dict, Any

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _flat_row(method: str, slice_name: str, x_key: str, x_val: Any, bucket: Dict[str, Any], i: int):
    """Build one row from results dict at index i."""
    return {
        "method": method,
        "slice": slice_name,
        "x_type": x_key,            # "speed" or "users"
        "x_value": bucket[x_key][i],
        "throughput": bucket["throughput"][i],
        "throughput_ci": bucket["throughput_ci"][i],
        "ber": bucket["ber"][i],
        "ber_ci": bucket["ber_ci"][i],
        "latency": bucket["latency"][i],
        "latency_ci": bucket["latency_ci"][i],
        "jitter": bucket["jitter"][i],
        "jitter_ci": bucket["jitter_ci"][i],
        "handover": bucket["handover"][i],
        "ho_pairs": json.dumps(bucket["ho_pairs"][i]),
    }

def _write_csv(rows, out_path: str):
    if not rows:
        return
    _ensure_dir(os.path.dirname(out_path) or ".")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def save_results_to_csv(results_speed: Dict, results_users: Dict = None, out_dir: str = "results_csv"):
  
    _ensure_dir(out_dir)

    # -------- speed sweep --------
    rows_speed = []
    for method, by_slice in results_speed.items():
        for slice_name, bucket in by_slice.items():
            n = len(bucket.get("speed", []))
            for i in range(n):
                rows_speed.append(_flat_row(method, slice_name, "speed", bucket["speed"][i], bucket, i))

    _write_csv(rows_speed, os.path.join(out_dir, "speed_results.csv"))

    # -------- users sweep (optional) --------
    if results_users:
        rows_users = []
        for method, by_slice in results_users.items():
            for slice_name, bucket in by_slice.items():
                n = len(bucket.get("users", []))
                for i in range(n):
                    rows_users.append(_flat_row(method, slice_name, "users", bucket["users"][i], bucket, i))
        _write_csv(rows_users, os.path.join(out_dir, "users_results.csv"))
