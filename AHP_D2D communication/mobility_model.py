import numpy as np

def move_users(users, speed,
               area=(1000.0, 1000.0),
               dt=1.0,
               pause_range=(0.0, 2.0),
               waypoint_threshold=1e-6):
   
    W, H = float(area[0]), float(area[1])

    for u in users:
        if "x" not in u or "y" not in u:
            raise ValueError("Each user must have 'x' and 'y' keys.")

        if "wx" not in u or "wy" not in u:
            u["wx"] = np.random.uniform(0.0, W)
            u["wy"] = np.random.uniform(0.0, H)

        if "pause" not in u:
            u["pause"] = 0.0

        if "speed" not in u or u["speed"] is None:
            u["speed"] = float(speed)  # per-user speed

        if u["pause"] > 0.0:
            u["pause"] = max(0.0, u["pause"] - dt)
            continue

        x, y = float(u["x"]), float(u["y"])
        wx, wy = float(u["wx"]), float(u["wy"])

        dx = wx - x
        dy = wy - y
        dist = float(np.hypot(dx, dy))

        step = float(u["speed"]) * dt

        if dist <= max(waypoint_threshold, step):
            u["x"], u["y"] = wx, wy

            # Pick a new waypoint
            u["wx"] = np.random.uniform(0.0, W)
            u["wy"] = np.random.uniform(0.0, H)

            # Optional pause at waypoint
            pmin, pmax = pause_range
            u["pause"] = float(np.random.uniform(pmin, pmax))

            continue

        ux = dx / dist
        uy = dy / dist
        u["x"] = x + ux * step
        u["y"] = y + uy * step

        u["x"] = float(np.clip(u["x"], 0.0, W))
        u["y"] = float(np.clip(u["y"], 0.0, H))

    return users
