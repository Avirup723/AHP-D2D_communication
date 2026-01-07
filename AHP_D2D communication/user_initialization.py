import numpy as np

def assign_users(user_type="eMBB", num_users=10):
    users = []
    for i in range(num_users):
        user = {
            "id": i,
            "x": np.random.uniform(0, 1000),
            "y": np.random.uniform(0, 1000),
            "tech": "5G",  # default
            "type": user_type
        }
        users.append(user)
    return users