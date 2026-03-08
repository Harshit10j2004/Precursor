import numpy as np
from tslearn.metrics import dtw

def normalize_multimetric(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    std[std == 0] = 1

    return (data - mean) / std

np.random.seed(40)
historical_series = np.random.randint(30,90,(175000,3))
window_size = 20

windows = []

current_window = np.random.randint(30,90,(20,3))

best_distance = float("inf")
best_index = None

top = []
# print(historical_series)

current_norm = normalize_multimetric(current_window)

for i in range(len(historical_series) - window_size + 1):

    window = historical_series[i:i+window_size]
    window_norm = normalize_multimetric(window)

    distance = np.linalg.norm(window_norm-current_norm)
    top.append((i, distance))

top.sort(key=lambda x: x[1])
temp = top[:500]
# print(temp)

for idx, _ in temp:

    window = historical_series[idx:idx + window_size]
    window_norm = normalize_multimetric(window)

    dist = dtw(current_norm, window_norm)

    if dist < best_distance:
        best_distance = dist
        best_index = idx

print(best_index)
print(best_distance)

next_window = historical_series[
    best_index + window_size :
    best_index + 2 * window_size
]

print(next_window)