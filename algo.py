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

current_norm = normalize_multimetric(current_window)

for i in range(len(historical_series) - window_size + 1):

    window = historical_series[i:i+window_size]
    window_norm = normalize_multimetric(window)

    distance = np.linalg.norm(window_norm-current_norm)
    top.append((i, distance))

top.sort(key=lambda x: x[1])
temp = top[:500]
# print(temp)

dtw_res = []

for idx, _ in temp:

    window = historical_series[idx:idx + window_size]
    window_norm = normalize_multimetric(window)

    dist = dtw(current_norm, window_norm)
    dtw_res.append((idx, dist))



dtw_res.sort(key=lambda x: x[1])

top5 = dtw_res[:5]
print(top5)


future_windows = []

for idx, dist in top5:

    future = historical_series[
        idx + window_size :
        idx + 2 * window_size
    ]

    if len(future) == window_size:
        future_windows.append((idx, dist, future))


for idx, dist, future in future_windows:

    cpu = future[:,0]
    mem = future[:,1]
    req = future[:,2]

    cpu_slope = cpu[-1] - cpu[0]
    mem_slope = mem[-1] - mem[0]
    req_slope = req[-1] - req[0]

    print("Match:", idx)
    print("DTW:", dist)
    print("CPU slope:", cpu_slope)
    print("Memory slope:", mem_slope)
    print("Requests slope:", req_slope)
    print()