import os
import numpy as np
import matplotlib.pyplot as plt
import random

from data.loadData import load_data, process_data
from power_flow_with_dss_refactored import RadialNetwork

EXPERIMENT_NAME = "ieee_123_single_phase_power_flow"
EPSILON_VALUES = np.linspace(50, 1000, 10)

NETWORK_NAME = "ieee_123_single_phase"
N_NODES = 6
V0 = 12e2
T_SET = 100

ALPHA = 0.0
R = 0.01
X = 0.01

N_TRIALS = 10

def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd"),
    }

def load_redd_houses():

    paths = get_paths()

    files = [
        os.path.join(paths["redd"], f)
        for f in os.listdir(paths["redd"])
        if f.endswith(".mat") and "HF" not in f
    ]

    raw_data = []
    lengths = []

    # Load and Pre-Process
    for k in range(6):

        data = process_data(load_data(files[k]), "redd")

        if T_SET:
            data = {
                "Y": data["Y"][:T_SET],
                "X": data["X"][:,:T_SET],
            }

        raw_data.append(data)
        lengths.append(len(data["Y"]))

    houses = []

    for data in raw_data:

        Y = data["Y"]
        X = data["X"]

        P = Y
        B = np.max(X, axis=0)

        houses.append({
            "P": P,
            "B": B
        })

    return houses

def max_power_per_house(houses): return [np.max(house["P"]) for house in houses]

def assign_houses_to_loads_old(houses, target_loads, tol=0.10, max_iter=1000):
    house_powers_kw = [np.max(h["P"]) / 1000 for h in houses]

    assignments = {}   # key: max load (kW), value: total load vector
    house_counts = []  # new list

    for target in target_loads:
        lower = target * (1 - tol)
        upper = target * (1 + tol)

        total = 0.0
        chosen = []
        iters = 0

        while total < lower and iters < max_iter:
            i = random.randrange(len(houses))
            total += house_powers_kw[i]
            chosen.append(houses[i])
            iters += 1

            # If overshoot too much, restart
            if total > upper:
                total = 0.0
                chosen = []
                iters = 0

        if chosen:
            total_vector = np.sum([h["P"] for h in chosen], axis=0)
            max_load = np.max(total_vector) / 1000 # kw
            assignments[max_load] = total_vector

        house_counts.append(len(chosen))  # track number of houses

    return assignments, achieved_kw, house_counts

def assign_houses_to_loads(houses, target_loads, tol=0.10, max_iter=1000):

    house_powers_kw = [np.max(h["P"])/1000 for h in houses]

    assignments = {}
    house_counts = []

    for target in target_loads:
        lower = target * (1 - tol)
        upper = target * (1 + tol)

        total = 0.0
        chosen = []
        iters = 0

        while total < lower and iters < max_iter:
            i = random.randrange(len(houses))
            total += house_powers_kw[i]
            chosen.append(houses[i])
            iters += 1

            if total > upper:
                total = 0.0
                chosen = []

        if chosen:
            total_vector = np.sum([h["P"] for h in chosen], axis=0)
            max_load = np.max(total_vector) / 1000 # kw
            assignments.setdefault(max_load, []).append(total_vector)

        house_counts.append(len(chosen))

    return assignments, house_counts

# Instead of returning assignments, a list of the total load vectors

houses = load_redd_houses()
max_powers = max_power_per_house(houses)
unique_load_powers = [13.33, 6.67, 35, 46.67, 25]
assignments, house_counts = assign_houses_to_loads(houses, unique_load_powers)
assignment_keys = np.array(list(assignments.keys()))
# max powers = [348.08, 38.35, 781.45, 1845.93, 552,18, 313.24]

dss_filepath = os.path.join(os.path.dirname(__file__), 'ieee_123bus_1ph.dss')

network = RadialNetwork(NETWORK_NAME, build_from_dss=True, dss_filepath=dss_filepath)

P_vector = network.P
load_house_counts = []
for i, P_time_series in P_vector.items():
    # skip zero loads
    if np.allclose(P_time_series, 0):
        load_house_counts.append(0)
        continue

    # get target peak (kW)
    target_peak = np.max(P_time_series) / 1000

    # find closest assignment key
    idx = np.argmin(np.abs(assignment_keys - target_peak))
    closest_key = assignment_keys[idx]

    # select one profile from that bucket
    candidate_profiles = assignments[closest_key]
    chosen_profile = random.choice(candidate_profiles)

    # scale profile to match original peak
    scale = np.max(P_time_series) / np.max(chosen_profile)
    P_vector[i] = chosen_profile * scale
    load_house_counts.append(house_counts[idx])

network.dss_power_flow_step_by_step(tilde=False)
network.power_flow_results(display_results=True)

print('')

# Build Loads
# DSS 123Bus Original Loads - 8 Unique Loads
# Load 1 - 13.33 kw, 6.667 kVAr
# Load 2 - 6.667 kW, 3.333 kVAr
# Load 3 - 35 kw, 25 kVAr
# Load 4 - 70 kw, 50 kVAr
# Load 5 - 46.67 kw, 31.67 kVAr
# Load 6 - 46.67 kw, 33.33 kVAr
# Load 7 - 25 kw, 11.67 kVAr
# Load 8 - 81.67 kw, 60 kVAr

# 6 REDD Houses (W): [348.08, 38.35, 781.45, 1845.93, 552,18, 313.24]
# Combine the REDD houses to get loads with same magnitudes as original 123Bus system loads.

# For each Load, add random house from set of 6 houses until load is within +/- 10% of desired list