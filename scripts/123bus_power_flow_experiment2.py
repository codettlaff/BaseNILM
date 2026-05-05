import os
import numpy as np
import random

from data.loadData import load_data, process_data
from power_flow_with_dss_refactored2 import RadialNetwork

EXPERIMENT_NAME = "ieee_123_single_phase_power_flow"
EPSILON_VALUES = np.linspace(50, 1000, 10)

NETWORK_NAME = "redd_house_neighborhood"
N_NODES = 6
V0 = 12e2
T_SET = 100

ALPHA = 0.0
R = 0.01
X = 0.01

N_TRIALS = 10

B = 5e3 # HVAC Peak
# B = 9.6 k# EV Charger Peak

def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "ieee_123bus": os.path.join(base, "data", "ieee_123bus"),
        "scripts": os.path.join(base, "scripts"),
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

# Read Original IEEE123 Bus Files
paths = get_paths()
ieee123_bus_original_filepath = os.path.join(paths["ieee_123bus"], 'Master.dss')
ieee123_bus_modified_filepath = os.path.join(paths["scripts"], f'{NETWORK_NAME}.dss')

network = RadialNetwork(name=NETWORK_NAME, dss_filepath=ieee123_bus_original_filepath)

houses = load_redd_houses()
max_powers = max_power_per_house(houses)

print('')