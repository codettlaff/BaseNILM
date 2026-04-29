import os
import numpy as np
import matplotlib.pyplot as plt

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

houses = load_redd_houses()
max_powers = max_power_per_house(houses)
# max powers = [348.08, 38.35, 781.45, 1845.93, 552,18, 313.24]

print('')

# Build Loads
# DSS 123Bus Original Loads
# Load 1 - 13.33 kw, 6.667 kVAr
# Load 2 - 6.667 kW, 3.333 kVAr
# Load 3 - 35 kw, 25 kVAr
# Load 4 - 70 kw, 50 kVAr
# Load 5 - 46.67 kw, 31.67 kVAr
# Load 6 - 46.67 kw, 33.33 kVAr
# Load 7 - 25 kw, 11.67 kVAr
# Load 8 - 81.67 kw, 60 kVAr

# 6 REDD Houses (W): [348.08, 38.35, 781.45, 1845.93, 552,18, 313.24]
