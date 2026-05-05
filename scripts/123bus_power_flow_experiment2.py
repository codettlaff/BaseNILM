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

    houses = []

    for k in range(6):
        data = process_data(load_data(files[k]), "redd")
        P = data["Y"]
        if T_SET:
            P = P[:T_SET]
        houses.append(P) # List of 1D Numpy Arrays

    return houses

def assign_houses_to_load(houses, desired_load_power):

    num_houses = 0
    n = len(houses[0])
    load_profile = np.zeros(n)

    while np.max(load_profile) < desired_load_power:
        i = random.randrange(len(houses))
        temp_profile = load_profile + houses[i]
        temp = np.max(temp_profile)
        if temp > desired_load_power:
            num_houses += 1
            factor = desired_load_power / np.max(temp)
            scaled_load_profile = houses[i] * factor
            load_profile = load_profile + scaled_load_profile
            break
        else:
            num_houses += 1
            load_profile += houses[i]

    return num_houses, load_profile

# Read Original IEEE123 Bus Files
paths = get_paths()
ieee123_bus_original_filepath = os.path.join(paths["ieee_123bus"], 'Master.dss')
ieee123_bus_modified_filepath = os.path.join(paths["scripts"], f'{NETWORK_NAME}.dss')

network = RadialNetwork(name=NETWORK_NAME, dss_filepath=ieee123_bus_original_filepath)

houses = load_redd_houses()

P_loads_original = network.P
P_loads_modified = {}
for node, profile in P_loads_original.items():
    desired_load = np.max(profile)
    num_houses, load_profile = assign_houses_to_load(houses, desired_load)
    P_loads_modified[node] = load_profile

num_houses, load_profile = assign_houses_to_load(houses, 1.5*1e3)

print('')