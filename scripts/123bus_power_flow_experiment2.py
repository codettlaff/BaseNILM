import os
import numpy as np
import random
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from power_flow_with_dss_refactored2 import RadialNetwork

EXPERIMENT_NAME = "ieee_123_single_phase_power_flow"
EPSILON_VALUES = np.linspace(50, 1000, 10)
EPSILON = 1000

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

def make_private_load_profile(load_profile, num_houses, B, epsilon):

    b = 2 * B / epsilon
    noise = np.random.laplace(0, b, size=(num_houses, len(load_profile)))
    noisy_load_profile = load_profile + noise.sum(axis=0)
    return noisy_load_profile

# Read Original IEEE123 Bus Files
paths = get_paths()
ieee123_bus_original_filepath = os.path.join(paths["ieee_123bus"], 'Master.dss')
ieee123_bus_modified_filepath = os.path.join(paths["scripts"], f'{NETWORK_NAME}.dss')
ieee123_bus_modified_private_filepath = os.path.join(paths["scripts"], f'{NETWORK_NAME}_private.dss')

network = RadialNetwork(name=NETWORK_NAME, dss_filepath=ieee123_bus_original_filepath)

houses = load_redd_houses()

P_loads_original = network.P
Q_loads_original = network.Q
P_loads_modified = {}
Q_loads_modified = {}
P_tilde = {}
Q_tilde = {}
mean_norm_errors = []
num_houses_list = []
for node, profile in P_loads_original.items():

    desired_load = np.max(profile)
    Q_max = np.max(Q_loads_original[node])
    alpha = 0.0 if np.isclose(desired_load, 0.0) else Q_max / desired_load

    num_houses, load_profile = assign_houses_to_load(houses, desired_load)
    num_houses_list.append(num_houses)
    P_loads_modified[node] = load_profile
    Q_loads_modified[node] = load_profile * alpha
    P_tilde[node] = make_private_load_profile(load_profile, num_houses, B, EPSILON)
    Q_tilde[node] = P_tilde[node] * alpha

network.P = P_loads_modified
network.Q = Q_loads_modified
network.P_tilde = P_tilde
network.Q_tilde = Q_tilde
network.dss_filepath = ieee123_bus_modified_filepath
network.export_to_dss(tilde=False)
network.dss_filepath = ieee123_bus_modified_private_filepath
network.export_to_dss(tilde=True)

network.lin_dist_flow(tilde=False)
network.lin_dist_flow(tilde=True)

# --- Get Results --- #
p_acc_emp_ldf, V_acc_emp_ldf, p_acc_line_emp_ldf, V_acc_node_emp_ldf = network.empirical_accuracy()
p_acc_line_th_ldf, v_acc_node_th_ldf, p_acc_lb_ldf, v_acc_lb_ldf = network.theoretical_accuracy(B, EPSILON)

network.solve_dss(tilde=False)
network.solve_dss(tilde=True)

p_acc_emp_dss, V_acc_emp_dss, p_acc_line_emp_dss, V_acc_node_emp_dss = network.empirical_accuracy()
p_acc_line_th_dss, v_acc_node_th_dss, p_acc_lb_dss, v_acc_lb_dss = network.theoretical_accuracy(B, EPSILON)

# =========================
# Voltage accuracy (node)
# =========================
dist_emp = []
v_emp = []

for j, v in V_acc_node_emp.items():
    dist_emp.append(network.distance_to_root(j))
    v_emp.append(v)

dist_th = []
v_th = []

for j, v in v_acc_node_th.items():
    dist_th.append(network.distance_to_root(j))
    v_th.append(v)

x_emp = np.array(dist_emp)
y_emp = np.array(v_emp)

x_th = np.array(dist_th)
y_th = np.array(v_th)

# Fit lines
m_emp, b_emp = np.polyfit(x_emp, y_emp, 1)
m_th, b_th = np.polyfit(x_th, y_th, 1)

idx_emp = np.argsort(x_emp)
idx_th = np.argsort(x_th)

plt.figure()
plt.scatter(x_emp, y_emp, label="Empirical")
plt.scatter(x_th, y_th, label="Theoretical")

plt.plot(x_emp[idx_emp], (m_emp * x_emp + b_emp)[idx_emp])
plt.plot(x_th[idx_th], (m_th * x_th + b_th)[idx_th])

plt.xlabel("Distance from root")
plt.ylabel("Voltage accuracy (v_acc)")
plt.title("Voltage Accuracy vs Distance (Empirical vs Theoretical)")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# Power accuracy (line)
# =========================
dist_emp = []
p_emp = []

for (i, j), p in p_acc_line_emp.items():
    if p != 0:
        dist_emp.append(network.distance_to_root(j))
        p_emp.append(p)

dist_th = []
p_th = []

for (i, j), p in p_acc_line_th.items():
    if p != 0:
        dist_th.append(network.distance_to_root(j))
        p_th.append(p)

x_emp = np.array(dist_emp)
y_emp = np.array(p_emp)

x_th = np.array(dist_th)
y_th = np.array(p_th)

# Fit lines
m_emp, b_emp = np.polyfit(x_emp, y_emp, 1)
m_th, b_th = np.polyfit(x_th, y_th, 1)

idx_emp = np.argsort(x_emp)
idx_th = np.argsort(x_th)

plt.figure()
plt.scatter(x_emp, y_emp, label="Empirical")
plt.scatter(x_th, y_th, label="Theoretical")

plt.plot(x_emp[idx_emp], (m_emp * x_emp + b_emp)[idx_emp])
plt.plot(x_th[idx_th], (m_th * x_th + b_th)[idx_th])

plt.xlabel("Distance from root")
plt.ylabel("Power flow accuracy (p_acc)")
plt.title("Power Flow Accuracy vs Distance (Empirical vs Theoretical)")
plt.legend()
plt.grid(True)
plt.show()

print('')
