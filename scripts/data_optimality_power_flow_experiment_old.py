import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data
from differential_privacy import differential_privacy
from power_flow import RadialNetwork


# ============================================================
# PARAMETERS
# ============================================================
EXPERIMENT_NAME = "data_optimality_power_flow"
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
EPSILON_VALUES = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# EPSILON_VALUES = [75, 100, 500, 1000] # For Testing
N_NODES = 6
V0 = 12.47e3
ROOT = 0
T_set = 10 # Limit Timesteps

# -----------------------------
# Path Utilities (reuse style)
# -----------------------------
def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')

    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment_results": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd")
    }

def get_redd_files(redd_path):
    return [
        os.path.join(redd_path, f)
        for f in os.listdir(redd_path)
        if f.endswith(".mat") and "HF" not in f
    ]

# ============================================================
# NETWORK CONSTRUCTION
# ============================================================
def build_radial_network(n_nodes, V0):
    nodes = list(range(n_nodes))
    edges = []

    for i in range(n_nodes - 1):
        r = 0.01
        x = 0.01
        edges.append((i, i + 1, r, x))

    return RadialNetwork(nodes, edges, root=ROOT, V0=V0)


# ============================================================
# SENSITIVITY: B_t = max appliance power at time t
# ============================================================
def compute_B_t(p_apps):
    return np.max(p_apps, axis=1)  # shape (T,)


# ============================================================
# APPLY DP:  p̃_i(t) = p_i(t) + η_i(t)
# ============================================================
def apply_dp_per_node(p_nodes, p_apps, epsilon):
    p_nodes_tilde = {}

    for i, signal in p_nodes.items():
        p_nodes_tilde[i] = differential_privacy(
            signal,
            p_apps,
            epsilon
        )

    return p_nodes_tilde


# ============================================================
# BUILD p_i(t) DICTIONARY
# ============================================================
def build_p_dict(p_nodes, t):
    return {i: p_nodes[i][t] for i in p_nodes}

# ============================================================
# ACCURACY METRICS (EMPIRICAL, TIME-AGGREGATED)
# ============================================================
def compute_accuracy_metrics(V_true_time, V_noisy_time, I_true_time, I_noisy_time):
    """
    Computes variance-based accuracy metrics over all time steps:

        Acc_V^(var) = 1 - (sum_{t,i} e_{v,i,t}^2) / (2 sum_{t,i} V_{i,t}^2)
        Acc_I^(var) = 1 - (sum_{t,ℓ} e_{ℓ,t}^2) / (2 sum_{t,ℓ} I_{ℓ,t}^2)

    Parameters
    ----------
    V_true_time, V_noisy_time : list of dicts
        Each element is V_{·,t}, indexed by node i
    I_true_time, I_noisy_time : list of dicts
        Each element is I_{·,t}, indexed by edge ℓ

    Returns
    -------
    acc_v, acc_l : floats
        Voltage and current accuracy
    """

    num_v = 0.0
    den_v = 0.0
    num_l = 0.0
    den_l = 0.0

    T = len(V_true_time)

    for t in range(T):

        V_true = V_true_time[t]
        V_noisy = V_noisy_time[t]

        I_true = I_true_time[t]
        I_noisy = I_noisy_time[t]

        # --- Voltage ---
        for i in V_true:
            e_v = V_noisy[i] - V_true[i]
            num_v += e_v**2
            den_v += V_true[i]**2

        # --- Line current ---
        for edge in I_true:
            e_l = I_noisy[edge] - I_true[edge]
            num_l += e_l**2
            den_l += I_true[edge]**2

    acc_v = 1 - num_v / (2 * den_v)
    acc_l = 1 - num_l / (2 * den_l)

    return acc_v, acc_l


# ============================================================
# THEORETICAL ACCURACY (FROM PAPER)
# ============================================================
def compute_a_jh_t(network, j, h, t, V, P):

    if h in network.D(j):
        term1 = 1 / V[j,t]
    else: term1 = 0

    term2 = 0
    for k in network.C(j):
        if h in network.D(k):
            term2 = term2 + network.beta(j,k)

def compute_a_jh_t(j, h, t, V_time, P_time, D, C, beta):
    """
    Compute a_{jh,t} =
        (1 / V_{j,t}) * 1_{h in D(j)}
        + (P_{ij,t} / V_{j,t}^2) * sum_{k in C(j)} beta_{jk} * 1_{h in D(k)}

    Notes:
    - This version assumes P_{ij,t} corresponds to the line ending at node j.
      (i.e., parent -> j edge is used implicitly)
    """

    V_jt = V_time[t][j]

    # --- first term ---
    term1 = (1 / V_jt) if h in D(j) else 0.0

    # --- second term ---
    beta_sum = 0.0
    for k in C: # Bug C(j) is being treated as immediate children only, not all children.
        if h in D(k):
            beta_sum += beta(j, k)

    # You need P_{ij,t}; assume parent edge is provided via P_time[(i,j)]
    # If multiple parents are possible, this must be adjusted.
    P_ij_t = None
    for (i_candidate, j_candidate), val in P_time[t].items():
        if j_candidate == j:
            P_ij_t = val
            break

    if P_ij_t is None:
        raise ValueError(f"No incoming edge found for node {j}")

    term2 = (P_ij_t / (V_jt**2)) * beta_sum

    return term1 + term2

def compute_i_acc_theory(
    I_time, V_time, P_time,
    network, B, epsilon, edges
):
    """
    Theoretical current accuracy using expected absolute error:

    Acc_I = 1 - (4 / ε) *
        [ sum_{t} sum_{l} sqrt( sum_h B_h^2 a_{jh,t}^2 ) ]
        / [ sum_{t} sum_{l} i_{ij,t} ]

    Parameters
    ----------
    I_time : list of arrays
        I_time[t][ell] = current on edge ell at time t
    V_time : list/array
        V_time[t][j] = voltage at node j at time t
    P_time : list of dicts
        P_time[t][(i,j)] = power flow on edge (i,j)
    D, C : dict
        Downstream and path sets
    beta : function
        beta(j,k)
    B : array
        Appliance bounds B_h
    epsilon : float
    edges : list of (i,j)

    Returns
    -------
    acc_i : float
    """

    num = 0.0
    denom = 0.0

    T = len(I_time)

    for t in tqdm(range(T), desc="Computing Abs Theoretical Current Accuracy"):

        I_t = I_time[t]

        for ell, (i, j) in enumerate(edges):

            # --- denominator ---
            denom += abs(I_t[(i, j)])

            # --- compute inner sum ---
            inner_sum = 0.0

            for h in N:
                a_jh_t = compute_a_jh_t(j, h, t, V_time, P_time, network.D, C, beta)
                inner_sum += (B[h]**2) * (a_jh_t**2)

            num += np.sqrt(inner_sum)

    acc_i = 1 - (4 / epsilon) * (num / denom)

    return acc_i

def compute_v_acc_theory(
    V_time, P_time,
   network, B, epsilon, edges
):
    """
    Theoretical voltage accuracy using expected absolute error:

    Acc_V = 1 -
        [ sum_{t} sum_{j}
            sqrt( (8 / ε^2) * sum_h B_h^2 ( sum_{ij in C(j)} β_{ij} a_{jh,t} )^2 )
        ]
        / [ 2 sum_{t} sum_{j} V_{j,t} ]

    Parameters
    ----------
    V_time : array-like (T, N)
    P_time : list of dicts
        P_time[t][(i,j)] = power flow
    D, C : dict
    beta : function
    B : array
    epsilon : float
    edges : list of (i,j)

    Returns
    -------
    acc_v : float
    """

    num = 0.0
    denom = 0.0

    T = len(V_time)

    for t in tqdm(range(T), desc="Computing Abs Theoretical Voltage Accuracy"):

        for j in N:

            # --- denominator ---
            denom += abs(V_time[t][j])

            # --- compute inner sum over h ---
            inner_sum = 0.0

            for h in N:

                # compute sum_{ij in C(j)} β_{ij} a_{jh,t}
                beta_a_sum = 0.0

                for (i, j_edge) in edges:
                    if j_edge == j:  # edges in C(j)
                        a_jh_t = compute_a_jh_t(j, h, t, V_time, P_time, D, C, beta)
                        beta_a_sum += beta(i, j) * a_jh_t

                inner_sum += (B[h]**2) * (beta_a_sum**2)

            # sqrt(8/ε^2 * inner_sum) = (sqrt(8)/ε) * sqrt(inner_sum)
            num += (np.sqrt(8) / epsilon) * np.sqrt(inner_sum)

    acc_v = 1 - num / (2 * denom)

    return acc_v

def trim_data(data, T_new):
    """
    Trim dataset to first T_new time steps for faster debugging.

    Parameters
    ----------
    data : dict
        Expected keys:
            'Y' : aggregate signal (T,)
            'X' : appliance-level signals (T, n_appliances)
    T_new : int
        Number of time steps to keep

    Returns
    -------
    data_trimmed : dict
        Same structure as input, but truncated
    """

    data_trimmed = {}

    # --- Trim aggregate ---
    if 'Y' in data:
        data_trimmed['Y'] = data['Y'][:T_new]

    # --- Trim appliance-level data ---
    if 'X' in data:
        data_trimmed['X'] = data['X'][:T_new]

    # --- Copy any other fields safely ---
    for key in data:
        if key not in ['Y', 'X']:
            val = data[key]

            # If it's time-series aligned, try trimming
            try:
                if hasattr(val, '__len__') and len(val) >= T_new:
                    data_trimmed[key] = val[:T_new]
                else:
                    data_trimmed[key] = val
            except:
                data_trimmed[key] = val

    return data_trimmed

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment():

    paths = get_paths()
    results_folder = paths["experiment_results"]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    redd_files = get_redd_files(paths["redd"])
    data = process_data(load_data(redd_files[0]), "redd")

    if T_set: data = trim_data(data, T_set)

    p_agg = data['Y']
    p_apps = data['X']

    T = p_agg.shape[0]

    # --- Network ---
    network = build_radial_network(N_NODES, V0)

    # --- Assign loads ---
    p_nodes = {i: p_agg.copy() for i in network.nodes}

    # --- Sensitivity ---
    B_t = compute_B_t(p_apps)

    results = []

    # Precompute edges (parent → child)
    edges = [(network.parent[j], j) for j in network.nodes if j != network.root]

    for epsilon in EPSILON_VALUES:

        print(f"\nRunning epsilon = {epsilon}")

        # Apply DP
        p_nodes_tilde = apply_dp_per_node(p_nodes, p_apps, epsilon)

        # Store trajectories
        V_i_true_time = []
        I_ij_true_time = []
        P_ij_true_time = []

        V_i_noisy_time = []
        I_ij_noisy_time = []
        P_ij_noisy_time = []

        # ---------------------------
        # TIME LOOP
        # ---------------------------
        for t in tqdm(range(T), desc="Computing Empirical Accuracy"):

            p_true = build_p_dict(p_nodes, t)
            p_tilde = build_p_dict(p_nodes_tilde, t)

            V_i_true, P_ij_true, I_ij_true = network.solve_power_flow(p_true)
            V_i_noisy, P_ij_noisy, I_ij_noisy = network.solve_power_flow(p_tilde)

            V_i_true_time.append(V_i_true)
            P_ij_true_time.append(P_ij_true)
            I_ij_true_time.append(I_ij_true)

            V_i_noisy_time.append(V_i_noisy)
            P_ij_noisy_time.append(P_ij_noisy)
            I_ij_noisy_time.append(I_ij_noisy)

        # ---------------------------
        # EMPIRICAL ACCURACY (ABS ERROR)
        # ---------------------------
        num_v = 0.0
        den_v = 0.0
        num_i = 0.0
        den_i = 0.0

        for t in range(T):

            V_true = V_i_true_time[t]
            V_noisy = V_i_noisy_time[t]

            I_true = I_ij_true_time[t]
            I_noisy = I_ij_noisy_time[t]

            # Voltage
            for i in V_true:
                num_v += abs(V_noisy[i] - V_true[i])
                den_v += abs(V_true[i])

            # Current
            for ell, (i, j) in enumerate(edges):
                num_i += abs(I_noisy[(i, j)] - I_true[(i, j)])
                den_i += abs(I_true[(i, j)])

        acc_v_emp = 1 - num_v / (2 * den_v)
        acc_i_emp = 1 - num_i / (2 * den_i)

        # ---------------------------
        # THEORETICAL ACCURACY (ABS ERROR)
        # ---------------------------
        acc_v_th = compute_v_acc_theory(
            V_i_true_time,
            P_ij_true_time,
            network,
            B_t,
            epsilon,
            edges
        )

        acc_i_th = compute_i_acc_theory(
            I_ij_true_time,
            V_i_true_time,
            P_ij_true_time,
            network,
            B_t,
            epsilon,
            edges
        )

        results.append({
            "epsilon": epsilon,
            "Acc_V_emp": acc_v_emp,
            "Acc_V_theory": acc_v_th,
            "Acc_I_emp": acc_i_emp,
            "Acc_I_theory": acc_i_th
        })

    # Save results
    results_filepath = os.path.join(results_folder, "results.csv")
    pd.DataFrame(results).to_csv(results_filepath, index=False)

    return results

def plot_results(show=False, log_scale=False):
    """
    Reads results CSV and plots:

        1) Acc_V vs epsilon
        2) Acc_I vs epsilon

    Parameters
    ----------
    show : bool
        Whether to display plots
    log_scale : bool
        If True, use logarithmic scale on epsilon axis

    Saves plots to experiment_results folder.
    """

    paths = get_paths()
    results_folder = paths["experiment_results"]
    results_filepath = os.path.join(results_folder, 'results.csv')

    # Ensure folder exists
    os.makedirs(results_folder, exist_ok=True)

    # ---------------------------
    # Load results
    # ---------------------------
    df = pd.read_csv(results_filepath)
    epsilon = df["epsilon"]

    # ---------------------------
    # Plot: Voltage Accuracy
    # ---------------------------
    plt.figure()

    plt.plot(epsilon, df["Acc_V_emp"], marker='o', label="Empirical")
    plt.plot(epsilon, df["Acc_V_theory"], linestyle='--', label="Theoretical")

    if log_scale:
        plt.xscale('log')

    plt.xlabel(r"Privacy Budget $\epsilon$")
    plt.ylabel(r"$\mathrm{Acc}_V$")
    plt.title(r"Voltage Accuracy vs $\epsilon$")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save figure
    filename_v = "acc_v_vs_epsilon_log.png" if log_scale else "acc_v_vs_epsilon.png"
    save_path_v = os.path.join(results_folder, filename_v)
    plt.savefig(save_path_v, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    # ---------------------------
    # Plot: Line Current Accuracy
    # ---------------------------
    plt.figure()

    plt.plot(epsilon, df["Acc_I_emp"], marker='o', label="Empirical")
    plt.plot(epsilon, df["Acc_I_theory"], linestyle='--', label="Theoretical")

    if log_scale:
        plt.xscale('log')

    plt.xlabel(r"Privacy Budget $\epsilon$")
    plt.ylabel(r"$\mathrm{Acc}_\ell$")
    plt.title(r"Line Current Accuracy vs $\epsilon$")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save figure
    filename_l = "acc_l_vs_epsilon_log.png" if log_scale else "acc_l_vs_epsilon.png"
    save_path_l = os.path.join(results_folder, filename_l)
    plt.savefig(save_path_l, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":

    run_experiment()
    plot_results(show=True,log_scale=True)