import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.loadData import load_data, process_data

# -------------------------------
# Experiment Parameters
# -------------------------------
EPSILON_VALUES = [50, 75, 80, 90, 95, 100, 150, 200, 250, 500, 750, 1000]
EXPERIMENT_NAME = "power_flow_data_optimality_3"

# -------------------------------
# Path Utilities
# -------------------------------
def get_paths():
    base = os.path.join(os.path.dirname(__file__), '..')

    return {
        "base": base,
        "data": os.path.join(base, "data"),
        "results": os.path.join(base, "results"),
        "experiment": os.path.join(base, "results", EXPERIMENT_NAME),
        "redd": os.path.join(base, "data", "redd")
    }

def get_redd_files(redd_path):
    return [
        os.path.join(redd_path, f)
        for f in os.listdir(redd_path)
        if f.endswith(".mat") and "HF" not in f
    ]

# -------------------------------
# Radial Network Model
# -------------------------------
class RadialNetwork:
    def __init__(self, nodes, edges, root=0):
        self.nodes = nodes
        self.root = root

        self.children = {i: [] for i in nodes}
        self.parent = {}
        self.r = {}
        self.x = {}

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

    def get_subtree_nodes(self, i):
        result = []
        stack = [i]
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self.children.get(node, []))
        return result

    def compute_branch_flows(self, p):
        P = {}
        order = self.topological_sort()[::-1]

        subtree_sum = {i: p.get(i, 0.0) for i in self.nodes}

        for j in order:
            if j != self.root:
                i = self.parent[j]
                subtree_sum[i] += subtree_sum[j]
                P[(i, j)] = subtree_sum[j]

        return P

    def compute_voltages(self, p, V0=1.0):
        P = self.compute_branch_flows(p)
        V = {self.root: V0}

        for j in self.topological_sort():
            if j == self.root:
                continue

            i = self.parent[j]
            beta = 2 * (self.r[(i, j)] + self.x[(i, j)])
            V[j] = V[i] - beta * P[(i, j)]

        return V, P

    def topological_sort(self):
        order = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            order.append(node)
            queue.extend(self.children.get(node, []))

        return order

# -------------------------------
# Power Flow Runner
# -------------------------------
def run_power_flow(net, p_apps):
    T, N = p_apps.shape

    node_order = net.nodes
    edge_order = list(net.r.keys())

    V_ts, P_ts = [], []

    for t in range(T):
        p = {0: 0.0}
        for i in range(N):
            p[i + 1] = p_apps[t, i]

        V, P = net.compute_voltages(p)

        V_ts.append([V[n] for n in node_order])
        P_ts.append([P[e] for e in edge_order])

    return np.array(V_ts), np.array(P_ts)

# -------------------------------
# Differential Privacy (Per timestep)
# -------------------------------
def differential_privacy_per_node(p_apps, epsilon):
    T, N = p_apps.shape
    noisy = np.zeros_like(p_apps)

    for t in range(T):
        B_t = np.max(p_apps[t, :])
        scale = (2 * B_t) / epsilon
        noise = np.random.laplace(0, scale, size=N)
        noisy[t, :] = p_apps[t, :] + noise

    return noisy

# -------------------------------
# Main Experiment
# -------------------------------
def run_experiment():

    paths = get_paths()
    os.makedirs(paths["experiment"], exist_ok=True)

    redd_files = get_redd_files(paths["redd"])

    for redd_file in redd_files:

        print(f"Processing: {redd_file}")

        data = process_data(load_data(redd_file), "redd")

        p_apps = data['X']
        T, N = p_apps.shape

        # ---------------------------
        # Network
        # ---------------------------
        nodes = list(range(N + 1))
        edges = [(i, i + 1, 0.01, 0.02) for i in range(N)]
        net = RadialNetwork(nodes, edges, root=0)

        # ---------------------------
        # True Power Flow
        # ---------------------------
        V_true, P_true = run_power_flow(net, p_apps)

        beta = {
            e: 2 * (net.r[e] + net.x[e])
            for e in net.r.keys()
        }

        results = []

        for epsilon in EPSILON_VALUES:

            # ---------------------------
            # Apply DP
            # ---------------------------
            p_private = differential_privacy_per_node(p_apps, epsilon)

            V_noisy, P_noisy = run_power_flow(net, p_private)

            # ---------------------------
            # EMPIRICAL TOTAL RMSE
            # ---------------------------
            v_error = np.sqrt(np.sum((V_true - V_noisy) ** 2))
            p_error = np.sqrt(np.sum((P_true - P_noisy) ** 2))

            # ---------------------------
            # THEORY (EXACT FROM PAPER)
            # ---------------------------
            v_var_total = 0
            p_var_total = 0

            for t in range(T):

                # ---------------------------
                # Node-level B_k^2
                # ---------------------------
                B_sq = np.zeros(len(net.nodes))
                for k in net.nodes:
                    if k != 0:
                        B_sq[k] = p_apps[t, k - 1] ** 2

                # ---------------------------
                # Compute subtree sums (bottom-up)
                # ---------------------------
                subtree_sum = B_sq.copy()

                for node in net.topological_sort()[::-1]:
                    if node != net.root:
                        parent = net.parent[node]
                        subtree_sum[parent] += subtree_sum[node]

                # ---------------------------
                # Power variance
                # ---------------------------
                for (i, j) in net.r.keys():
                    p_var_total += (8 / epsilon ** 2) * subtree_sum[j]

                # ---------------------------
                # Voltage variance
                # ---------------------------
                for node in net.nodes:
                    if node == net.root:
                        continue

                    current = node
                    while current != net.root:
                        parent = net.parent[current]
                        e = (parent, current)

                        v_var_total += (8 / epsilon ** 2) * (
                                beta[e] ** 2 * subtree_sum[current]
                        )

                        current = parent

            p_theory = np.sqrt(p_var_total)
            v_theory = np.sqrt(v_var_total)

            results.append({
                "epsilon": epsilon,
                "voltage_rmse": v_error,
                "powerflow_rmse": p_error,
                "voltage_theory": v_theory,
                "powerflow_theory": p_theory
            })

        df = pd.DataFrame(results)

        # ---------------------------
        # Save Results
        # ---------------------------
        name = os.path.basename(redd_file).replace(".mat", "")

        csv_path = os.path.join(paths["experiment"], f"{name}_pf_results.csv")
        df.to_csv(csv_path, index=False)

        # ---------------------------
        # Plot
        # ---------------------------
        plt.figure()

        plt.plot(df["epsilon"], df["voltage_rmse"], marker='o', label="Voltage RMSE (empirical)")
        plt.plot(df["epsilon"], df["powerflow_rmse"], marker='s', label="Branch Flow RMSE (empirical)")

        plt.plot(df["epsilon"], df["voltage_theory"], linestyle='--', label="Voltage RMSE (theory)")
        plt.plot(df["epsilon"], df["powerflow_theory"], linestyle='--', label="Branch Flow RMSE (theory)")

        plt.xscale("log")
        plt.xlabel("Epsilon")
        plt.ylabel("Total RMSE")
        plt.title(f"Power Flow Error vs Privacy ({name})")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(paths["experiment"], f"{name}_pf_plot.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved: {name}")

# -------------------------------
if __name__ == "__main__":
    run_experiment()