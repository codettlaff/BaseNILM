import numpy as np
import pandas as pd

class RadialNetwork:
    def __init__(self, name, nodes, edges, root=0, V0=1.0, alpha=0.0, epsilon=None):
        """
        nodes : dict {i: {"P": [P_i(t)], "B": [B_i(t)]}}
        edges : list of (i, j, r_ij, x_ij)
        root  : root node
        V0    : root node voltage
        alpha : constant power factor parameter (Q_i = alpha P_i)
        """

        self.name = name

        # -----------------------------
        # Time Series
        # -----------------------------
        self.nodes = list(nodes.keys())  # Node indices
        self.P = {i: list(data["P"]) for i, data in nodes.items()} # Nodal active power injections
        self.B = {i: list(data["B"]) for i, data in nodes.items()} # Appliance power bound
        self.T = len(next(iter(self.P.values()))) # Number of timesteps

        # -----------------------------
        # Constants
        # -----------------------------
        self.root = root
        self.V0 = V0
        self.alpha = alpha
        self.epsilon = epsilon
        self.do_differential_privacy = epsilon is not None

        # -----------------------------
        # Tree structure
        # -----------------------------
        self.children = {i: [] for i in self.nodes}
        self.parent = {}
        self.lines = []

        # -----------------------------
        # Line parameters
        # -----------------------------
        self.r = {}
        self.x = {}
        self.z = {}
        self.beta = {}
        self.c = {}

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))

            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

            z_ij = r_ij + x_ij
            beta_ij = r_ij + self.alpha * x_ij
            c_ij = beta_ij / z_ij

            self.z[(i, j)] = z_ij
            self.beta[(i, j)] = beta_ij
            self.c[(i, j)] = c_ij

        # -----------------------------
        # Time-series results (initialized empty dicts)
        # -----------------------------
        self.p = {}  # {(i,j,t): P_ij(t)}
        self.i = {} # {(i,j,t): i_ij(t)}
        self.V = {}  # {(i,t): V_i(t)}
        self.v = {}  # {(i,j,t): v_ij(t)}

        # -----------------------------
        # Privacy
        # -----------------------------
        self.eta = {}  # {(i,t): noise}
        self.P_tilde = {}  # {(i,t): noisy nodal power}

        self.p_tilde = {}  # {(i,j,t): noisy branch flow}
        self.i_tilde = {}  # {(i,j,t): i_ij(t)}
        self.V_tilde = {}  # {(i,t): noisy voltage}
        self.v_tilde = {}  # {(i,j,t): noisy voltage drop}

        # -----------------------------
        # Theoretical Results
        # -----------------------------
        self.sigma_p_th = {} # {(i,j): branch power flow variance}
        self.sigma_i_th = {} # {(i,j): branch current flow variance}
        self.sigma_V_th = {} # {i: node voltage variance}

        self.acc_p_th_bound = 0
        self.acc_i_th_bound = 0
        self.acc_V_th_bound = 0
        self.acc_p_th_exp = 0
        self.acc_i_th_exp = 0
        self.acc_V_th_exp = 0

        # -----------------------------
        # Errors
        # -----------------------------
        self.e_i = {}  # {(i,j,t): current error}
        self.e_p = {}  # {(i,j,t): power error}
        self.e_V = {}  # {(i,t): voltage error}

        self.acc_p = 0
        self.acc_i = 0
        self.acc_v = 0

    # ------------------------------------------------------------------
    # C(i):Set of all nodes along path from root to node.
    # ------------------------------------------------------------------
    def C(self, i):
        """Return list of nodes on the path from root to node i (inclusive)."""
        path = []
        current = i

        # Walk up to the root using parent pointers
        while True:
            path.append(current)
            if current == self.root:
                break
            current = self.parent[current]

        # Reverse so it's root → i (not i → root)
        path.reverse()

        return path

    # ------------------------------------------------------------------
    # D(i): Set of all nodes downstream of node i.
    # ------------------------------------------------------------------
    def D(self, i):
        """Return list of all nodes in the subtree rooted at node i (including i)."""
        stack = [i]
        downstream = []

        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))

        return downstream

    # ------------------------------------------------------------------
    # L(i): Set of all lines along the path from root to node.
    # ------------------------------------------------------------------
    def L(self, i):
        """Return list of edges (i,j) along the path from root to node i."""
        path = self.C(i)
        return [(path[k], path[k + 1]) for k in range(len(path) - 1)]

    # ------------------------------------------------------------------
    # Differential Privacy
    # ------------------------------------------------------------------
    def differential_privacy(self):
        """
        Add Laplace noise to nodal power injections for all timesteps.
        """
        for t in range(self.T):
            for i in self.nodes:
                var = 8 * self.B[i][t] / (self.epsilon ** 2)
                b = np.sqrt(var / 2)

                noise = np.random.laplace(0, b)

                self.eta[(i, t)] = noise
                self.P_tilde[(i, t)] = self.P[i][t] + noise

    # ------------------------------------------------------------------
    # Power Flow
    # ------------------------------------------------------------------
    def power_flow(self):
        """
        Compute branch flows, voltage drops, and node voltages for all timesteps.
        """
        for t in range(self.T):

            # -------------------------
            # Line quantities
            # -------------------------
            for (i, j) in self.lines:
                # Branch power flow
                p_ij = sum(self.P[h][t] for h in self.D(j))
                self.p[(i, j, t)] = p_ij

                # Voltage drop
                v_ij = self.beta[(i, j)] * p_ij
                self.v[(i, j, t)] = v_ij

                # Branch current flow
                i_ij = v_ij / self.z[(i,j)]
                self.i[(i,j,t)] = i_ij

            # -------------------------
            # Node voltages
            # -------------------------
            for i in self.nodes:
                drops = sum(self.v[(k, j, t)] for (k, j) in self.L(i))
                self.V[(i, t)] = self.V0 - drops

    def noisy_power_flow(self):
        """
        Compute power flow using noisy injections P_tilde.
        """
        for t in range(self.T):

            # -------------------------
            # Line quantities
            # -------------------------
            for (i, j) in self.lines:
                # Noisy branch power flow
                p_ij = sum(self.P_tilde[(h, t)] for h in self.D(j))
                self.p_tilde[(i, j, t)] = p_ij

                # Voltage drop
                v_ij = self.beta[(i, j)] * p_ij
                self.v_tilde[(i, j, t)] = v_ij

                # Branch current flow
                i_ij = v_ij / self.z[(i, j)]
                self.i_tilde[(i, j, t)] = i_ij

            # -------------------------
            # Node voltages
            # -------------------------
            for i in self.nodes:
                drops = sum(self.v_tilde[(k, j, t)] for (k, j) in self.L(i))
                self.V_tilde[(i, t)] = self.V0 - drops

    # ------------------------------------------------------------------
    # Theoretical Accuracy
    # ------------------------------------------------------------------
    def compute_theoretical_accuracy(self):

        acc_p_bound_num = 0
        acc_p_bound_den = 0
        acc_i_bound_num = 0
        acc_i_bound_den = 0

        for t in range(self.T):
            for (i,j) in self.lines:

                sigma_p_sq = sum(8 * (self.B[h][t] ** 2) / (self.epsilon ** 2) for h in self.D(j))
                sigma_i_sq = self.c[(i,j)] * sigma_p_sq

                sigma_p = np.sqrt(sigma_p_sq)
                sigma_i = np.sqrt(sigma_i_sq)

                acc_p_bound_num += sigma_p
                acc_i_bound_num += sigma_i

                acc_p_bound_den += self.p[(i,j,t)]
                acc_i_bound_den += self.i[(i,j,t)]

        acc_p_bound_den = acc_p_bound_den * 2
        acc_i_bound_den = acc_i_bound_den * 2

        acc_p_bound = acc_p_bound_num / acc_p_bound_den
        acc_i_bound = acc_i_bound_num / acc_i_bound_den

        acc_p_exp = acc_p_bound * np.sqrt(2/np.pi)
        acc_i_exp = acc_i_bound * np.sqrt(2/np.pi)

        acc_V_bound_num = 0
        acc_V_bound_den = 0

        for t in range(self.T):
            for i in self.nodes:
                self.sigma_V[(i,t)] = (4 / self.epsilon) * np.sqrt(
                    sum(
                        self.beta[(k, j)] * (self.B[h][t] ** 2)
                        for (k, j) in self.L(i)
                        for h in self.D(j)
                    )
                )

                acc_V_bound_num += self.sigma_V
                acc_V_bound_den += self.V[(i,t)]

        acc_V_bound_den = acc_V_bound_den * 2

        acc_V_bound = acc_V_bound_num / acc_V_bound_den
        acc_V_exp = acc_V_bound * np.sqrt(2/np.pi)

        self.acc_p_th_bound = acc_p_bound
        self.acc_i_th_bound = acc_i_bound
        self.acc_V_th_bound = acc_V_bound
        self.acc_p_th_exp = acc_p_exp
        self.acc_i_th_exp = acc_i_exp
        self.acc_V_th_exp = acc_V_exp


    # ------------------------------------------------------------------
    # Empirical Accuracy
    # ------------------------------------------------------------------
    def compute_empirical_accuracy(self):

        p_num = 0
        p_den = 0
        i_num = 0
        i_den = 0

        for t in range(self.T):
            for (i,j) in self.lines:

                self.e_p[(i,j,t)] = np.abs(self.p_tilde[(i,j,t)] - self.p[(i,j,t)])
                self.e_i[(i,j,t)] = np.abs(self.i_tilde[(i,j,t)] - self.i[(i,j,t)])

                p_num += np.sqrt(self.e_p[(i,j,t)])
                p_den += self.p[(i,j,t)]

                i_num += np.sqrt(self.e_i[(i,j,t)])
                i_den += self.i[(i,j,t)]

        p_den = 2 * p_den
        i_den = 2 * i_den

        self.acc_p = 1 - p_num / p_den
        self.acc_i = 1 - i_num / i_den

        V_num = 0
        V_dem = 0

        for t in range(self.T):
            for i in self.nodes:

                self.e_V[(i,t)] = np.abs(self.V_tilde[(i, t)] - self.V[(i, t)])

                V_num += np.sqrt(self.e_V[(i,t)])
                V_dem += self.V[(i, t)]

        V_dem = 2 * V_dem
        self.acc_v = 1 - V_num / V_dem

    # ------------------------------------------------------------------
    # Display Power Flow Results
    # ------------------------------------------------------------------
    def power_flow_results(self, t=0, return_results=False, display_results=False, write_csv=False, results_folderpath=None):

        # ============================================================
        # NODE TABLE
        # ============================================================
        node_data = []
        for i in self.nodes:
            node_data.append({
                "node": i,
                "V": self.V.get((i, t), None),
                "P_injection": self.P[i][t]
            })

        df_nodes = pd.DataFrame(node_data).sort_values(by="node")

        # ============================================================
        # LINE TABLE
        # ============================================================
        line_data = []
        for (i, j) in self.lines:
            line_data.append({
                "from": i,
                "to": j,
                "r": self.r[(i, j)],
                "x": self.x[(i, j)],
                "p_flow": self.p.get((i, j, t), None),
                "i_flow": self.i.get((i, j, t), None),
                "v_drop": self.v.get((i, j, t), None),
            })

        df_lines = pd.DataFrame(line_data).sort_values(by=["from", "to"])

        # ============================================================
        # DISPLAY
        # ============================================================
        if display_results:
            print("\n=== NODE STATES (t={}) ===".format(t))
            print(df_nodes.to_string(index=False))

            print("\n=== LINE STATES (t={}) ===".format(t))
            print(df_lines.to_string(index=False))

        # ============================================================
        # SAVE TO CSV
        # ============================================================

        if write_csv:
            nodes_csv_filepath = results_folderpath + f"{self.name}_nodes_t{t}.csv"
            lines_csv_filepath = results_folderpath + f"{self.name}_lines_t{t}.csv"
            df_nodes.to_csv(nodes_csv_filepath)
            df_lines.to_csv(lines_csv_filepath)

        if return_results: return df_nodes, df_lines