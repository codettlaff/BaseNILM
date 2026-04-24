import numpy as np
import pandas as pd
import opendssdirect as dss

# This Version will also include a method that solves the system using OpenDSS.

class RadialNetwork:
    def __init__(self, name, nodes, edges, root=0, V0=1.0, alpha=0.0, epsilon=None, dss_filepath='network.dss'):
        """
        nodes : dict {i: {"P": [P_i(t)], "B": [B_i(t)]}}
        edges : list of (i, j, r_ij, x_ij)
        root  : root node
        V0    : root node voltage
        alpha : constant power factor parameter (Q_i = alpha P_i)
        """

        self.name = name
        self.dss_filepath = dss_filepath

        # -----------------------------
        # Time Series
        # -----------------------------
        self.nodes = list(nodes.keys())  # Node indices
        self.P = {i: list(data["P"]) for i, data in nodes.items()} # Nodal active power injections
        self.P_tilde = {i: list(data["P"]) for i, data in nodes.items()}  # Nodal active power injections
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

        # Expected Error
        self.e_i_exp = {}  # {(i,j,t): current error}
        self.e_p_exp = {}  # {(i,j,t): power error}
        self.e_V_exp = {}  # {(i,t): voltage error}

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

        # Normalized Error
        self.e_i_norm = {}  # {(i,j,t): current error}
        self.e_p_norm = {}  # {(i,j,t): power error}
        self.e_V_norm = {}  # {(i,t): voltage error}

        self.acc_p = 0
        self.acc_i = 0
        self.acc_v = 0

    # ------------------------------------------------------------------
    # Set Uniform B
    # ------------------------------------------------------------------
    def set_uniform_B(self, value):
        for i in self.nodes:
            self.B[i] = [value] * self.T

    # ------------------------------------------------------------------
    # Build DSS Network From Nodes and Edges
    # ------------------------------------------------------------------

    def export_to_dss(self, t=0, tilde=False):

        with open(self.dss_filepath, 'w') as f:

            # Circuit Definition - Automatically creates default source Vsource.Source
            f.write(f"Clear\n")
            f.write(f"New Circuit.{self.name} basekv={self.V0 / 1000} pu=1.0\n\n") # Root Node 1 pu
            f.write("Edit Vsource.Source bus1=bus0\n\n") # Root Node index 0

            # Lines
            for (i,j) in self.lines:
                r = self.r[(i,j)]
                x = self.x[(i,j)]
                f.write(
                    f"New Line.L_{i}_{j} "
                    f"bus1=bus{i} bus2=bus{j} "
                    f"r1={r} x1={x} r0={r} x0={x} "
                    f"length=1 units=km\n" # Ohms per unit length
                )
            f.write("\n")

            # Loads
            for i in self.nodes:
                if i == self.root:
                    continue

                if tilde:
                    P = self.P_tilde[(i, j)]
                else: P = self.P[i][t]
                Q = self.alpha * P

                f.write(
                    f"New Load.Load_{i} "
                    f"bus1=bus{i} "
                    f"phases=1 "
                    f"conn=wye "
                    f"model=1 "
                    f"kV=12.47 "
                    f"kW={P / 1000} "
                    f"kvar={Q / 1000}\n"
                )

            f.write("\n")
            f.write("Solve\n")

    # ------------------------------------------------------------------
    # Build Nodes and Edges from DSS Network
    # ------------------------------------------------------------------
    def build_from_dss(self, t=0):

        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

        # Map buses to indices
        bus_names = dss.Circuit.AllBusNames()
        bus_map = {name: idx for idx, name in enumerate(bus_names)}

        # Initialize node structure
        nodes = {
            i: {"P": self.P[i], "B": self.B[i]}
            for i in bus_map.values()
        }

        # Extract Loads to Nodal P
        dss.Loads.First()
        while True:

            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]

            P_kw = dss.Loads.kW()
            nodes[i]["P"][t] = P_kw * 1000.0
            nodes[i]["B"][t] = abs(P_kw * 1000.0) # Conservative B - set B to Total Power (One Appliance)

            if not dss.Loads.Next():
                break

        # Extract Lines to Edges
        edges = []

        dss.Lines.First()
        while True:
            bus1 = dss.Lines.Bus1().split(".")[0]
            bus2 = dss.Lines.Bus2().split(".")[0]

            i = bus_map[bus1]
            j = bus_map[bus2]

            length = dss.Lines.Length()
            r = dss.Lines.R1() * length
            x = dss.Lines.X1() * length

            edges.append((i, j, r, x))

            if not dss.Lines.Next():
                break

        self.nodes = nodes
        self.edges = edges

    # ------------------------------------------------------------------
    # Solve DSS Power Flow - New (Tilde Setting)
    # ------------------------------------------------------------------
    def dss_power_flow(self, tilde=False):

        V_dst = {}
        p_dst = {}
        i_dst = {}
        v_dst = {}

        for t in range(self.T):

            self.export_to_dss(t, tilde)

            dss.Text.Command("Clear")
            dss.Text.Command(f"compile [{self.dss_filepath}]")
            dss.Text.Command("Solve")

            # Bus voltage magnitudes
            bus_names = dss.Circuit.AllBusNames()
            for bus in bus_names:
                dss.Circuit.SetActiveBus(bus)
                vmag = dss.Bus.puVmagAngle()[0]
                i = int(bus.replace("bus", ""))
                V_dst[(i, t)] = vmag

            # Line flows and currents
            dss.Lines.First()
            while True:
                name = dss.Lines.Name()

                # Parse Line Name
                _, i_str, j_str = name.split("_")
                i = int(i_str)
                j = int(j_str)

                # Activate Element
                dss.Circuit.SetActiveElement(f"Line.{name}")

                # Powers: [P1, Q1, P2, Q2] (kW, kvar)
                powers = dss.CktElement.Powers()

                # Currents: [I1_real, I1_imag, I2_real, I2_imag]
                currents = dss.CktElement.Currents()

                # Real power flow (from i to j)
                P_ij = powers[0] * 1000.0  # kW → W
                p_dst[(i, j, t)] = P_ij

                # Current Magnitude (from i to j)
                I_real = currents[0]
                I_imag = currents[1]
                I_mag = np.sqrt(I_real ** 2 + I_imag ** 2)
                i_dst[(i, j, t)] = I_mag

                v_dst[(i, j, t)] = V_dst[(i,t)] - V_dst[(j,t)]
                if dss.Lines.Next() == 0:
                    break

        if tilde:
            self.V_tilde = V_dst
            self.v_tilde = v_dst
            self.i_tilde = i_dst
            self.p_tilde = p_dst
        else:
            self.V = V_dst
            self.v = v_dst
            self.i = i_dst
            self.p = p_dst

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
        Output structure matches self.P: {i: [P_i(t)]}
        """

        # Initialize structure
        self.P_tilde = {i: [0.0] * self.T for i in self.nodes}
        self.eta = {}

        for t in range(self.T):
            for i in self.nodes:
                var = 8 * self.B[i][t] ** 2 / (self.epsilon ** 2)
                b = np.sqrt(var / 2)

                noise = np.random.laplace(0, b)

                self.eta[(i, t)] = noise
                self.P_tilde[i][t] = self.P[i][t] + noise

    def differential_privacy_old(self):
        """
        Add Laplace noise to nodal power injections for all timesteps.
        """
        for t in range(self.T):
            for i in self.nodes:
                var = 8 * self.B[i][t]**2 / (self.epsilon ** 2)
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
            for (i, j) in self.lines:
                # Variances
                sigma_p_sq = sum(8 * (self.B[h][t] ** 2) / (self.epsilon ** 2) for h in self.D(j))
                sigma_i_sq = self.c[(i, j)] * sigma_p_sq

                sigma_p = np.sqrt(sigma_p_sq)
                sigma_i = np.sqrt(sigma_i_sq)

                # Store sigmas
                self.sigma_p_th[(i, j, t)] = sigma_p
                self.sigma_i_th[(i, j, t)] = sigma_i

                # Expected absolute error (Gaussian)
                e_exp_p = np.sqrt(2 / np.pi) * sigma_p
                e_exp_i = np.sqrt(2 / np.pi) * sigma_i

                # Store expected errors
                self.e_p_exp[(i, j, t)] = e_exp_p
                self.e_i_exp[(i, j, t)] = e_exp_i

                # Accumulate bounds
                acc_p_bound_num += sigma_p
                acc_i_bound_num += sigma_i

                acc_p_bound_den += self.p[(i, j, t)]
                acc_i_bound_den += self.i[(i, j, t)]

        acc_p_bound_den *= 2
        acc_i_bound_den *= 2

        acc_p_bound = 1 - acc_p_bound_num / acc_p_bound_den
        acc_i_bound = 1 - acc_i_bound_num / acc_i_bound_den

        acc_p_exp = 1 - np.sqrt(2 / np.pi) * acc_p_bound_num / acc_p_bound_den
        acc_i_exp = 1 - np.sqrt(2 / np.pi) * acc_i_bound_num / acc_i_bound_den

        acc_V_bound_num = 0
        acc_V_bound_den = 0

        for t in range(self.T):
            for i in self.nodes:
                sigma_v = (4 / self.epsilon) * np.sqrt(
                    sum(
                        self.beta[(k, j)] * (self.B[h][t] ** 2)
                        for (k, j) in self.L(i)
                        for h in self.D(j)
                    )
                )

                # Store sigma
                self.sigma_V_th[(i, t)] = sigma_v

                # Expected error
                e_exp_v = np.sqrt(2 / np.pi) * sigma_v

                # Store expected error
                self.e_V_exp[(i, t)] = e_exp_v

                # Accumulate bounds
                acc_V_bound_num += sigma_v
                acc_V_bound_den += self.V[(i, t)]

        acc_V_bound_den *= 2

        acc_V_bound = 1 - acc_V_bound_num / acc_V_bound_den
        acc_V_exp = 1 - np.sqrt(2 / np.pi) * acc_V_bound_num / acc_V_bound_den

        # Normalize Error
        self.e_p_exp = {
            key: self.e_p_exp[key] / (2 * self.p[key]) if self.p[key] != 0 else 0
            for key in self.e_p_exp
        }
        self.e_i_exp = {
            key: self.e_i_exp[key] / (2 * self.i[key]) if self.i[key] != 0 else 0
            for key in self.e_i_exp
        }
        self.e_v_exp = {
            key: self.e_V_exp[key] / (2 * self.V[key]) if self.V[key] != 0 else 0
            for key in self.e_V_exp
        }

        self.acc_p_th_bound = acc_p_bound
        self.acc_i_th_bound = acc_i_bound
        self.acc_V_th_bound = acc_V_bound
        self.acc_p_th_exp = acc_p_exp
        self.acc_i_th_exp = acc_i_exp
        self.acc_V_th_exp = acc_V_exp

    # ------------------------------------------------------------------
    # Empirical Accuracy
    # ------------------------------------------------------------------
    # To change - also get normalized errors
    def compute_empirical_accuracy(self):

        p_num = 0
        p_den = 0
        i_num = 0
        i_den = 0

        for t in range(self.T):
            for (i,j) in self.lines:

                # Absolute errors
                self.e_p[(i,j,t)] = np.abs(self.p_tilde[(i,j,t)] - self.p[(i,j,t)])
                self.e_i[(i,j,t)] = np.abs(self.i_tilde[(i,j,t)] - self.i[(i,j,t)])

                # Normalized errors
                denom_p = 2 * np.abs(self.p[(i, j, t)])
                denom_i = 2 * np.abs(self.i[(i, j, t)])

                self.e_p_norm[(i, j, t)] = self.e_p[(i, j, t)] / denom_p if denom_p != 0 else 0
                self.e_i_norm[(i, j, t)] = self.e_i[(i, j, t)] / denom_i if denom_i != 0 else 0

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

                # Absolute errors
                self.e_V[(i,t)] = np.abs(self.V_tilde[(i, t)] - self.V[(i, t)])

                # Normalized error
                denom_V = 2 * np.abs(self.V[(i, t)])
                self.e_V_norm[(i, t)] = self.e_V[(i, t)] / denom_V if denom_V != 0 else 0

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

    def noisy_power_flow_results(self, t=0, return_results=False, display_results=False, write_csv=False, results_folderpath=None):

        # ============================================================
        # NODE TABLE
        # ============================================================
        node_data = []
        for i in self.nodes:
            node_data.append({
                "node": i,
                "V": self.V_tilde.get((i, t), None),
                "P_injection": self.P_tilde[i][t]
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
                "p_flow": self.p_tilde.get((i, j, t), None),
                "i_flow": self.i_tilde.get((i, j, t), None),
                "v_drop": self.v_tilde.get((i, j, t), None),
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
            nodes_csv_filepath = results_folderpath + f"{self.name}_nodes_noisy_t{t}.csv"
            lines_csv_filepath = results_folderpath + f"{self.name}_lines_noisy_t{t}.csv"
            df_nodes.to_csv(nodes_csv_filepath)
            df_lines.to_csv(lines_csv_filepath)

        if return_results: return df_nodes, df_lines