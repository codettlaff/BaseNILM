import numpy as np
import pandas as pd
import opendssdirect as dss

class RadialNetwork:

    def __init__(self, name, dss_filepath='master.dss', B=5e3, epsilon=1000):

        self.name = name
        self.dss_filepath = dss_filepath

        self.B=5e3
        self.epsilon=1000

        self.nodes = []
        self.P = {}
        self.P_tilde = {}
        self.Q = {}
        self.Q_tilde = {}
        self.children = {}
        self.parent = {}

        self.build_from_dss()

        # True Power Flow Results
        self.p = {}  # {(i,j,t): P_ij(t)} Branch power flow
        self.i = {}  # {(i,j,t): i_ij(t)} Branch current flow
        self.V = {}  # {(i,t): V_i(t)} Node Voltage Magnitude
        self.v = {}  # {(i,j,t): v_ij(t)} Squared Node Voltage Magnitude

        # Noisy Power Flow Results
        self.p_tilde = {}  # {(i,j,t): P_ij(t)} Branch power flow
        self.i_tilde = {}  # {(i,j,t): i_ij(t)} Branch current flow
        self.V_tilde = {}  # {(i,t): V_i(t)} Node Voltage Magnitude
        self.v_tilde = {}  # {(i,j,t): v_ij(t)} Squared Node Voltage Magnitude

    def build_from_dss(self):

        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{self.dss_filepath}]")

        # Get Base Voltage of the Source
        dss.Text.Command("CalcVoltageBases")
        dss.Vsources.First()
        bus_full = dss.CktElement.BusNames()[0]
        bus = bus_full.split('.')[0]
        dss.Circuit.SetActiveBus(bus)
        self.V0 = dss.Bus.kVBase() * 1e3

        # Map buses to indices
        bus_names = dss.Circuit.AllBusNames()
        bus_map = {name: idx for idx, name in enumerate(bus_names)}

        # Determine number of timesteps from first Loadshape
        T = 0
        dss.Loads.First()
        if dss.Loads.Count() > 0:
            shape_name = dss.Loads.Daily()
            if shape_name:
                dss.LoadShape.Name(shape_name)
                T = dss.LoadShape.Npts()

        self.T = T

        # Initialize nodes with zero time-series
        nodes = {
            i: {"P": [0.0] * T, "Q": [0.0] * T}
            for i in bus_map.values()
        }

        # Extract Loads - Full time-series
        dss.Loads.First()
        while True:
            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]

            load_peak_kw = dss.Loads.kW()
            load_peak_kvar = dss.Loads.kvar()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = [load_peak_kw * s for s in dss.LoadShape.PMult()]
            kvar = [load_peak_kvar * s for s in dss.LoadShape.PMult()]

            for t in range(T):
                nodes[i]["P"] = [p * 1e3 for p in kw]  # kW to W
                nodes[i]["Q"] = [q * 1e3 for q in kvar]  # kvar to Var

            if not dss.Loads.Next():
                break

        # Extract Lines - Edges
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

        self.nodes = list(nodes.keys())
        self.T = len(nodes[1]["P"])  # Number of Timesteps
        self.P = {i: data["P"] for i, data in nodes.items()}  # Copy True Injections
        self.Q = {i: data["Q"] for i, data in nodes.items()}  # Copy True Injections
        self.P_tilde = {i: [0.0] * self.T for i, data in nodes.items()}  # Initialize Noisy Injections
        self.Q_tilde = {i: [0.0] * self.T for i, data in nodes.items()} # Initialize Noisy Injections

        # Tree Structure
        self.children = {i: [] for i in self.nodes}  # Initialize Dict
        self.parent = {}
        self.lines = []

        # Line Parameters
        self.r = {}  # Unit Ohms
        self.x = {}  # Unit Ohms

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

    def export_to_dss(self, tilde=False):
        with open(self.dss_filepath, 'w') as f:

            # Circuit Definition
            f.write(f"Clear\n")
            f.write(f"New Circuit.{self.name} basekv={self.V0/1e3} pu=1.0\n")
            f.write(f"Edit Vsource.Source bus1=bus0\n") # Make sure root node is index 0.

            # Lines
            for (i, j) in self.lines:
                r = self.r[(i, j)]
                x = self.x[(i, j)]
                f.write(
                    f"New Line.L_{i}_{j} "
                    f"bus1=bus{i} bus2=bus{j} "
                    f"r1={r} x1={x} r0={r} x0={x} "
                    f"length=1 units=km\n"  # Ohms per Unit Length
                )

            f.write("\n")

            load_kw = []
            load_kvar = []

            # Load-Shapes
            for i in self.nodes:
                if i == 0:
                    continue

                if tilde:
                    P_series = self.P_tilde[i]
                    Q_series = self.Q_tilde[i]
                else:
                    P_series = self.P[i]
                    Q_series = self.Q[i]

                max_kw = np.max(P_series) / 1e3
                load_kw.append(max_kw)
                P_series_scaled = P_series / np.max(P_series) if np.max(P_series) != 0 else np.zeros_like(P_series)

                max_kvar = np.max(Q_series) / 1e3
                load_kvar.append(max_kvar)
                Q_series_scaled = Q_series / np.max(Q_series) if np.max(Q_series) != 0 else np.zeros_like(Q_series)

                P_mult_str = " ".join(str(p) for p in P_series_scaled)
                Q_mult_str = " ".join(str(q) for q in Q_series_scaled)

                f.write(
                    f"New LoadShape.LS_{i} "
                    f"npts={self.T} "
                    f"interval=0.000833 " # For 3s Resolution Data
                    f"Pmult=({P_mult_str})\n"
                    f"Qmult=({Q_mult_str})\n"
                )

            f.write("\n")

            # Loads
            for i in self.nodes:
                if i == 0: continue

                f.write(
                    f"New Load.Load_{i} "
                    f"bus1=bus{i} "
                    f"phases=1 "
                    f"conn=wye "
                    f"model=1 "
                    f"kV={self.V0/1e3} "
                    f"kW={load_kw[i-1]} "
                    f'kvar={load_kvar[i-1]} '
                    f"Daily=LS_{i}\n"
                )

            f.write("\n")

            # Simulation Setup
            f.write(f"Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize=3s\n") # Adjust if needed (should equal time resolution of load data).
            f.write(f"\nSolve\n")

    # Set of all nodes along path from root to node
    def C(self, i):
        path = []
        current = i

        # Walk to root using parent pointers
        while True:
            path.append(current)
            if current == 0:
                break
            current = self.parent[current]

        path.reverse()
        return path

    # Set of all nodes downstream of node i
    def D(self, i):
        stack = [i]
        downstream = []
        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))
        return downstream

    # Set of all lines along path from root to node
    def L(self, i):
        path = self.C(i)
        return [(path[k], path[k + 1]) for k in range(len(path) - 1)]

    def lin_dist_flow(self, tilde=False):

        v_drop = {} # |V_j|^2 - |V_i|^2
        v_dst = {}
        p_dst = {}
        q_dst = {}

        for t in range(self.T):

            for (i,j) in self.lines:

                p_ij = sum(self.P[h][t] for h in self.D(j))
                p_dst[(i, j, t)] = p_ij

                q_ij = sum(self.Q[h][t] for h in self.D(i))
                q_dst[(i,j,t)] = q_ij

                r_ij = self.r[(i, j)]
                x_ij = self.x[(i, j)]

                v_drop[(i, j, t)] = - 2 * (r_ij * p_ij + x_ij * q_ij) # |V_j|^2 - |V_i|^2

            for i in self.nodes:

                v_dst[(i,t)] = self.V0**2 - sum(v_drop[(h, k, t)] for h,k in self.L(i))

        if tilde:
            self.V_tilde = {k: np.sqrt(v) for k, v in v_dst.items()}
            self.p_tilde = p_dst
            self.q_tilde = q_dst
        else:
            self.V = {k: np.sqrt(v) for k, v in v_dst.items()}
            self.p_tilde = p_dst
            self.q_tilde = q_dst

    def power_flow_results(self, t=0, return_results=False, show=False, csv_folderpath=None, tilde=False):

        if tilde:
            V_src = self.V_tilde
            P_src = self.P_tilde
            Q_src = self.Q_tilde
            p_src = self.p_tilde
            q_src = self.q_tilde
        else:
            V_src = self.V_tilde
            P_src = self.P_tilde
            Q_src = self.Q_tilde
            p_src = self.p_tilde
            q_src = self.q_tilde

        # Node Table
        node_data = []
        for i in self.nodes:
            node_data.append({
                "node": i,
                "V": V_src[(i, t)],
                "P": P_src[i][t],
                "Q": Q_src[i][t],
            })
        df_nodes = pd.DataFrame(node_data).sort_values(by="node")

        # Line Table
        line_data = []
        for (i, j) in self.lines:
            line_data.append({
                "from": i,
                "to": j,
                "r": self.r[(i, j)],
                "x": self.x[(i, j)],
                "p_flow": p_src[(i, j, t)],
                "q_flow": q_src[(i, j, t)]
            })
        df_lines = pd.DataFrame(line_data).sort_values(by=["from", "to"])

        # Display
        if show:
            print(f"\n=== NODE STATES (t={t}) ===")
            print(df_nodes.to_string(index=False))
            print(f"\n=== LINE STATES (t={t}) ===")
            print(df_lines.to_string(index=False))

        # Save to CSV
        if csv_folderpath:
            nodes_csv_filepath = csv_folderpath + f"{self.name}_nodes.csv"
            lines_csv_filepath = csv_folderpath + f"{self.name}_lines.csv"
            df_nodes.to_csv(nodes_csv_filepath, index=False)
            df_lines.to_csv(lines_csv_filepath, index=False)

        if return_results: return df_nodes, df_lines
