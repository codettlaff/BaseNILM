import numpy as np
import pandas as pd
import opendssdirect as dss
from opendssdirect.Lines import Length
from sympy.matrices.benchmarks.bench_matrix import timeit_Matrix__getitem_II


# Assumes root node index 0
# Assumes entire system one voltage level
# Assumes entire system one appliance bound
# Assumes single-phase system

# Want to be able to initialize from DSS file, without providing nodes and edges.
# This will require dealing with OpenDSS in time-series, rather than for a single timestep.

class RadialNetwork:
    def __init__(self, name, nodes=None, edges=None, V0=12.47e3, alpha=0.0, epsilon=None, B=5e3, build_from_dss=False, dss_filepath='network.dss'):
        """
        nodes: dict {i: {"P": [P_i(t)]}}
        edges: list of (i, j, r_ij, x_ij)
        V0: root node voltage, used as per-unit voltage base.
        alpha: constant power factor parameter (Q_i = alpha * P_i)
        epsilon: privacy budget
        B: appliance power bound
        dss_filepath: path to master dss file
        """

        self.name = name
        self.dss_filepath = dss_filepath

        if build_from_dss:
            self.build_from_dss_timeseries()

        else:

            # Time Series
            self.nodes = list(nodes.keys()) # List of Node Indices
            self.T = len(nodes[1]["P"]) # Number of Timesteps
            self.P = {i: data["P"] for i, data in nodes.items()} # Copy True Injections
            self.P_tilde = {i: [0.0] * self.T for i, data in nodes.items()} # Initialize Noisy Injections

            # Root Node
            self.V0 = V0
            self.nodes.insert(0, 0)
            self.P[0] = [0.0] * self.T

            # Tree Structure
            self.children = {i: [] for i in self.nodes} # Initialize Dict
            self.parent = {}
            self.lines = []

            # Line Parameters
            self.r = {} # Unit Ohms
            self.x = {} # Unit Ohms

            for i, j, r_ij, x_ij in edges:
                self.children[i].append(j)
                self.parent[j] = i
                self.lines.append((i,j))
                self.r[(i,j)] = r_ij
                self.x[(i,j)] = x_ij

        # Constants
        self.alpha = alpha
        self.epsilon = epsilon
        self.B = B

        # Theoretical Error
        self.e_p_th = {} # {(i,j,t): e_p_ij} # Power Flow Error (Theoretical)
        self.e_i_th = {} # {(i,j,t): e_i_ij} # Current Flow Error (Theoretical)
        self.e_v_th = {}  # {(i,j,t): e_v_ij} # Voltage Drop Error (Theoretical)
        self.e_V_th = {} # {(i,t): e_V_ij} # Nodal Voltage Error (Theoretical)

        # Theoretical Error Standard Deviation
        self.std_e_p_th = {} # {(i,j,t): std_e_p_ij} # Power Flow Error Standard Deviation (Theoretical)
        self.std_e_i_th = {} # {(i,j,t): std_e_i_ij} # Current Flow Error Standard Deviation (Theoretical)
        self.std_e_v_th = {} # {(i,j,t): std_e_v_ij} # Voltage Drop Error Standard Deviation (Theoretical)
        self.std_e_V_th = {} # {(i,t): std_e_V_ij} # Nodal Voltage Error Standard Deviation (Theoretical)

        # Theoretical Accuracy Lower Bound
        self.p_acc_th_lb = 0.0 # Power Flow Accuracy Lower Bound (Theoretical)
        self.i_acc_th_lb = 0.0 # Current Flow Accuracy Lower Bound (Theoretical)
        self.v_acc_th_lb = 0.0 # Voltage Drop Accuracy Lower Bound (Theoretical)
        self.V_acc_th_lb = 0.0 # Nodal Voltage Accuracy Lower Bound (Theoretical)

        # Theoretical Expected Accuracy (Gaussian Approximation)
        self.p_acc_th = 0.0 # Expected Power Flow Accuracy (Theoretical)
        self.i_acc_th = 0.0 # Expected Current Flow Accuracy (Theoretical)
        self.v_acc_th = 0.0 # Expected Voltage Drop Accuracy (Theoretical)
        self.V_acc_th = 0.0 # Expected Nodal Voltage Accuracy (Theoretical)

        # True Time Series Results
        self.p = {}  # {(i,j,t): P_ij(t)}
        self.i = {}  # {(i,j,t): i_ij(t)}
        self.v = {}  # {(i,j,t): v_ij(t)}
        self.V = {}  # {(i,t): V_i(t)}

        # Noisy Time Series Results
        self.p_tilde = {}  # {(i,j,t): noisy branch flow}
        self.i_tilde = {}  # {(i,j,t): i_ij(t)}
        self.V_tilde = {}  # {(i,t): noisy voltage}
        self.v_tilde = {}  # {(i,j,t): noisy voltage drop}

        # Empirical Results Error
        self.e_p = {} # {(i,j,t): e_p_ij} # Power Flow Error (Empirical)
        self.e_i = {} # {(i,j,t): e_i_ij} # Current Flow Error (Empirical)
        self.e_v = {} # {(i,j,t): e_v_ij} # Voltage Drop Error (Empirical)
        self.e_V = {} # {(i,t): e_V_i} # Nodal Voltage Error (Empirical)

    def export_to_dss_timeseries(self, tilde=False):
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

            # Load-Shapes
            for i in self.nodes:
                if i == 0:
                    continue

                if tilde: series = self.P_tilde[i]
                else: series = self.P[i]

                max_kw = np.max(series) / 1e3
                load_kw.append(max_kw)
                series_scaled = series / np.max(series) if np.max(series) != 0 else np.zeros_like(series)

                mult_str = " ".join(str(p) for p in series_scaled)

                f.write(
                    f"New LoadShape.LS_{i} "
                    f"npts={self.T} "
                    f"interval=0.000833 " # For 3s Resolution Data
                    f"Pmult=({mult_str})\n"
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
                    f"Daily=LS_{i}\n"
                )

            f.write("\n")

            # Simulation Setup
            f.write(f"Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize=3s\n") # Adjust if needed (should equal time resolution of load data).
            f.write(f"\nSolve\n")

    def build_from_dss_timeseries(self):

        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

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
            i: {"P": [0.0] * T}
            for i in bus_map.values()
        }

        # Extract Loads - Full time-series
        dss.Loads.First()
        while True:
            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]

            load_peak_kw = dss.Loads.kW()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = [load_peak_kw * s for s in dss.LoadShape.PMult()]

            for t in range(T):
                nodes[i]["P"] = [p * 1e3 for p in kw] # kW to W

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
        self.P_tilde = {i: [0.0] * self.T for i, data in nodes.items()}  # Initialize Noisy Injections

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

    # LinDist Power Flow
    def lin_dist_power_flow(self, tilde=False):

        V_dst = {}
        p_dst = {}
        i_dst = {}
        v_dst = {}

        for t in range(self.T):

            for (i,j) in self.lines:

                # Branch Power Flow
                p_ij = sum(self.P[h][t] for h in self.D(j))
                p_dst[(i, j, t)] = p_ij

                # Voltage Drop
                beta = self.r[(i, j)] + self.alpha * self.x[(i, j)]
                v_ij = beta * p_ij
                v_dst[(i, j, t)] = v_ij

                # Branch Current Flow
                z = beta / (self.r[(i, j)] + self.x[(i, j)])
                i_ij = z * p_ij
                i_dst[(i, j, t)] = i_ij

        for i in self.nodes:
            drops = sum(v_dst[(k, j, t)] for (k, j) in self.L(i))
            V_dst[(i, t)] = self.V0 - drops

        if tilde:
            self.V_tilde = V_dst
            self.p_tilde = p_dst
            self.i_tilde = i_dst
            self.v_tilde = v_dst

        if not tilde:
            self.V = V_dst
            self.p = p_dst
            self.i = i_dst
            self.v = v_dst

    # Solve DSS Power Flow
    def dss_power_flow_step_by_step(self, tilde=False):

        V_dst = {}
        p_dst = {}
        i_dst = {}
        v_dst = {}

        # Export time-series DSS file
        self.export_to_dss_timeseries(tilde=tilde)

        # Compile Once
        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

        # Solve step-by-step
        for t in range(self.T):

            dss.Text.Command(f"set time=(0, {t*3}")
            dss.Text.Command("Solve")

            # Bus Voltages
            bus_names = dss.Circuit.AllBusNames()
            for bus in bus_names:
                dss.Circuit.SetActiveBus(bus)
                vmag = dss.Bus.puVmagAngle()[0]
                i = int(bus.replace("bus", ""))
                V_dst[(i,t)] = vmag

            # Line Flows
            dss.Lines.First()
            while True:
                name = dss.Lines.Name()

                # Parse Line Name
                _, i_str, j_str = name.split("_")
                i = int(i_str)
                j = int(j_str)

                dss.Circuit.SetActiveElement(f"Line.{name}")

                powers = dss.CktElement.Powers()
                P_from = powers[0]  # terminal 1
                P_to = abs(powers[1])  # terminal 2
                P_loss = abs(P_from - P_to)

                currents = dss.CktElement.Currents()

                # Real Power
                P_ij = P_from * 1e3 # kW to W
                p_dst[(i, j, t)] = P_ij

                # Current magnitude
                I_real  = currents[0]
                I_imag = currents[1]
                I_mag = np.sqrt(I_real ** 2 + I_imag ** 2)
                i_dst[(i, j, t)] = I_mag

                # Voltage Drop
                v_dst[(i, j, t)] = V_dst[(i, t)] - V_dst[(j, t)]

                if not dss.Lines.Next():
                    break

        # Store Results
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

    # This function may not work when system voltage is too high.
    def dss_power_flow_timeseries(self, tilde=False):

        V_dst = {}
        p_dst = {}
        i_dst = {}
        v_dst = {}

        # Export full time-series DSS
        self.export_to_dss_timeseries(tilde=tilde)

        # Compile
        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

        # Bus Voltage Monitor
        for i in self.nodes:
            if i ==0: continue
            dss.Text.Command(
                f"New Monitor.V_{i} element=Load.Load_{i} mode=0 terminal=1"
            )

        # Line Monitors (power + current)
        for (i,j) in self.lines:
            dss.Text.Command(
                f"New Monitor.P_{i}_{j} element=Line.L_{i}_{j} mode=1 terminal=1" # Power Mode
            )
            dss.Text.Command(
                f"New Monitor.I_{i}_{j} element=Line.L_{i}_{j} mode=0 terminal=1" # Standard Mode
            )

        dss.Text.Command("Solve")

        # Extract Voltage Monitor Data
        for i in self.nodes:
            if i == 0:
                dss.Circuit.SetActiveBus(dss.Circuit.AllBusNames()[0])
                for t in range(self.T):
                    V_dst[(i, t)] = dss.Bus.puVmagAngle()[0]
                continue
            name = f"v_{i}"
            dss.Monitors.Name(name)
            data = dss.Monitors.Channel(1) # Voltage magnitude
            for t, v in enumerate(data):
                V_dst[(i,t)] = v

        # Line Flows + Currents
        for (i,j) in self.lines:

            # Power
            dss.Monitors.Name(f"p_{i}_{j}")
            p_data = dss.Monitors.Channel(1)

            # Current
            dss.Monitors.Name(f"i_{i}_{j}")
            I_mag = dss.Monitors.Channel(3) # Channel 3 Imag, Channel 4 I Phase

            # Voltage - Method 2 - Same result - Voltage not changing.
            # V_mag = dss.Monitors.Channel(1)

            for t in range(len(p_data)):
                P_ij = p_data[t] * 1e3 # kW to W
                p_dst[(i, j, t)] = P_ij
                i_dst[(i, j, t)] = I_mag[t]

        # Voltage Drops
        for (i,j) in self.lines:
            for t in range(self.T):
                v_dst[(i, j, t)] = V_dst[(i, t)] - V_dst[(j, t)]

        # Store Results
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

    def power_flow_results(self, t=0, return_results=False, display_results=False, write_csv=False,
                           results_folderpath=None, tilde=False):

        if tilde:
            V_src = self.V_tilde
            P_src = self.P_tilde
            i_src = self.i_tilde
            p_src = self.p_tilde
            v_src = self.v_tilde
        else:
            V_src = self.V
            P_src = self.P
            i_src = self.i
            p_src = self.p
            v_src = self.v

        # Node Table
        node_data = []
        for i in self.nodes:
            node_data.append({
                "node": i,
                "V": V_src[(i, t)],
                "P_injection": P_src[i][t],
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
                "i_flow": i_src[(i, j, t)],
                "v_drop": v_src[(i, j, t)],
            })
        df_lines = pd.DataFrame(line_data).sort_values(by=["from", "to"])

        # Display
        if display_results:
            print("\n=== NODE STATES (t={}) ===".format(t))
            print(df_nodes.to_string(index=False))
            print("\n=== LINE STATES (t={}) ===".format(t))
            print(df_lines.to_string(index=False))

        # Save to CSV
        if write_csv:
            nodes_csv_filepath = results_folderpath + f"{self.name}_nodes_t{t}.csv"
            lines_csv_filepath = results_folderpath + f"{self.name}_lines_t{t}.csv"
            df_nodes.to_csv(nodes_csv_filepath)
            df_lines.to_csv(lines_csv_filepath)
        if return_results: return df_nodes, df_lines

    def compute_theoretical_accuracy(self):

        acc_p_bound_num = 0
        acc_p_bound_den = 0
        acc_i_bound_num = 0
        acc_i_bound_den = 0

        sigma_p = {}
        sigma_i = {}
        sigma_V = []

        for t in range(self.T):

            for (i,j) in self.lines:

                sum = 0.0
                for h in self.D(j):
                    sum += self.B**2

                sigma_p[(i,j,t)] = (np.sqrt(8) / self.epsilon) * np.sqrt(sum)

                r_ij = self.r[(i, j)]
                x_ij = self.x[(i, j)]

                sigma_i[(i, j, t)] = (r_ij + self.alpha * x_ij) / (r_ij + x_ij) * sigma_p[(i,j,t)]

            for i in self.nodes:
                beta = 0.0
                var = 0.0
                for j in self.L(i):
                    for h in self.D(j):
                        r_ij = self.r[(i, j)]
                        x_ij = self.x[(i, j)]
                        beta += (r_ij + self.alpha * x_ij)
                        var += (8 * beta**2 * self.B**2) / self.epsilon**2
                sigma_V.append(np.sqrt(var))

        print('')