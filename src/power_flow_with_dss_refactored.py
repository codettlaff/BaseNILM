import numpy as np
import pandas as pd
import opendssdirect as dss
from opendssdirect.Lines import Length


# Assumes root node index 0
# Assumes entire system one voltage level
# Assumes entire system one appliance bound
# Assumes single-phase system

# Want to be able to initialize from DSS file, without providing nodes and edges.
# This will require dealing with OpenDSS in time-series, rather than for a single timestep.

class RadialNetwork:
    def __init__(self, name, nodes, edges, V0=12.47e3, alpha=0.0, epsilon=None, B=5e3, dss_filepath='network.dss'):
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

        # Time Series
        self.nodes = list(nodes.keys()) # List of Node Indices
        self.T = len(nodes[0]["P"]) # Number of Timesteps
        self.P = {i: data["P"] for i, data in nodes.items()} # Copy True Injections
        self.P_tilde = {i: [0.0] * self.T for i, data in nodes.items()} # Initialize Noisy Injections

        # Constants
        self.V0 = V0
        self.alpha = alpha
        self.epsilon = epsilon
        self.B = B

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

    def export_to_dss(self, t=0, tilde=False):

        with open(self.dss_filepath, 'w') as f:

            # Circuit Definition - Automatically Creates Default Source Vsource.Source
            f.write(f"Clear\n")
            f.write(f"New Circuit.{self.name} basekv={self.V0/1e3} pu=1.0 \n")
            f.write(f"Edit Vsource.Source bus1=bus0 \n") # Make Sure Root is Index 0

            # Lines
            for (i,j) in self.lines:
                r = self.r[(i,j)]
                x = self.x[(i,j)]
                f.write(
                    f"New Line.L_{i}_{j} "
                    f"bus1=bus{i} bus2=bus{j} "
                    f"r1={r} x1={x} r0={r} x0{x} "
                    f"length=1 units=km\n" # Ohms per Unit Length
                )

            f.write("\n")

            # Loads
            for i in self.nodes:
                if i == 0: # No Load at Substation
                    continue

                if tilde: P = self.P_tilde[i][t]
                else: P = self.P_tilde[i][t]
                Q = self.alpha * P

                f.write(
                    f"New Load.Load_{i} "
                    f"bus1=bus{i} "
                    f"phases=1 "
                    f"conn=wye "
                    f"model=1 "
                    f"kV={self.V0/1e3} "
                    f"kW={P/1e3} "
                    f"kvar={Q/1e3} "
                )

        f.write("\n")
        f.write("Solve\n")

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

            # Load-Shapes
            for i in self.nodes:
                if i == 0:
                    continue

                if tilde: series = self.P_tilde[i]
                else: series = self.P[i]

                mult_str = " ".join(str(p / 1e3) for p in series) # Convert W to kW

                f.write(
                    f"New LoadShape.LS_{i} "
                    f"npts={self.T} "
                    f"interval=0.000833 " # For 3s Resolution Data
                    f"UseActual=Yes "
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
                    f"Daily=LS_{i}\n"
                )

            f.write("\n")

            # Simulation Setup
            f.write(f"Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize=3s\n") # Adjust if needed (should equal time resolution of load data).
            f.write(f"\nSolve\n")

    def build_from_dss(self, t=0):

        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

        # Map buses to indices
        bus_names = dss.Circuit.AllBusNames()
        bus_map = {name: idx for idx, name in enumerate(bus_names)}

        # Initialize
        nodes = {
            i: {"P": self.P[i]}
            for i in bus_map.values()
        }

        # Extract Loads to Nodal P
        dss.Loads.First()
        while True:
            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]
            p_kw = dss.Loads.kW()
            nodes[i]["P"][t] = p_kw * 1e3
            if not dss.Loads.Next():
                break

        # Extract Lines to Edges
        edges = []

        dss.Lines.First()
        while True:
            bus1 = dss.Lines.Bus1().split(".")[0]
            bus2 = dss.Lines.Bus2().cplit(".")[0]
            i = bus_map[bus1]
            j = bus_map[bus2]
            length = dss.Lines.Length()
            r = dss.Lines.R1() * length
            x = dss.Lines.X1() * length
            edges.append((i,j,r,x))
            if not dss.Lines.Next():
                break

        self.nodes = nodes
        self.edges = edges

    def build_from_dss_timeseries(self):
        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

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

            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = dss.LoadShape.PMult()
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

        self.nodes = nodes
        self.edges = edges