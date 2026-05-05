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
                nodes[i]["P"] = [p * 1e3 for p in kw]  # kW to W

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