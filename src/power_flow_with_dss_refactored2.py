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
