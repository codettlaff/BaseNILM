import numpy as np
import pandas as pd
import opendssdirect as dss

# Assumes root node index 0
# Assumes entire system one voltage level
# Assumes entire system one appliance bound
# Assumes single-phase system

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
        self.T = len(nodes["P"]) # Number of Timesteps
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
                    f"r1={r} x1={x} r0={r} x0{x}"
                    f"length=1 units=km\n" # Ohms per Unit Length
                )

            f.write("\n")

            # Loads
            for i in self.nodes:
                if i == self.root: # No Load at Substation
                    continue

                if tilde: P = self.P_tilde[i][t]
                else: P = self.P_tilde[i][t]
                Q = self.alpha * P

                f.write(
                    f"New Load.Load_{i} "
                    f"bus1=bus{i} "
                    f"phases=1"
                    f"conn=wye "
                    f"model=1 "
                    f"kV={self.V0/1e3} "
                    f"kW={P/1e3} "
                    f"kvar={Q/1e3} "
                )

        f.write("\n")
        f.write("Solve\n")