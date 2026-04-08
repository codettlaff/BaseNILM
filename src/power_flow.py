import numpy as np
from tensorflow.python.autograph.utils.tensors import is_dense_tensor


class RadialNetwork:
    def __init__(self, nodes, edges, root=0, V0=1.0, alpha=1.0, epsilon=None):
        """
        nodes : dict {i: {"P": [P_i(t)], "B": [B_i(t)]}}
        edges : list of (i, j, r_ij, x_ij)
        root  : root node
        V0    : root node voltage
        alpha : constant power factor parameter (Q_i = alpha P_i)
        """

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

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))

            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij
            self.z[(i, j)] = r_ij + x_ij
            self.beta[(i, j)] = r_ij + self.alpha * x_ij

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
                self.i[(i,j)] = i_ij

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
                self.i_tilde[(i, j)] = i_ij

            # -------------------------
            # Node voltages
            # -------------------------
            for i in self.nodes:
                drops = sum(self.v_tilde[(k, j, t)] for (k, j) in self.L(i))
                self.V_tilde[(i, t)] = self.V0 - drops

    # ------------------------------------------------------------------
    # Empirical Accuracy
    # ------------------------------------------------------------------
    def compute_empirical_accuracy(self):

        p_num = 0
        p_den = 0
        i_num = 0
        i_den = 0

        for t in range(len(self.T)):
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

        for t in range(len(self.T)):
            for i in self.nodes:

                self.e_V = np.abs(self.V_tilde[(i, t)] - self.V[(i, t)])

                V_num += np.sqrt(self.e_V)
                V_dem += self.V[(i, t)]

        V_dem = 2 * V_dem
        self.acc_v = 1 - V_num / V_dem


