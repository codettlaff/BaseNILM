import numpy as np

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
        self.beta = {}

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))

            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij
            self.beta[(i, j)] = r_ij + self.alpha * x_ij

        # -----------------------------
        # Time-series results (initialized empty dicts)
        # -----------------------------
        self.p = {}  # {(i,j,t): P_ij(t)}
        self.V = {}  # {(i,t): V_i(t)}
        self.v = {}  # {(i,j,t): v_ij(t)}

        # -----------------------------
        # Privacy
        # -----------------------------
        self.eta = {}  # {(i,t): noise}
        self.P_tilde = {}  # {(i,t): noisy nodal power}

        self.p_tilde = {}  # {(i,j,t): noisy branch flow}
        self.V_tilde = {}  # {(i,t): noisy voltage}
        self.v_tilde = {}  # {(i,j,t): noisy voltage drop}

        # -----------------------------
        # Errors
        # -----------------------------
        self.e_i = {}  # {(i,j,t): current error}
        self.e_p = {}  # {(i,j,t): power error}
        self.e_V = {}  # {(i,t): voltage error}

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

        for i in self.nodes:

            var = 8 * self.B[i] / self.epsilon^2
            b = np.sqrt(var)
            self.eta[i] = np.random.laplace(0, b)
            self.P_tilde[i] = self.P[i] + self.eta[i]

    # ------------------------------------------------------------------
    # Power Flow
    # ------------------------------------------------------------------
    def power_flow(self):

        for ell in range(len(self.lines)):

            (i, j) = self.lines[ell]

            # line power flows = sum of downstream power injections at node j
            self.p[ell] = sum(self.P[h] for h in self.D(j))

            # line voltage drops
            self.v[ell] = self.beta[ell] * self.p[ell]

            # line current flows
            self.i[ell] = self.v[ell] / (self.r[ell] + self.x[ell])

        for i in self.nodes:

            L = self.L(i)
            for ell in range(len(L)):

                (i,j) = L(ell)

                # Nodal voltage magnitudes
                self.V[i] = self.V0 - sum(self.v[ell])

    def noisy_power_flow(self):

        for ell in range(len(self.lines)):

            (i, j) = self.lines[ell]

            # line power flows = sum of downstream power injections at node j
            self.p_tilde[ell] = sum(self.P_tilde[h] for h in self.D(j))

            # line voltage drops
            self.v_tilde[ell] = self.beta[ell] * self.p_tilde[ell]

            # line current flows
            self.i_tilde[ell] = self.v_tilde[ell] / (self.r[ell] + self.x[ell])

        for i in self.nodes:

            L = self.L(i)
            for ell in range(len(L)):

                (i,j) = L(ell)

                # Nodal voltage magnitudes
                self.V_tilde[i] = self.V0 - sum(self.v_tilde[ell])

    # ------------------------------------------------------------------
    # Empirical Accuracy
    # ------------------------------------------------------------------
    def compute_accuracy(self, e_p, T):