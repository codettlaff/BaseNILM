import numpy as np

class RadialNetwork:
    def __init__(self, nodes, edges, root=0, V0=1.0, alpha=1.0, epsilon=None):
        """
        nodes : dict {i: {"P": P_i, "B": B_i}}
        edges : list of (i, j, r_ij, x_ij)
        root  : root node
        V0    : root node voltage
        alpha : constant power factor parameter (Q_i = alpha P_i)
        """
        self.nodes = list(nodes.keys())  # Node indices
        self.P = {i: data["P"] for i, data in nodes.items()} # Nodal active power injections
        self.B = {i: data["B"] for i, data in nodes.items()} # Appliance power bound
        self.root = root
        self.V0 = V0
        self.alpha = alpha

        # Tree structure
        self.children = {i: [] for i in self.nodes}
        self.parent = {}
        self.lines = []

        # Line parameters
        self.r = []
        self.x = []
        self.beta = []

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))
            self.r.append(r_ij)
            self.x.append(x_ij)
            self.beta.append(r_ij + self.alpha * x_ij) # β_ij = r_ij + α x_ij

        self.p = [] # list of floats active power branch flows. empty until power flow solved
        self.V = [] # list of floats nodal voltage injections. empty until power flow solved
        self.v = [] # list of floats branch voltage drops, empty until power flow solved

        # Privacy Stuff
        if epsilon: self.do_differential_privacy = True
        else: self.do_differential_privacy = False
        self.epsilon = epsilon
        self.eta = [] # list of noise added to active power injection at each node, empty until differntial privacy calculated

        self.p_tilde = [] # list of noisy active power branch flows. empty until power flow solved
        self.V_tilde = [] # list of noisy floats nodal voltage injections. empty until power flow solved
        self.v_tilde = [] # list of noisy branch voltage drops, empty until power flow solved

        self.e_i = [] # list of line current flow error, empty until error calculated
        self.e_p = [] # list of line power flow error, empty until error calculated
        self.e_V = [] # list of nodal voltage error, empty until error calculated

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
    # Power Flow
    # ------------------------------------------------------------------
    def compute_voltage_drops(self):
        """
        Compute voltage drops v_{i,j} on each line (i,j),
        where v_{i,j} = beta_{i,j} * sum_{h in D(j)} P_h.
        """

        for (i, j) in self.lines:
            # Sum of downstream power injections at node j
            P_sum = sum(self.P[h] for h in self.D(j))

            # Voltage drop on line (i,j)
            self.v.append(self.beta[(i, j)] * P_sum)

    def compute_line_flows(self):
        """
        Compute current flows i_{i,j} on each line (i,j)
        """

        for (i,j) in self.lines:
            self.i.append(self.v / (self.r + self.x))