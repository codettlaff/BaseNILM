import numpy as np


class RadialNetwork:
    def __init__(self, nodes, edges, root=0, alpha=1.0):
        """
        nodes : list of node indices i ∈ {0,...,N}
        edges : list of (i, j, r_ij, x_ij)
        root  : root node (substation)
        alpha : constant power factor parameter (Q_i = alpha * P_i)
        """
        self.nodes = nodes
        self.root = root
        self.alpha = alpha

        # Tree structure
        self.children = {i: [] for i in nodes}
        self.parent = {}

        # Line parameters
        self.r = {}
        self.x = {}

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

    # ------------------------------------------------------------------
    # D(i): downstream nodes (including i)
    # ------------------------------------------------------------------
    def D(self, i):
        """Return D(i): all nodes downstream of i (including i)"""
        stack = [i]
        downstream = []

        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))

        return downstream

    # ------------------------------------------------------------------
    # Branch Flow:  P_ij = sum_{h ∈ D(j)} P_h
    # ------------------------------------------------------------------
    def compute_branch_flows(self, P_i):
        """
        P_i : dict {i: real power injection at node i}

        Returns:
            P_ij : dict {(i,j): branch flow}
        """
        P_ij = {}

        # Initialize subtree sums with nodal injections
        subtree_sum = {i: P_i.get(i, 0.0) for i in self.nodes}

        # Process nodes bottom-up
        order = self.topological_order()[::-1]

        for j in order:
            if j == self.root:
                continue

            i = self.parent[j]

            # accumulate downstream load
            subtree_sum[i] += subtree_sum[j]

            # branch flow definition
            P_ij[(i, j)] = subtree_sum[j]

        return P_ij

    # ------------------------------------------------------------------
    # β_ij = 2(r_ij + α x_ij)
    # ------------------------------------------------------------------
    def beta(self, i, j):
        return 2 * (self.r[(i, j)] + self.alpha * self.x[(i, j)])

    # ------------------------------------------------------------------
    # Voltage: V_j = V_i - β_ij P_ij
    # ------------------------------------------------------------------
    def compute_voltages(self, P_i, V0=1.0):
        """
        P_i : nodal real power
        V0  : root voltage

        Returns:
            V_i  : nodal voltages
            P_ij : branch flows
        """
        P_ij = self.compute_branch_flows(P_i)

        V_i = {self.root: V0}

        for j in self.topological_order():
            if j == self.root:
                continue

            i = self.parent[j]
            beta_ij = self.beta(i, j)

            V_i[j] = V_i[i] - beta_ij * P_ij[(i, j)]

        return V_i, P_ij

    # ------------------------------------------------------------------
    # V_j = V_0 - sum_{ℓ ∈ C(j)} β_ℓ P_ℓ
    # (path formulation, matches paper exactly)
    # ------------------------------------------------------------------
    def compute_voltages_path_form(self, P_i, V0=1.0):
        """
        Alternative implementation using path set C(j)
        """
        P_ij = self.compute_branch_flows(P_i)
        V_i = {}

        for j in self.nodes:
            if j == self.root:
                V_i[j] = V0
                continue

            path = self.path_to_root(j)

            voltage_drop = 0.0
            for (i, k) in path:
                voltage_drop += self.beta(i, k) * P_ij[(i, k)]

            V_i[j] = V0 - voltage_drop

        return V_i, P_ij

    # ------------------------------------------------------------------
    # C(j): edges on path from root to j
    # ------------------------------------------------------------------
    def path_to_root(self, j):
        """Return list of edges ℓ ∈ C(j) from root to j"""
        path = []
        current = j

        while current != self.root:
            parent = self.parent[current]
            path.append((parent, current))
            current = parent

        return path[::-1]  # root → j order

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------
    def topological_order(self):
        """Breadth-first order from root"""
        order = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            order.append(node)
            queue.extend(self.children.get(node, []))

        return order