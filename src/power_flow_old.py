import numpy as np


class RadialNetwork:
    def __init__(self, nodes, edges, root=0, V0=1.0, alpha=1.0):
        """
        nodes : list of node indices
        edges : list of (i, j, r_ij, x_ij)
        root  : root node
        V0    : root node voltage
        alpha : constant power factor parameter (Q_i = alpha P_i)
        """
        self.nodes = nodes
        self.root = root
        self.V0 = V0
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
    # C(i): path_to_root
    # ------------------------------------------------------------------
    def C(self, i):
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
    # D(i): downstream nodes
    # ------------------------------------------------------------------
    def D(self, i):
        stack = [i]
        downstream = []

        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))

        return downstream

    # ------------------------------------------------------------------
    # Branch flow: P_ij = sum_{h ∈ D(j)} P_h
    # ------------------------------------------------------------------
    def compute_branch_flows(self, P_i):
        P_ij = {}

        subtree_sum = {i: P_i.get(i, 0.0) for i in self.nodes}
        order = self.topological_order()[::-1]

        for j in order:
            if j == self.root:
                continue

            i = self.parent[j]
            subtree_sum[i] += subtree_sum[j]
            P_ij[(i, j)] = subtree_sum[j]

        return P_ij

    # ------------------------------------------------------------------
    # β_ij = 2(r_ij + α x_ij)
    # ------------------------------------------------------------------
    def beta(self, i, j):
        return 2 * (self.r[(i, j)] + self.alpha * self.x[(i, j)])

    # ------------------------------------------------------------------
    # Main function:
    # Returns:
    #   V_i  = |V_i|^2  (squared voltage magnitude)
    #   I_ij = |I_ij|   (current magnitude)
    # ------------------------------------------------------------------
    def solve_power_flow(self, P_i):
        """
        Solve LinDistFlow power flow.

        Parameters
        ----------
        P_i : dict
            Nodal real power injections P_i

        Returns
        -------
        V_i : dict
            Squared voltages {i: |V_i|^2}
        P_ij : dict
            Branch real power flows {(i,j): P_{ij}}
        I_ij : dict
            Branch current magnitudes {(i,j): |I_{ij}|}
        """

        # --------------------------------------------------
        # Step 1: compute branch flows P_{ij}
        # --------------------------------------------------
        P_ij = self.compute_branch_flows(P_i)

        # --------------------------------------------------
        # Step 2: compute squared voltages V_i = |V_i|^2
        # --------------------------------------------------
        V_i = {self.root: self.V0}

        for j in self.topological_order():
            if j == self.root:
                continue

            i = self.parent[j]
            beta_ij = self.beta(i, j)

            # LinDistFlow voltage equation
            V_i[j] = V_i[i] - beta_ij * P_ij[(i, j)]

        # --------------------------------------------------
        # Step 3: compute current magnitudes I_{ij}
        # --------------------------------------------------
        I_ij = {}

        for (i, j), P in P_ij.items():

            if V_i[i] <= 0:
                raise ValueError(f"Non-physical voltage at node {i}: {V_i[i]}")

            # |I_{ij}| ≈ P_{ij} / sqrt(V_i)
            I_ij[(i, j)] = P / np.sqrt(V_i[i])

        return V_i, P_ij, I_ij

    # ------------------------------------------------------------------
    # Tree traversal
    # ------------------------------------------------------------------
    def topological_order(self):
        order = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            order.append(node)
            queue.extend(self.children.get(node, []))

        return order