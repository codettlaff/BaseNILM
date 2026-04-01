import numpy as np

class RadialNetwork:
    def __init__(self, nodes, edges, root=0):
        """
        nodes: list of node indices
        edges: list of tuples (i, j, r_ij, x_ij)
        root: root node index
        """
        self.nodes = nodes
        self.root = root

        # store edge data
        self.children = {i: [] for i in nodes}
        self.parent = {}
        self.r = {}
        self.x = {}

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

    def get_subtree_nodes(self, i):
        """Return all downstream nodes D(i) including i"""
        result = []
        stack = [i]
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(self.children.get(node, []))
        return result

    def compute_branch_flows(self, p):
        """
        Compute P_ij = sum of loads downstream of j
        p: dict {node: real power injection}
        """
        P = {}

        # post-order traversal (bottom-up)
        order = self.topological_sort()[::-1]

        subtree_sum = {i: p.get(i, 0.0) for i in self.nodes}

        for j in order:
            if j != self.root:
                i = self.parent[j]
                subtree_sum[i] += subtree_sum[j]
                P[(i, j)] = subtree_sum[j]

        return P

    def compute_voltages(self, p, V0=1.0):
        """
        Compute voltages using LinDistFlow:
        V_j = V0 - sum(beta_l * P_l) along path
        """
        P = self.compute_branch_flows(p)
        V = {self.root: V0}

        order = self.topological_sort()

        for j in order:
            if j == self.root:
                continue

            i = self.parent[j]
            r_ij = self.r[(i, j)]
            x_ij = self.x[(i, j)]

            beta_ij = 2 * (r_ij + x_ij)

            V[j] = V[i] - beta_ij * P[(i, j)]

        return V, P

    def topological_sort(self):
        """Simple BFS order for radial tree"""
        order = []
        queue = [self.root]

        while queue:
            node = queue.pop(0)
            order.append(node)
            queue.extend(self.children.get(node, []))

        return order

def run_power_flow_example(aggregate_load_profile):
    """
    Runs radial power flow for each timestep.

    Parameters
    ----------
    aggregate_load_profile : 1D numpy array (T,)
        Aggregate load over time

    Returns
    -------
    V_ts : numpy array (T, n_nodes)
        Voltage time series

    P_ts : numpy array (T, n_edges)
        Branch power flow time series
    """

    # -------------------------------
    # Network Definition
    # -------------------------------
    nodes = [0, 1, 2, 3]

    edges = [
        (0, 1, 0.01, 0.02),
        (1, 2, 0.01, 0.02),
        (1, 3, 0.01, 0.02),
    ]

    net = RadialNetwork(nodes, edges, root=0)

    # Keep consistent ordering
    node_order = sorted(nodes)
    edge_order = [(i, j) for (i, j, _, _) in edges]

    T = len(aggregate_load_profile)

    V_ts = []
    P_ts = []

    # -------------------------------
    # Time Series Simulation
    # -------------------------------
    for t in range(T):

        p_total = aggregate_load_profile[t]

        # Distribute aggregate load across nodes
        # (you can refine this later if needed)
        p = {
            0: 0.0,
            1: 0.4 * p_total,
            2: 0.35 * p_total,
            3: 0.25 * p_total
        }

        V, P = net.compute_voltages(p, V0=1.0)

        # Convert dicts → ordered arrays
        V_vec = [V[n] for n in node_order]
        P_vec = [P[e] for e in edge_order]

        V_ts.append(V_vec)
        P_ts.append(P_vec)

    # Convert to numpy arrays
    V_ts = np.array(V_ts)  # shape (T, n_nodes)
    P_ts = np.array(P_ts)  # shape (T, n_edges)

    return V_ts, P_ts