import networkx as nx
from collections import deque

def three_in_a_tree(graph, terminals):
    """
    Polynomial-time implementation to determine if an induced tree contains three terminal vertices.

    Args:
        graph (dict): Adjacency list of the graph.
        terminals (list): List of three terminal vertices.

    Returns:
        list: Nodes of the induced tree if found; otherwise False.
    """
    if len(terminals) != 3:
        raise ValueError("Exactly three terminals must be provided.")

    G = nx.Graph(graph)
    t1, t2, t3 = terminals

    for r in G:
        paths = []
        used = set()
        success = True
        for t in terminals:
            path = _shortest_induced_path(G, r, t, used)
            if path is None:
                success = False
                break
            internal = path[1:-1]
            if any(v in used for v in internal):
                success = False
                break
            used.update(internal)
            paths.append(path)
        if success:
            nodes = set().union(*paths)
            T = G.subgraph(nodes)
            if nx.is_tree(T):
                return list(T.nodes)

    return False

def _shortest_induced_path(G, src, dst, forbidden):
    """
    Finds a shortest induced path between two nodes avoiding forbidden nodes.

    Args:
        G (networkx.Graph): The graph.
        src (int): Source node.
        dst (int): Destination node.
        forbidden (set): Nodes to avoid in the path.

    Returns:
        list: List of nodes forming the induced path, or None if not found.
    """
    queue = deque([[src]])
    while queue:
        path = queue.popleft()
        last = path[-1]

        if last == dst:
            return path

        for nbr in G.neighbors(last):
            if nbr in path or nbr in forbidden:
                continue
            if not _is_induced(G, path, nbr):
                continue
            queue.append(path + [nbr])
    return None

def _is_induced(G, path, new_node):
    last = path[-1]
    if not G.has_edge(last, new_node):
        return False
    for node in path[:-1]:
        if G.has_edge(node, new_node):
            return False
    return True
