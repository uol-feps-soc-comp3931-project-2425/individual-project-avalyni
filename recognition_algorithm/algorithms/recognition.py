from itertools import combinations

def contains_triangle(graph):
    """
    Checks whether the input graph contains a triangle (3-cycle).

    Args:
        graph (dict): Adjacency list of the graph.

    Returns:
        bool: True if a triangle is found, False otherwise.
    """
    for v in graph:
        neighbors = graph[v]
        for u in neighbors:
            for w in neighbors:
                if u != w and w in graph.get(u, set()):
                    return True
    return False

def prune_graph(graph, a, keep):
    """
    Produces a pruned version of the graph where only the selected neighbors of a are kept.

    Args:
        graph (dict): Original graph.
        a (int): Central node whose neighbors are pruned.
        keep (list): Neighbors of 'a' to retain.

    Returns:
        dict: Pruned graph adjacency list.
    """
    to_remove = set(graph[a]) - set(keep)
    pruned = {}
    for v in graph:
        if v == a or v in to_remove:
            continue
        pruned[v] = [u for u in graph[v] if u not in to_remove and u != a]
    return pruned

def find_theta(graph, three_in_a_tree):
    """
    Determines whether the graph contains a theta structure.

    Args:
        graph (dict): Graph to analyze.
        three_in_a_tree (function): Function to detect induced trees containing 3 terminals.

    Returns:
        (bool, None): True if a theta is found, False otherwise.
    """
    if contains_triangle(graph):
        return True, None

    for a in graph:
        neighbors = list(graph[a])
        if len(neighbors) < 3:
            continue

        for b1, b2, b3 in combinations(neighbors, 3):

            G_prime = prune_graph(graph, a, [b1, b2, b3])
            if three_in_a_tree(G_prime, [b1, b2, b3]):
                return True, None

    return False, None
