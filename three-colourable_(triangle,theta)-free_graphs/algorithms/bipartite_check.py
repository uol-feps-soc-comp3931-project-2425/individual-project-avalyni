import networkx
from collections import deque

def is_bipartite_with_colouring(graph):

    """
    Checks if the graph is bipartite using BFS and returns the partition, and the colouring if it is.

    Parameters:
        graph(dict): Adjacency list of the graph
    
    Returns:
        tuple: (bool, tuple or None, dict or None)
            - True, the partition as (X, Y), and the colouring if bipartite.
            - False, None, and None if not bipartite.
    """

    X = set() # Nodes at even distances
    Y = set() # Nodes at odd distances

    # Dictionary to track visited nodes and the length of their paths
    visited = {}

    for start_node in graph:
        if start_node not in visited:
            # Initialise BFS
            queue = deque([(start_node, 0)]) 
            visited[start_node] = 0

            while queue:
                node, dist = queue.popleft()

                if dist % 2 == 0:
                    X.add(node) # Even distance
                else:
                    Y.add(node) # Odd distance

                # Traverse neighbours
                for neighbour in graph[node]:
                    if neighbour not in visited:
                        # Mark the neighbour with its distance
                        visited[neighbour] = dist + 1
                        queue.append((neighbour, dist + 1))
                    elif visited[neighbour] % 2 == dist % 2:
                        # Nodes are in the same set, the graph is not bipartite
                        return False, None, None

    # If there are no conflicts, the graph is bipartite
    colouring = {node: "Colour 1" for node in X}
    colouring.update({node: "Colour 2" for node in Y})
    return True, (X, Y), colouring