from collections import deque

def is_bipartite_with_colouring(graph):

    """
    Determines if a graph is bipartite using Breadth-First Search (BFS) and, if so,
    returns a valid bipartite partition and colouring.

    A graph is bipartite if its vertices can be divided into two disjoint sets (X, Y)
    such that no two adjacent vertices belong to the same set. This function
    assigns alternating levels (even/odd) while performing BFS to verify this property.

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        tuple: (bool, tuple or None, dict or None)
            - (True, (X, Y), colouring): If the graph is bipartite.
            - (False, None, None): If the graph is not bipartite.
    
    """

    # Sets representing the two partitions of the bipartite graph
    X = set() # Nodes at even distances from the BFS root
    Y = set() # Nodes at odd distances

    # Dictionary to track visited nodes and their levels in the BFS tree
    visited = {}

    # Iterate over all nodes to handle disconnected graphs
    for start_node in graph:
        if start_node not in visited:
            # Initialise BFS with the first unvisited node at level 0
            queue = deque([(start_node, 0)]) 
            visited[start_node] = 0 # Mark the start node with level 0

            while queue:
                node, dist = queue.popleft()

                # Assign the node to the appropriate set based on BFS level
                if dist % 2 == 0:
                    X.add(node) # Even level nodes go to set X
                else:
                    Y.add(node) # Odd level nodes go to set Y

                # Traverse all adjacent nodes
                for neighbour in graph[node]:
                    if neighbour not in visited:
                        # Assign an alternating level and enqueue the neighbour
                        visited[neighbour] = dist + 1
                        queue.append((neighbour, dist + 1))
                    elif visited[neighbour] % 2 == dist % 2:
                        # Conflict detected: two adjacent nodes have the same level
                        return False, None, None # Graph is not bipartite

    # If no conflicts were found, construct the bipartite colouring
    colouring = {node: "Colour 1" for node in X}
    colouring.update({node: "Colour 2" for node in Y})

    return True, (X, Y), colouring