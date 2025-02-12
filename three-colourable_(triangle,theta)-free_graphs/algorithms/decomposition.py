from collections import deque, defaultdict
from itertools import combinations

def is_connected(graph):
    """Check if a graph is connected using BFS."""
    nodes = list(graph.keys())
    if not nodes:
        return True 
    
    visited = set()
    queue = deque([nodes[0]])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    
    return len(visited) == len(nodes)

def is_graph_disconnected(graph, removed_nodes):
    """Returns True if removing the given nodes disconnects the graph."""
    for start in graph:  # Find the first non-removed node
        if start not in removed_nodes:
            break
    else:
        return True  # All nodes are removed, graph is trivially disconnected
    
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited and neighbor not in removed_nodes)
    
    # If we didn't visit all non-removed nodes, the graph is disconnected
    return len(visited) < len(graph) - len(removed_nodes)

def is_clique(graph, nodes):
    """Check if a set of nodes form a clique more efficiently."""
    nodes = set(nodes)  # Convert list to set for fast lookup
    for node in nodes:
        if len(graph[node]) < len(nodes) - 1:  # A clique requires full connectivity
            return False
        if not nodes.issubset(set(graph[node]) | {node}):  # Ensure all neighbors exist
            return False
    return True

def find_clique_cutset(graph):
    """
    Find the smallest clique cutset (K₁ or K₂) in the graph.
    Ensures that the cutset is a true separator and forms a clique.
    """
    if not graph:
        return {"error": "Graph is empty. Algorithm terminated."}
    if not is_connected(graph):
        return {"error": "Graph is not connected. Algorithm terminated."}

    # Debugging: Print graph structure at each check
    #print("\n\033[1m[DEBUG] Checking clique cutset for the following graph structure:\033[0m")
    graph_snapshot = {node: list(neighbors) for node, neighbors in graph.items()}  # Prevent mutation
    for node, neighbors in sorted(graph_snapshot.items()):
        print(f"  {node}: {sorted(neighbors)}")

    # Look for a K1 cutset (single vertex whose removal disconnects the graph)
    for node in graph:
        if is_graph_disconnected(graph, {node}) and len(graph) > 1:
            #print(f"\n\033[1m[DEBUG] Found K1 Cutset:\033[0m {node}")
            return {"type": "K1", "cutset": [node]}

    # Look for a K2 cutset (pair of adjacent nodes whose removal disconnects the graph)
    checked_pairs = set()
    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:  # Ensure each pair is checked only once
                if (neighbor in graph[node]) and is_graph_disconnected(graph, {node, neighbor}):
                    # Ensure that removing the cutset actually splits the graph into at least two parts
                    components = get_connected_components(graph, {node, neighbor})
                    if len(components) >= 2 and len(graph) > 2:  # Must split into at least 2 components
                        #print(f"\n\033[1m[DEBUG] Found K2 Cutset:\033[0m [{node}, {neighbor}]")
                        return {"type": "K2", "cutset": [node, neighbor]}

    #print("\n\033[1m[DEBUG] No valid clique cutset found.\033[0m")  # Debugging when no cutset is found
    return {"type": None, "cutset": None}


def get_connected_components(graph, removed_nodes):
    """Find connected components after removing specified nodes."""
    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited and neighbor not in removed_nodes:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited and node not in removed_nodes:
            component = []
            dfs(node, component)
            components.append(set(component))

    return components

class DecompositionTree:
    def __init__(self, graph, cutset=None):
        """
        Represents a node in the decomposition tree.
        """
        self.graph = graph  # The current subgraph at this node
        self.cutset = cutset  # The clique cutset that led to this node
        self.children = []  # List of child nodes (subgraphs)
        self.is_extreme_block = False  # Will be True if this is a leaf/extreme block

    def print_tree(self, level=0):
        """
        Recursively prints the decomposition tree structure with indentation.
        """
        indent = "    " * level  # Indentation for visualization
        if self.is_extreme_block:
            print(f"{indent}- Extreme Block (Leaf): {sorted(self.graph.keys())}")
        else:
            print(f"{indent}- Cutset {self.cutset}: Decomposing into {len(self.children)} blocks...")
            for child in self.children:
                child.print_tree(level + 1)

def construct_decomposition_tree(graph):
    """
    Constructs a decomposition tree using clique cutsets.
    Ensures that a clique cutset only proceeds if it creates a valid extreme block.
    If no such clique cutset exists, terminates the algorithm.
    """
    print("\nStarting decomposition for graph:", sorted(graph.keys()))

    tested_cutsets = set()  # Store tested cutsets to avoid infinite loops

    while True:
        cutset_result = find_clique_cutset(graph)

        if cutset_result["type"] is None:
            # No more clique cutsets exist, meaning the current graph is an extreme block
            leaf = DecompositionTree(graph)
            leaf.is_extreme_block = True
            print(f"Extreme Block (Leaf): {sorted(graph.keys())}\n")
            return leaf

        # Extract the clique cutset
        clique_cutset = tuple(sorted(cutset_result["cutset"]))  # Convert to tuple for set tracking
        cutset_type = cutset_result["type"]

        # Check if this cutset was already tested
        if clique_cutset in tested_cutsets:
            print(f"\n[DEBUG] Already tested cutset {clique_cutset}. Exiting due to looping...\n")
            return None  # Terminate if looping occurs

        tested_cutsets.add(clique_cutset)  # Mark this cutset as tested
        print(f"\n[{cutset_type}] Clique Cutset Found: {list(clique_cutset)}")

        # Remove the clique cutset and find connected components
        components = get_connected_components(graph, set(clique_cutset))

        valid_cutset_found = False
        extreme_block = None
        remaining_graph = None

        for component in components:
            subgraph_nodes = component.union(set(clique_cutset))  # Add cutset back to the component
            subgraph = {node: [nbr for nbr in graph[node] if nbr in subgraph_nodes] for node in subgraph_nodes}

            # Check if this subgraph contains a clique cutset
            if find_clique_cutset(subgraph)["type"] is None:
                if extreme_block is None:
                    extreme_block = subgraph
                else:
                    remaining_graph = subgraph
                valid_cutset_found = True
            else:
                if remaining_graph is None:
                    remaining_graph = subgraph
                else:
                    extreme_block = subgraph

        # If no valid decomposition is found, terminate the function
        if not valid_cutset_found or extreme_block is None or remaining_graph is None:
            print(f"\n[DEBUG] No valid decomposition found. Terminating.\n")
            return None  # Properly terminate instead of looping

        # Print decomposition blocks safely
        print(f"Blocks of Decomposition: {sorted(extreme_block.keys())} (Next Block), {sorted(remaining_graph.keys())} (Next Block)\n")

        # Create the decomposition tree node
        root = DecompositionTree(graph, list(clique_cutset))

        # Recursively continue decomposition with both blocks
        child_tree_1 = construct_decomposition_tree(extreme_block)
        child_tree_2 = construct_decomposition_tree(remaining_graph)

        if child_tree_1:
            root.children.append(child_tree_1)
        if child_tree_2:
            root.children.append(child_tree_2)

        return root  # Return tree after successful decomposition
