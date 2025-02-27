from collections import deque
from itertools import permutations
from algorithms.bipartite_check import is_bipartite_with_colouring
import networkx as nx
import heapq

def is_connected(graph):

    """
    Determines whether a graph is connected using a Breadth-First Search (BFS) approach.

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        bool: True if the graph is connected, False otherwise.

    Notes:
        - A graph is connected if every vertex is reachable from any other vertex.
        - If the graph has no nodes, it is trivially connected.
    """

    nodes = list(graph.keys())
    if not nodes:
        return True # An empty graph is trivially connected
    
    visited = set()
    queue = deque([nodes[0]]) # Start BFS from any node
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(neighbour for neighbour in graph[node] if neighbour not in visited)
    
    # If we visited all nodes, the graph is trivially connected
    return len(visited) == len(nodes)

def is_graph_disconnected(graph, removed_nodes):

    """
    Checks whether removing a set of nodes from the graph disconnects it.

    Parameters:
        graph (dict): An adjacency list representation of the graph.
        removed_nodes (set): A set of nodes to be removed from the graph.

    Returns:
        bool: True if removing the given nodes disconnects the graph; False otherwise.

    Notes:
        - A graph is disconnected if there exists at least one pair of nodes that can no longer reach each other after the removal.
        - If all nodes are removed, the graph is trivially disconnected.
    """

    # Find the first node that is NOT in the removed set
    for start in graph: 
        if start not in removed_nodes:
            break
    else:
        return True  # If all nodes are removed, graph is trivially disconnected
    
    # Performs BFS to check connectivity of remaining nodes
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(neighbour for neighbour in graph[node] if neighbour not in visited and neighbour not in removed_nodes)
    
    # If we didn't visit all non-removed nodes, the graph is disconnected
    return len(visited) < len(graph) - len(removed_nodes)

def contains_triangle(graph):

    """
    Determines whether the given graph contains a triangle (K₃).

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        bool: True if a triangle exists, False otherwise.

    Notes:
        - A triangle (K₃) exists if there is a node with two neighbours that are also connected.
        - This function iterates through each node and checks for triangular connections.
    """
    for node in graph:
        for neighbour1 in graph[node]:
            for neighbour2 in graph[node]:
                if neighbour1 != neighbour2 and neighbour2 in graph[neighbour1]:
                    return True # Found a triangle
    return False

def is_clique(graph, nodes):

    """
    Determines whether a given set of nodes forms a clique in the graph.

    A clique is a subset of vertices such that every pair of vertices in the set
    is adjacent.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
        nodes (list or set): A set of nodes to check for clique property.

    Returns:
        bool: True if the given nodes form a clique, False otherwise.

    Notes:
        - A valid clique must have `n-1` edges for each of its `n` nodes.
        - The function first checks whether each node has at least `n-1` neighbours.
        - Then, it verifies that all nodes in the given set are connected to each other.
    """

    nodes = set(nodes)  # Convert list to set for fast lookup

    for node in nodes:
        # A node in a clique should be connected to at least (n-1) other nodes in the clique
        if len(graph[node]) < len(nodes) - 1: # A clique requires full connectivity
            return False
        
        # Ensure all other nodes in the set are connected to the current node
        if not nodes.issubset(set(graph[node]) | {node}): 
            return False
        
    return True # All nodes satisfy the clique property

def find_clique_cutset(graph, tested_cutsets):

    """
    Identifies the smallest clique cutset (K₁ or K₂) that separates the graph into components.

    A clique cutset is a set of nodes that:
        1. Forms a clique (i.e., all nodes are fully connected).
        2. When removed, it disconnects the graph into multiple components.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
        tested_cutsets (set): A set of already tested cutsets to avoid redundancy.

    Returns:
        dict: A dictionary containing:
            - "type" (str or None): The type of the cutset, either "K1" (single node) or "K2" (two nodes).
            - "cutset" (list or None): The actual cutset nodes.
            - If no valid cutset is found, returns {"type": None, "cutset": None}.

    Notes:
        - The function first checks single-node cutsets (K1).
        - If no K1 cutsets are found, it then checks two-node cutsets (K2).
        - The function prioritises the smallest cutsets.
    """

    if not graph:
        return {"error": "Graph is empty. Algorithm terminated."}
    if not is_connected(graph):
        return {"error": "Graph is not connected. Algorithm terminated."}
        
    potential_cutsets = []

    # Step 1: Identify K1 cutsets
    for node in graph:
        if is_graph_disconnected(graph, {node}) and len(graph) > 1:
            if (node,) not in tested_cutsets:
                potential_cutsets.append({"type": "K1", "cutset": [node]})

    # Step 2: Identify K2 cutsets
    for node in graph:
        for neighbour in graph[node]:
            if node < neighbour: # Ensure each pair is checked only once
                if is_graph_disconnected(graph, {node, neighbour}):
                    if (node, neighbour) not in tested_cutsets:
                        potential_cutsets.append({"type": "K2", "cutset": [node, neighbour]})

    # Step 3: Choose the smallest cutset (K1 is prioritised over K2)
    if potential_cutsets:
        selected_cutset = min(potential_cutsets, key=lambda x: len(x["cutset"]))  
        return selected_cutset

    return {"type": None, "cutset": None} # No valid clique cutset found

def get_connected_components(graph, removed_nodes):

    """
    Computes the connected components of a graph after removing specified nodes.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
        removed_nodes (set): A set of nodes to be removed before computing components.

    Returns:
        list of sets: Each set represents a connected component after the node removal.

    Notes:
        - Uses Depth-First Search (DFS) to explore the graph.
        - Skips nodes that are in `removed_nodes`.
        - Returns a list of disjoint sets, where each set contains nodes forming a component.
    """

    visited = set()
    components = []

    def dfs(node, component):
        """
        Helper function to perform Depth First Search (DFS) and collect component nodes.
        """
        visited.add(node)
        component.append(node)
        for neighbour in graph[node]:
            if neighbour not in visited and neighbour not in removed_nodes:
                dfs(neighbour, component)

    # Traverse all nodes to identify separate components
    for node in graph:
        if node not in visited and node not in removed_nodes:
            component = []
            dfs(node, component)
            components.append(set(component)) # Store each component as a set

    return components

class DecompositionTree:

    """
    Represents a node in the decomposition tree used for colouring
    (triangle, theta)-free graphs.

    Each node in the decomposition tree represents a subgraph that results from 
    decomposing the original input graph using clique cutsets.

    Attributes:
        graph (dict): The adjacency list representation of the subgraph at this node.
        cutset (list or None): The clique cutset that led to the creation of this node.
        children (list): A list of child nodes representing further decomposed subgraphs.
        is_extreme_block (bool): True if this node represents an extreme block (a leaf in the decomposition tree).
        colouring (dict or None): Stores the final colouring of this subgraph.
    """

    def __init__(self, graph, cutset=None):

        """
        Initialises a decomposition tree node.

        Parameters:
            graph (dict): The subgraph represented by this node.
            cutset (list or None): The clique cutset that was removed to obtain this node.
        """

        self.graph = graph # The current subgraph at this node
        self.cutset = cutset # The clique cutset that led to this node
        self.children = [] # List of child nodes (subgraphs)
        self.is_extreme_block = False # Will be True if this is a leaf/extreme block
        self.colouring = None # Store final colouring here

    def print_tree(self, level=0):

        """
        Recursively prints the structure of the decomposition tree with indentation.

        Parameters:
            level (int): The current depth level in the decomposition tree (used for indentation).
        """

        indent = "    " * level # Indentation for visualization

        if self.is_extreme_block:
            print(f"{indent}- Extreme Block (Leaf): {sorted(self.graph.keys())}")
        else:
            print(f"{indent}- Cutset {self.cutset}: Decomposing into {len(self.children)} blocks...")

            # Identify the extreme and remaining subgraphs
            extreme_block = None
            remaining_block = None

            for child in self.children:
                if child.is_extreme_block:
                    extreme_block = child
                else:
                    remaining_block = child

            if remaining_block:
                print(f"{indent}    - Remaining Graph: {sorted(remaining_block.graph.keys())}")

            for child in self.children:
                child.print_tree(level + 1) # Recursively print child subtrees

def construct_decomposition_tree(graph):

    """
    Constructs the decomposition tree for a given graph.

    The decomposition follows the strategy from Radovanović and Vušković (2010) to
    recursively decompose the graph using clique cutsets until extreme blocks are reached.
    
    Parameters:
        graph (dict): The adjacency list representation of the graph.

    Returns:
        DecompositionTree or None:
            - If successful, returns the root of the decomposition tree.
            - If the graph does not belong to the (triangle, theta)-free class, returns None.

    Steps:
        1. If the graph contains a triangle, it does not belong to the class, so return None.
        2. Identify clique cutsets (K1 or K2) to decompose the graph into smaller components.
        3. Recursively apply decomposition on the connected components.
        4. If no further decomposition is possible, mark the node as an extreme block.
        5. Assign a valid colouring to the final decomposition tree.
    """

    if contains_triangle(graph):
        return None # The graph is not (triangle, theta)-free

    tested_cutsets = set() # Store tested cutsets to avoid infinite loops

    while True:
        cutset_result = find_clique_cutset(graph, tested_cutsets)

        if cutset_result["type"] is None:
            colouring = None 
            is_bipartite, _, bipartite_colouring = is_bipartite_with_colouring(graph) # If the leaf is a cube

            if is_bipartite:
                colouring = bipartite_colouring
            else:
                colouring = greedy_three_colouring(graph)

                if colouring is None:
                    return None # If the graph cannot be properly 3-coloured, return None

            # No more clique cutsets exist, create an extreme block
            leaf = DecompositionTree(graph)
            leaf.is_extreme_block = True
            leaf.colouring = colouring 
            return leaf

        # Extract the clique cutset
        clique_cutset = tuple(sorted(cutset_result["cutset"])) # Convert to tuple for set tracking

        if clique_cutset in tested_cutsets:
            continue # Try another clique cutset

        tested_cutsets.add(clique_cutset) # Mark this cutset as tested

        # Remove the clique cutset and find connected components
        components = get_connected_components(graph, set(clique_cutset))

        extreme_block = None
        merged_graph = {} # Store all non-extreme blocks to be merged later

        for component in components:
            subgraph_nodes = component.union(set(clique_cutset))  # Add cutset back to the component
            subgraph = {node: [nbr for nbr in graph[node] if nbr in subgraph_nodes] for node in subgraph_nodes}

            # Check if this subgraph is an extreme block
            if find_clique_cutset(subgraph, tested_cutsets)["type"] is None:
                if extreme_block is None:
                    extreme_block = subgraph # Found an extreme block
                else:
                    extreme_block = None
                    break # Discard this cutset and try another

            else:
                # Merge non-extreme blocks back together
                for node, neighbours in subgraph.items():
                    if node in merged_graph:
                        merged_graph[node].extend(neighbours)
                    else:
                        merged_graph[node] = neighbours

        if extreme_block is None:
            continue

        # Ensure the merged graph is properly structured
        remaining_graph = {node: list(set(neighbours)) for node, neighbours in merged_graph.items()}

        # Create the decomposition tree node
        root = DecompositionTree(graph, list(clique_cutset))

        # Recursively continue decomposition with the extreme block and the merged remaining graph
        child_tree_1 = construct_decomposition_tree(extreme_block)
        child_tree_2 = construct_decomposition_tree(remaining_graph)

        if child_tree_1:
            root.children.append(child_tree_1)
        if child_tree_2:
            root.children.append(child_tree_2)

        if root:
            colour_leaves(root) # Colour all leaves after decomposition
            propagate_colouring_upwards(root) # Ensure colours are consistent in the tree

        return root # Return tree after successful decomposition

def bfs_ordering(graph):

    """
    Generates a breadth-first search (BFS)-based vertex ordering that prioritises nodes with low degree.

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        list: A list of nodes representing a BFS traversal order.

    Notes:
        - The BFS starts from the node with the smallest degree to improve colouring efficiency.
        - Neighbours are sorted by degree (smallest first) to ensure an optimal ordering.
        - This ordering helps maintain a structured traversal for later use in greedy colouring.
    """

    start_node = min(graph, key=lambda x: len(graph[x]))  # Pick lowest-degree node
    queue = deque([start_node])
    visited = set()
    ordering = []

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        ordering.append(node)

        # Sort neighbours by degree (smallest degree first) to keep choices optimal
        neighbours = sorted(graph[node], key=lambda x: len(graph[x]))
        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append(neighbour)

    return ordering

def is_valid_ordering(ordering, graph):

    """
    Verifies whether a given vertex ordering satisfies the required constraints.

    Parameters:
        ordering (list): A list of nodes representing a vertex ordering.
        graph (dict): An adjacency list representation of the graph.

    Returns:
        bool: True if the ordering satisfies the constraints, False otherwise.

    Constraints:
        - All nodes in the graph must be included in the ordering.
        - Each node should have at most two earlier neighbours in the ordering (This ensures that the ordering follows a structure suitable for greedy colouring).
    """

    if set(ordering) != set(graph.keys()):  # Ensure all nodes are included
        return False  

    earlier_nodes = set()
    for node in ordering:
        earlier_neighbours = {nbr for nbr in graph[node] if nbr in earlier_nodes}
        if len(earlier_neighbours) > 2:
            return False # Invalid ordering, more than two earlier neighbours
        earlier_nodes.add(node) # Mark node as processed

    return True # This ordering is valid

def construct_valid_ordering(graph):

    """
    Constructs a valid vertex ordering for greedy colouring by prioritising low-degree nodes.

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        list or None:
            - A list of nodes in an ordering that ensures each node has at most two earlier neighbours.
            - None if no valid ordering exists.

    Notes:
        - Uses a greedy removal approach to construct the ordering.
        - Maintains a min-heap to always select the lowest-degree node first.
        - The goal is to maintain at most two remaining edges per node before removal.
    """

    temp_graph = {node: set(neighbours) for node, neighbours in graph.items()} # Copy adjacency list
    ordered_nodes = []
    min_heap = [(len(neighbours), node) for node, neighbours in temp_graph.items()]
    heapq.heapify(min_heap) # Min-heap ensures lowest-degree nodes are processed first

    while min_heap:
        while min_heap and len(temp_graph[min_heap[0][1]]) > 2:
            heapq.heappop(min_heap) # Skip nodes that don't meet the condition
        
        if not min_heap:
            return None # No valid ordering found
        
        _, node = heapq.heappop(min_heap)
        ordered_nodes.append(node)

        # Remove node and update neighbour degrees
        for neighbour in temp_graph[node]:
            temp_graph[neighbour].remove(node)

        del temp_graph[node] # Remove node from graph

        # Rebuild heap with updated degrees
        min_heap = [(len(temp_graph[nbr]), nbr) for nbr in temp_graph if len(temp_graph[nbr]) <= 2]
        heapq.heapify(min_heap)

    return ordered_nodes if len(ordered_nodes) == len(graph) else None # Ensure all nodes are included

def greedy_three_colouring(graph):

    """
    Attempts to find a valid 3-colouring of the graph using a structured greedy approach.

    Parameters:
        graph (dict): An adjacency list representation of the graph.

    Returns:
        dict or None:
            - A dictionary mapping nodes to one of three colours ("Colour 1", "Colour 2", "Colour 3").
            - None if no valid 3-colouring is found.

    Notes:
        - Uses a structured ordering obtained from `construct_valid_ordering(graph)`.
        - Assigns the first available colour to each node, ensuring no adjacent nodes share a colour.
        - If no valid ordering is found, returns `None`.
    """

    ordering = construct_valid_ordering(graph)
    if not ordering:
        return None  # No valid ordering found

    colours = {}
    available_colours = {'Colour 1', 'Colour 2', 'Colour 3'}

    for node in ordering:
        # Identify colours used by neighbouring nodes
        neighbour_colours = {colours[nbr] for nbr in graph[node] if nbr in colours}

        # Assign the first available colour
        for colour in available_colours:
            if colour not in neighbour_colours:
                colours[node] = colour
                break

    return colours if len(colours) == len(graph) else None  # Ensure all nodes are coloured

def colour_leaves(tree):

    """ 
    Assigns colours to the leaves of the decomposition tree.

    Parameters:
        tree (DecompositionTree): The root node of the decomposition tree.

    Behavior:
        - If a leaf (extreme block) is bipartite, it is 2-coloured.
        - If not bipartite, it is 3-coloured using a greedy approach.
        - If a leaf cannot be coloured properly, the function returns `None`, terminating the algorithm.
    """

    if tree.is_extreme_block:
        is_bipartite, _, colouring = is_bipartite_with_colouring(tree.graph)

        if is_bipartite:
            tree.colouring = colouring # 2-colouring for bipartite graphs
        else:
            try:
                tree.colouring = greedy_three_colouring(tree.graph) # Greedy 3-colouring
            except ValueError:
                tree.colouring = None # Ensure failure is recorded
                return None # Stop the algorithm if a leaf cannot be 3-coloured
    else:
        for child in tree.children:
            colour_leaves(child)

    if tree.colouring is None:
        return None

def propagate_colouring_upwards(tree):

    """
    Ensures consistent 3-colouring by aligning child colourings across clique cutsets.

    Parameters:
        tree (DecompositionTree): The root node of the decomposition tree.

    Process:
        - Starts at the lowest internal nodes and moves upwards.
        - Permutes child colourings to ensure cutset nodes have consistent colours.
        - Ensures that no two adjacent nodes share the same colour in the final colouring.

    Notes:
        - Uses all permutations of three colours to find the best match for consistency.
        - The function modifies the colouring bottom-up to avoid conflicts in the tree structure.
    """

    if tree.is_extreme_block:
        return  # Leaves are already correctly coloured

    # First, propagate colouring for all child nodes before processing this node
    for child in tree.children:
        propagate_colouring_upwards(child)

    if tree.cutset:
        reference_colouring = tree.children[0].colouring # Use the first child's colouring as reference

        for other_child in tree.children[1:]:
            original_colours = other_child.colouring.copy()

            # Find the best permutation that aligns the cutset colours
            best_colouring = None
            min_conflicts = float('inf')

            for perm in permutations(["Colour 1", "Colour 2", "Colour 3"]):
                perm_map = {"Colour 1": perm[0], "Colour 2": perm[1], "Colour 3": perm[2]}
                new_colouring = {node: perm_map[original_colours[node]] for node in original_colours}

                # Check if cutset nodes now match the reference
                conflicts = sum(1 for node in tree.cutset if node in reference_colouring and new_colouring[node] != reference_colouring[node])

                # Choose the permutation with the fewest conflicts
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_colouring = new_colouring

            # Apply the best colouring found to the child
            if best_colouring:
                other_child.colouring = best_colouring

    # Merge all child colourings into the parent
    tree.colouring = {}
    for child in tree.children:
        tree.colouring.update(child.colouring)

    # Ensure clique cutset nodes retain their correct reference colour
    for node in tree.cutset:
        if node in reference_colouring:
            tree.colouring[node] = reference_colouring[node]
