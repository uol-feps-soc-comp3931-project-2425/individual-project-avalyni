import json
import os
import networkx as nx
import random

# Ensure the file is saved in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "sample_graphs.json")

def generate_cube_free_graph(offset=0):

    """
    Generates a small cube-free graph that is both triangle-free and theta-free.

    Parameters:
        offset (int): A number to shift node labels to avoid conflicts when merging graphs.

    Returns:
        dict: An adjacency list representation of the cube-free graph.
    """

    G = nx.cycle_graph(5) # A simple cycle graph with 5 nodes
    return {node + offset: [neighbour + offset for neighbour in G.neighbours(node)] for node in G.nodes()}

def attach_graphs(base_graph, new_graph, cutset_type="K2"):

    """
    Attaches a new graph to an existing base graph using a clique cutset (K1 or K2).

    This function mimics a reverse decomposition process, combining graphs into
    larger triangle-theta-free structures.

    Parameters:
        base_graph (dict): The adjacency list of the existing graph.
        new_graph (dict): The adjacency list of the new graph to be attached.
        cutset_type (str): Type of clique cutset to use ("K1" for a single vertex, "K2" for two vertices).

    Returns:
        dict: A new adjacency list representation of the combined graph.

    Steps:
        1. Convert node labels in `base_graph` and `new_graph` to integers.
        2. Shift `new_graph` node labels to avoid overlap with `base_graph`.
        3. Connect `new_graph` to `base_graph` using:
            - K1: A single randomly chosen node.
            - K2: A randomly chosen edge where both nodes have >1 connection.
        4. Return the updated adjacency list.
    """

    G = nx.Graph()

    # Convert node labels to integers for consistency
    base_graph = {int(node): [int(neighbour) for neighbour in neighbours] for node, neighbours in base_graph.items()}
    new_graph = {int(node): [int(neighbour) for neighbour in neighbours] for node, neighbours in new_graph.items()}
    
    # Add base graph nodes and edges
    G.add_nodes_from(base_graph.keys())
    for node, neighbours in base_graph.items():
        for neighbour in neighbours:
            G.add_edge(node, neighbour)
    
    # Shift 'new_graph' nodes to avoid conflicts with existing labels
    highest_node = max(G.nodes)
    new_graph = {node + highest_node + 1: [n + highest_node + 1 for n in neighbours] for node, neighbours in new_graph.items()}
    
    # Attach the new graph using a clique cutset
    if cutset_type == "K1":
        # Choose a random node from `base_graph` and connect it to a random node in `new_graph`
        cutset_node = random.choice(list(G.nodes))
        new_graph_node = random.choice(list(new_graph.keys()))
        G.add_edge(cutset_node, new_graph_node)
    else:
        # Choose a valid edge from 'base_graph' where both nodes have more than one neighbour
        possible_cutsets = [tuple(edge) for edge in G.edges if len(G[edge[0]]) > 1 and len(G[edge[1]]) > 1]
        if possible_cutsets:
            cutset = random.choice(possible_cutsets)
            new_graph_nodes = list(new_graph.keys())[:2] # Take two nodes from 'new_graph'
            G.add_edge(cutset[0], new_graph_nodes[0])
            G.add_edge(cutset[1], new_graph_nodes[1])
    
    # Add edges of 'new_graph' to 'G'
    for node, neighbours in new_graph.items():
        for neighbour in neighbours:
            G.add_edge(node, neighbour)
    
    return {node: sorted(neighbours) for node, neighbours in G.adjacency()}

def generate_graphs():

    """
    Generates both small and large triangle-theta-free graphs using reverse decomposition.

    This function creates:
        - 5 small graphs (1-3 attachment steps)
        - 5 large graphs (5-10 attachment steps)

    Each graph is stored in sample_graphs.json.

    Process:
        1. Start with a base cube-free graph.
        2. Attach multiple new cube-free graphs using random K1/K2 clique cutsets.
        3. Store the final adjacency lists in a JSON file.

    Returns:
        None (writes to file).
    """

    graph_data = {}
    
    # Generate 5 small triangle-theta-free graphs
    for i in range(5):
        base_graph = generate_cube_free_graph()
        num_steps = random.randint(1, 3) # Small graphs have 1-3 attachment steps
        for _ in range(num_steps):
            new_graph = generate_cube_free_graph()
            base_graph = attach_graphs(base_graph, new_graph, cutset_type=random.choice(["K1", "K2"]))
        graph_data[f"small_triangle_theta_free_{i+1}"] = base_graph
    
    # Generate 5 large triangle-theta-free graphs
    for i in range(5):
        base_graph = generate_cube_free_graph()
        num_steps = random.randint(5, 10) # Large graphs have 5-10 attachment steps
        for _ in range(num_steps):
            new_graph = generate_cube_free_graph()
            base_graph = attach_graphs(base_graph, new_graph, cutset_type=random.choice(["K1", "K2"]))
        graph_data[f"large_triangle_theta_free_{i+1}"] = base_graph
    
    # Save the generated graphs to a JSON file
    with open(output_file, "w") as f:
        json.dump(graph_data, f, indent=4)
    
    print(f"Triangle-Theta-Free graphs generated and saved to: {output_file}")

# Execute graph generation
generate_graphs()