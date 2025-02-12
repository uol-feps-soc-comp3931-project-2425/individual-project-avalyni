import json
from algorithms.bipartite_check import is_bipartite_with_colouring
from algorithms.decomposition import find_clique_cutset, construct_decomposition_tree

def main():
    with open('data/sample_graphs.json', 'r') as f:
        data = json.load(f)

    for graph_name, adj_list in data.items():
        graph = {int(k): v for k, v in adj_list.items()}
        print(f"\n#################### Testing {graph_name}... ####################")

        # Check bipartiteness
        is_bipartite, partition, colouring = is_bipartite_with_colouring(graph)
        if is_bipartite:
            print("The graph is bipartite!")
            print("Partition:", partition)
            print("Colouring:", colouring)
            print("\nNo decomposition needed for bipartite graphs.")
            continue  # Skip decomposition

        print("The graph is NOT bipartite.")

        # Construct and print decomposition tree
        print("\nDecomposition Tree Structure:")
        decomposition_tree = construct_decomposition_tree(graph)

        if decomposition_tree is None:
            print("\nThis graph is NOT in our class. Decomposition terminated.\n")
        else:
            decomposition_tree.print_tree()

if __name__ == "__main__":
    main()

