import json
import sys
from algorithms.bipartite_check import is_bipartite_with_colouring
from algorithms.decomposition import construct_decomposition_tree

log_filename = "decomposition_output.log" 
log_file = open(log_filename, "w", encoding="utf-8")

class Tee:

    """
    Custom class to redirect output to both terminal and a log file.

    """

    def __init__(self, file):
        self.file = file

    def write(self, text):

        """
        Writes text to both standard output and the log file.
        """

        sys.__stdout__.write(text)  # Print to terminal
        self.file.write(text)  # Write to file

    def flush(self):

        """
        Ensures buffered output is written immediately.
        """

        sys.__stdout__.flush()
        self.file.flush()

sys.stdout = Tee(log_file)  # Redirect all print statements to the Tee class

def main():

    """
    Main function that loads graphs, performs bipartiteness check, and applies the decomposition.

    Steps:
    1. Read graph data from a JSON file.
    2. Iterate through each graph and determine if it is bipartite.
    3. If bipartite, output the partition and skip decomposition.
    4. If not bipartite, perform decomposition and check the final colouring.
    5. Log results and highlight any conflicts in colouring.
    """

    # Load graph data from JSON file
    with open('data/triangle_theta_free_graphs.json', 'r') as f:
        data = json.load(f) 

    # Iterate through graphs and process each
    for graph_name, adj_list in data.items(): 
        graph = {int(k): v for k, v in adj_list.items()} # Convert keys to integers

        # Print header for clarity in logs
        print(f"\n{'═' * 60}")
        print(f"STARTING TEST: {graph_name}".center(60))
        print(f"{'═' * 60}\n")

        # Display graph structure in adjacency list format
        print("\nGraph adjacency list:")
        for node, neighbours in sorted(graph.items()):
            print(f"  {node}: {sorted(neighbours)}")

        # Step 1: Check bipartiteness
        is_bipartite, partition, colouring = is_bipartite_with_colouring(graph)
        if is_bipartite:
            print("\nThe graph is bipartite!")
            print("Partition:", partition)
            print("Colouring:", colouring)
            print("\nNo decomposition needed for bipartite graphs.")
            continue  # Skip decomposition since bipartite graphs can be 2-coloured

        print("\nThe graph is not bipartite, decomposition will begin now.")

        # Step 2: Construct and print decomposition tree
        decomposition_tree = construct_decomposition_tree(graph)

        if decomposition_tree is None:
            print("\nThis graph is not in our class.") # If decomposition_tree returns None, the graph does not meet the requirements of our class
        else:
            # Print decomposition tree structure 
            decomposition_tree.print_tree()

            if decomposition_tree.colouring is None:
                print("\nThis graph is not in our class.")
                continue  # Skip further processing

            print("Here is the final colouring of the decomposition tree")
            print(decomposition_tree.colouring)

            # Step 3: Verify correctness of colouring
            adjacency_violations = sum(
                1 for node in graph
                for neighbour in graph[node]
                if neighbour in decomposition_tree.colouring and decomposition_tree.colouring[node] == decomposition_tree.colouring[neighbour]
            )

            # If conflicts exist, they are highlighted
            if adjacency_violations > 0:
                print(f"\n{adjacency_violations} conflicts found in the final colouring of {graph_name}!")
            else:
                print("\nNo conflicts found in the final colouring!")

    # Notify the user where the output has been saved
    print(f"\nOutput saved to: {log_filename}\n")

if __name__ == "__main__":
    main()

sys.stdout = sys.__stdout__  # Reset print output back to normal
log_file.close()  # Close the log file

