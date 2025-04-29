import json, time
from algorithms.recognition import find_theta
from algorithms.three_in_a_tree import three_in_a_tree

# Load graphs from a JSON file
with open("tests/evaluation_test_graphs.json") as f:
    graphs = json.load(f)

# Process each graph
for name, raw_graph in graphs.items():
    # Convert graph keys from string to integer
    graph = {int(k): v for k, v in raw_graph.items()}

    # Measure recognition time
    start= time.time()
    has_theta, _ = find_theta(graph, three_in_a_tree)
    end = time.time()

    # Report result
    print(f"Graph: {name}")
    print("Execution time: {:.6f} seconds".format(end - start))
    if has_theta:
        print("Graph is not (Δ, theta)-free.")
    else:
        print("Graph is (Δ, theta)-free.")
