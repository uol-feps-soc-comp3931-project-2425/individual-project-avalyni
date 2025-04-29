# Colouring Algorithm for (∆, Theta)-Free Graphs

This repository implements a decomposition-based 3-colouring algorithm for graphs that are free of both triangles (K₃) and theta subgraphs. It applies recursive clique cutset decomposition to reduce graphs into colourable components and merges the results into a global colouring.

This tool was developed as part of a final-year Computer Science dissertation.

---

## 📦 Requirements

- Python 3.10 or newer
- NetworkX (for graph utilities)

# How to run

python main.py

This will:
- Load graphs from data/sample_graphs.json
- For each graph:
    - Check if it is bipartite,
    - If not, apply clique cutset decomposition and 3-colour the components,
    - Print and log the final colouring or rejection if the graph is not (∆, theta)-free.

Output is saved to decomposition_output.log

# Project Structure

├── main.py                  # Entry point
├── decomposition.py         # Core decomposition + colouring logic
├── bipartite_check.py       # BFS-based bipartiteness checker
├── data/
│   └── sample_graphs.json   # Sample graphs (editable)
└── decomposition_output.log # Logs from last run

# Notes

- Input graphs must be undirected and stored in JSON adjacency list format.

Created by Ellie Green for academic assessment.
Uses NetworkX (BSD Liscense)