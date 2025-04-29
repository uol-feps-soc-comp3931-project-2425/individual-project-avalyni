# (∆, Theta)-Free Graph Recognition

This repository implements a recognition algorithm to determine whether an input graph is free of both triangles (K₃) and theta subgraphs.

The implementation is based on the "three-in-a-tree" framework, adapted for use with pruned subgraphs from the neighbourhood of a node. This forms part of a final-year dissertation in Computer Science.

---

## 📦 Requirements

- Python 3.10 or newer
- NetworkX library (for graph operations)

# How to Run

python main.py

Each graph will be tested and labelled either as:
- (∆, theta)-free
- Not (∆, theta)-free

# Project Structure

recognition/
├── main.py                 # Loads graphs and executes the recognition check
├── recognition.py          # Triangle check and theta-finding logic
├── three_in_a_tree.py      # Modified algorithm for induced tree detection
└── tests/
    └── evaluation_test_graphs.json  # Input graphs 

# Notes

- Input graphs must be undirected and in JSON adjacency list format.

Developed by Ellie GReen as part of a University of Leeds dissertation.

Uses NetworkX (BSD License)

