# Cheapest Network Algorithm 

Algorithm finds the minimum dominating set of a graph G that minimizes average pairwise distance.

Formally, let G = (V,E) be a positive weighted, connected, undirected graph. We would like to find a subgraph T of
G such that:
1. Every vertex v ∈ V is either in T or adjacent to a vertex in T.
2. T is a tree.
3. The average pairwise distance between all vertices in T is minimized.


TO run our solver, simply run python solver.py inputs. Several lines in the code can be uncommented to display certain metadata such as best algorithm per graph, graph costs per algorithm run, etc. 

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs
- `phase1.py`: creates the 25, 50, 100 vertex inputs
