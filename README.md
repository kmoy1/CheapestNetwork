# CS 170 Project Spring 2020

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs
- `phase1.py`: creates the 25, 50, 100 vertex inputs

You are an engineer at Horizon Wireless, a telecommunications company, and the CEO has tasked you with designing
a cell tower network which spans the major cities of the United States. You need to decide which cities to build cell
towers in and lay out the fiber network between cities. There are a few factors to consider:

1. You would like each major city in the United States to have access to your network. This means that for each
city, there must be a cell tower either in that city or in at least one neighboring city.
2. You need to connect the cell towers with fiber cables. Due to government regulations, each city that your fiber
network passes through must have a cell tower. The CEO wants to save money, so he tells you that your fiber
network must be a tree.
3. When transmitting a signal between two cell towers, the signal must be sent through the fiber cables. Each cell
tower expects to communicate with every other tower efficiently. When designing the network, you want to
minimize the average pairwise distance between cell towers along the fiber cables.

Find the placement of the cell towers satisfying conditions (1) and (2) which minimizes the average pairwise distance
between towers.
Formally, let G = (V,E) be a positive weighted, connected, undirected graph. We would like to find a subgraph T of
G such that:
1. Every vertex v ∈ V is either in T or adjacent to a vertex in T.
2. T is a tree.
3. The average pairwise distance between all vertices in T is minimized.


TO run our solver, simply run python solver.py inputs. Several lines in the code can be uncommented to display certain metadata such as best algorithm per graph, graph costs per algorithm run, etc. 
