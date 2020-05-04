import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance
import scipy as sp
import SimulatedAnnealing
import os
import sys
import random

#SIMULATEDA


def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!

    # Base Case: If no edges, we return the graph itself.
    if len(G.edges()) == 0:
        return G, None, 0
    
    distance_matrix = nx.to_numpy_array(G)
    print(distance_matrix)
    init_state = list(G.nodes)
    # for node in G.nodes:
    #   init_state.append(str(node))
    
    network = SimulatedAnnealing.SimulatedAnnealing(init_state, distance_matrix, G)
    network.set_schedule(network.auto(minutes=0.2))
    network.copy_strategy = "slice"
    path, cost = network.anneal()
    print(path)
    print(cost)
    T = nx.Graph()
    T.add_nodes_from(path)
    for first, second in zip(path, path[1:]):
      print(first)
      print(second)
      print(G.get_edge_data(first, second))
      T.add_edge(first, second, G[first][second]['weight'])
    return T

# Here's an example of how to run your solver.
# "/Users/narenyenuganti/Documents/GitHub/Cheapest-Network/inputs"
# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    arg_path = sys.argv[1]
    input_graphs = os.listdir(arg_path)
    for graph_in in input_graphs:
        G = read_input_file(arg_path + '/' + graph_in)

        T = solve(G)
        
        assert is_valid_network(G, T)
        graph_out = 'outputs/' + graph_in[:len(graph_in) - 3] + '.out'
        write_output_file(T, graph_out)
        read_output_file(graph_out, G)