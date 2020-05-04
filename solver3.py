import networkx as nx
from networkx.algorithms import approximation
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import os


def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph + the algorithm used to produce it.
    """
    # Base Case: If no edges, we return the graph itself.
    if len(G.edges()) == 0:
        return G, None, 0

    #Algorithm Candidate 1: Cut ALL Leaf Nodes off MST.

    return alg(G)


def alg(G):
    set = nx.min_weighted_dominating_set(G)
    T = nx.Graph()
    T.add_nodes_from(set)
    

    
# To run: python3 solver.py inputs
if __name__ == '__main__':
    assert len(sys.argv) == 2
    arg_path = sys.argv[1]
    alg_quality = {}
    total = 0
    input_graphs = os.listdir(arg_path)
    total_dist = 0
    for graph_in in input_graphs:
        #print("---------------")
        #print("Calculating Minimal Tree for: " + graph_in)
        G = read_input_file(arg_path + '/' + graph_in)
        # foo()
        T, alg, avgdist = solve(G) #solve will return both the graph T and the optimal algorithm name. 
        if (alg in alg_quality):
            alg_quality[alg] += 1 
        else:
            alg_quality[alg] = 1
        assert is_valid_network(G, T)
        # print("Average pairwise distance: {}".format(average_pairwise_distance(T)))
        graph_out = 'outputs/' + graph_in[:len(graph_in) - 3] + '.out'
        write_output_file(T, graph_out)
        read_output_file(graph_out, G)
        total += 1
        total_dist += avgdist

    print("AVG OF AVGS (minimize this): " + str(total_dist / total))