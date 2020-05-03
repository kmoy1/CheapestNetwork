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
        return G, None
    #Algorithm Candidate 1: Cut ALL Leaf Nodes off MST. 
    pruned_MST1 = alg1(G)
    #Algorithm Candidate 2: Cut ALL Leaf Nodes off MST by edge weight consideration.
    pruned_MST2 = alg2(G)
    # Algorithm Candidate 3: Dominating Set
    DS_MST = alg3(G)


    # Print out results. 

    if DS_MST:
        dsMST = ["Dominating Set MST", average_pairwise_distance_fast(DS_MST), DS_MST]
    else:
        dsMST = ['none', float('inf'), None]

    mst1 = ["Pruned MST 1", average_pairwise_distance_fast(pruned_MST1), pruned_MST1]
    mst2 = ["Pruned MST 2", average_pairwise_distance_fast(pruned_MST2), pruned_MST2]
 
    best_algorithm = min([dsMST, mst1, mst2], key=lambda x: x[1])
    best_alg_name = best_algorithm[0]
    best_alg_avg_dist = best_algorithm[1]
    best_alg_avg_subgraph = best_algorithm[2]
    #print(best_alg_name + 'optimal')
    #Return optimal subgraph + name of producing algorithm 
    return best_alg_avg_subgraph, best_alg_name


def alg1(G):
    """Apply alg 2: MST algorithm, and return a subgraph T"""
    MST1 = nx.minimum_spanning_tree(G) # Create MST from graph. 
    # prune MST's leaf nodes.
    finishedPruningAllTrees = False
    while not finishedPruningAllTrees:
        MST1, finishedPruningAllTrees = pruneMST(MST1, G)
    return MST1

def alg2(G):
    """Apply alg 2: MST algorithm, and return a subgraph T"""
    MST2 = nx.minimum_spanning_tree(G)
    # prune MST's leaf nodes 
    finishedPruningAllTrees = False
    while not finishedPruningAllTrees:
        MST2, finishedPruningAllTrees = pruneMST2(MST2, G)
    return MST2

def alg3(G):
    """Apply alg 3: dominant set algorithm. Return a subgraph T."""
    WeightNodes2(G)
    # calculate a dominating set of G.
    DSet = list(nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, weight='node_weight'))

    # Find shortest paths from G.
    shortest_paths = nx.algorithms.shortest_paths.generic.shortest_path(G)
    #Connect dominating set into tree with minimal total weight. 
    dom_MST = dominatingMST(DSet, shortest_paths, G)
    return dom_MST



def foo():
    print("Test Globality of G")
    print(G)

def pruneMST(MST, G):
    """Handles removing appropriate leaves from the given minimum spanning tree."""
    #Collect all leaves of MST.
    leaves = getLeaves(MST)
    # order leaves by weight: We will remove the largest-weighted leaves first.
    sortByWeight = lambda x: list(MST.edges(x, data='weight'))[0][2]
    leaves.sort(reverse=True, key=sortByWeight)

    # PRUNE leaves from MST.
    num_times_pruned = 0 #
    num_removed_leaves = 1
    finishedPruningTree = False
    # num_times_looped = 0
    #We exit out this loop when the pruning method doesn't do anything to the MST (removes no leaves).
    #This means that if this loop runs ONCE, we know that we've found our pruned_MST.
    while num_removed_leaves > 0:
        # print(MST.edges())
        # print("Leaves: " + str(leaves))
        MST, leaves, num_removed_leaves = removeLeaves1(leaves, MST, G)
        # print("Finished prune_leaves call.")
        num_times_pruned += 1
    if num_times_pruned == 1:
        finishedPruningTree = True
    return MST, finishedPruningTree

# Handles removing appropriate leaves from
# the given minimum spanning tree.
def pruneMST2(MST, G):
    """Begin pruning leaf process considering decreasing of edge weights."""
    leaves = getLeaves(MST)
    finishedPruningTree = False
    num_times_pruned = 0
    num_removed_leaves = 1
    while num_removed_leaves > 0:
        MST, leaves, num_removed_leaves = removeLeaves2(leaves, MST, G)
        num_times_pruned += 1
    if num_times_pruned == 1:
        finishedPruningTree = True
    return MST, finishedPruningTree


def dominatingMST(DSet, shortest_paths, G):
    """Return an MST based on dominating set DSET and a list of shortest paths. """
    
    # build a new graph based on a dominating set DSET.
    DSetAsGraph = convertToGraph(DSet, shortest_paths, G)
    DMST = nx.minimum_spanning_tree(DSetAsGraph, 'weight')
    return DMST

def convertToGraph(DSet, shortest_paths, G):
    """Convert DSet into a (minimal) subgraph 
    with the shortest_paths as reference."""
    DSetGraph = nx.Graph()
    V = len(DSet)
    #For all unique (u,v) in our dominating set, find the SHORTEST PATH
    #between them in G, and add those path edges to our subgraph.
    for i in range(V):
        u = DSet[i] #Vertex in dominating set
        for j in range(i+1, V):
            if (i != j):
                v = DSet[j]
                path = shortest_paths[u][v]
                edges = []
                for k in range(len(path) - 1):
                    u_path = path[k]
                    v_path = path[k+1]
                    if DSetGraph.has_edge(u_path, v_path): #BEAUTIFUL. DSetGraph.edges
                        continue
                    #Create edge in tuple format and add it to list. 
                    w = G.edges[u_path, v_path]['weight']
                    #Add edge to edges.
                    edges.append((u_path, v_path, w))
                if (edges):
                    DSetGraph.add_weighted_edges_from(edges)
    return DSetGraph

def WeightNodes1(G):
    """Weight edges of vertices in G proportional to (-)degree.
    We do this because we want our dominating set to have vertices
    with as high degree as possible such that edges can
    be minimized in the MST."""
    for v in G.nodes:
        G.nodes[v]['node_weight'] = -1 * G.degree[v]

def WeightNodes2(G):
    """Weight edges of vertices in G 
    proportional to AVERAGE adjacent edge weight"""
    for v in G.nodes:
        G.nodes[v]['node_weight'] = AAEW(G, v)

def AAEW(G, v):
    """REturn average adjacent edge weight for vertex v in graph G"""
    # print("HERE")
    total = 0
    num_adj = len(list(G.edges(v)))     
    #TODO: How the fuck do I see the edge weights bih
    for e in list(G.edges(v, data='weight')):
        total += e[2]
    return total / num_adj  

def getLeaves(T):
    """Assuming T = tree, return list of leaves."""
    leaves = []
    for v in T.nodes:
        if T.degree[v] == 1:
            leaves.append(v)
    return leaves


def removeLeaves1(leaves, MST, G):
    """Remove leaves in LEAVES from an MST until MST is no 
    longer a dominating set of G, and return MST."""

    num_removed_leaves = 0 #Number of removed leaves from MST.
    kept_leaves = [] #Leaf edges we NEED to maintain dominating set.

    for leaf in leaves:
        if (leaf):
            leaf_edge = list(MST.edges(leaf, data='weight'))[0] #Get the only edge connected to this vertex. 
            #We need this if we find our graph violates Dominating set after removal of the vertex.
            #Remove the leaf from MST.
            MST.remove_node(leaf)

            #Check if removing the leaf would cause the DOMINATING SET requirement 
            #(all vertices in subgraph adjacent to all vertices in original G).
            # If it is, then add the leaf + edge back. 
            if (nx.is_dominating_set(G, MST.nodes) == False):
                MST.add_node(leaf)
                MST.add_edge(leaf_edge[0], leaf_edge[1], weight=leaf_edge[2])
                kept_leaves.append(leaf)
            else:
                num_removed_leaves += 1
    return MST, kept_leaves, num_removed_leaves


def removeLeaves2(leaves, MST, G):
    """Remove leaves in LEAVES from an MST until MST is no 
    longer a dominating set of G AND the average pairwise 
    distance is decreased."""
    num_removed_leaves = 0
    kept_leaves = []
    for leaf in leaves:
        if (leaf):
            e = list(MST.edges(leaf, data='weight'))[0]
            cost = average_pairwise_distance_fast(MST)
            #Test removing an edge and finding new avg pairwise dist. 
            MST.remove_node(leaf)
            new_cost = average_pairwise_distance_fast(MST)
            if (new_cost > cost or not nx.is_dominating_set(G, MST.nodes)):
                #Put the leaf node and edge back. 
                MST.add_node(leaf)
                MST.add_edge(e[0], e[1], weight=e[2])
                #print("Leaf kept")
                kept_leaves.append(leaf)
            else:
                num_removed_leaves += 1

    return MST, kept_leaves, num_removed_leaves

def print_best_algorithm(alg_quality, total):
    """Print best algorithm and score """
    optimal_algorithm = max(alg_quality, key=alg_quality.get)
    alg_score = (float) (alg_quality[optimal_algorithm] * 100) / total
    print("Best Alg: " + str(optimal_algorithm))
    print("Score: "  + str(alg_score) + "%")

# To run: python3 solver.py inputs
if __name__ == '__main__':
    assert len(sys.argv) == 2
    arg_path = sys.argv[1]
    alg_quality = {}
    total = 0
    input_graphs = os.listdir(arg_path)
    for graph_in in input_graphs:
        #print("---------------")
        #print("Calculating Minimal Tree for: " + graph_in)
        G = read_input_file(arg_path + '/' + graph_in)
        # foo()
        T, alg = solve(G) #solve will return both the graph T and the optimal algorithm name. 
        if (alg in alg_quality):
            alg_quality[alg] += 1 
        else:
            alg_quality[alg] = 1
        assert is_valid_network(G, T)
        print("Average pairwise distance: {}".format(average_pairwise_distance(T)))
        out = 'outputs/' + graph_in[:len(graph_in) - 3] + '.out'
        write_output_file(T, out)
        read_output_file(out, G)
        total += 1
    #Best method to solve is the one that produced the 
    print_best_algorithm(alg_quality, total)