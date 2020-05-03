import networkx as nx
from networkx.algorithms import approximation
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import os
import time


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
    dom_mst = alg3(G)


    # Print out results. 

    dom = ['none', float('inf'), None]
    if dom_mst:
        cost1 = average_pairwise_distance_fast(dom_mst)
        dom = ["Dominating Set", cost1, dom_mst]
        # print("Dominating Set Cost: " + str(cost1))

    cost2 = average_pairwise_distance_fast(pruned_MST1)
    mst = ["Pruned MST 1", cost2, pruned_MST1]
    # print("Pruned MST 1 Cost: " + str(cost2))

    cost3 = average_pairwise_distance_fast(pruned_MST2)
    cost_mst = ["Pruned MST 2", cost3, pruned_MST2]
    # print("Pruned MST 2 Cost: " + str(cost3))
 
    best = min([dom, mst, cost_mst], key=lambda x: x[1])
    print(best[0] + ' is optimal')
    #Return best graph and its cost. x
    return best[2], best[0]


def alg1(G):
    MST1 = nx.minimum_spanning_tree(G) # Create MST from graph. 
    # prune MST's leaf nodes.
    finishedPruningAllTrees = False
    while not finishedPruningAllTrees:
        # print("Iters Before remove_leaves called: " + str(iters))
        # print("Calling remove_leaves. Iters: " + str(iters))
        MST1, finishedPruningAllTrees = pruneMST(MST1, G)
    return MST1

def alg2(G):
    """Apply alg 2: MST algorithm. """
    MST2 = nx.minimum_spanning_tree(G)
    # prune MST's leaf nodes 
    finishedPruningAllTrees = False
    while not finishedPruningAllTrees:
        MST2, finishedPruningAllTrees = pruneMST2(MST2, G)
    return MST2

def alg3(G):
    """Apply alg 3: dominant set algorithm. """
    degreeWeightNodes(G)
    # calculate a dominating set of G.
    DSet = list(nx.algorithms.approximation.dominating_set.min_weighted_dominating_set(G, weight='wt'))

    # Find shortest paths from G.
    shortest_paths = nx.algorithms.shortest_paths.generic.shortest_path(G)
    #Connect dominating set into tree with minimal total weight. 
    dom_MST = dominatingMST(DSet, shortest_paths, G)
    return dom_MST



def foo():
    print("Test Globality of G")
    print(G)

def dominatingMST(DSet, shortest_paths, G):
    """Return an MST based on dominating set DSET and a list of shortest paths. """
    # build a new graph based on a dominating set DSET.
    DSetAsGraph = convertToGraph(DSet, shortest_paths)
    DMST = nx.minimum_spanning_tree(DSetAsGraph, 'weight')
    return DMST

def convertToGraph(DSet, shortest_paths):
    """Convert DSet into a (minimal) subgraph 
    with the shortest_paths as reference."""
    DSetGraph = nx.Graph()
    V = len(DSet)


    #For all unique (u,v), 
    for i in range(V):
        u = DSet[i] #Vertex in dominating set
        for j in range(i+1, V):
            if (i != j):
                v = DSet[j]
                path = shortest_paths[v][u]
                edges = []
                for k in range(len(path) - 1):
                    u = path[k]
                    v = path[k+1]
                    try: #If edge (v,u) already in the graph, then continue.
                        p = DSetGraph.edges[u, v]
                    except:
                        #Create edge in tuple format and add it to list. 
                        w = G.edges[u, v]['weight']
                        #Add edge to edges.
                        edges.append((u, v, w))
                if (edges):
                    DSetGraph.add_weighted_edges_from(edges)
    return DSetGraph

def degreeWeightNodes(G):
    """Weight edges of vertices in G proportional to (-)degree."""
    for v in G.nodes:
        G.nodes[v]['wt'] = -1 * G.degree[v]

def pruneMST(MST, G):
    """Handles removing appropriate leaves from the given minimum spanning tree."""
    #Collect all leaves of MST.
    leaves = getLeaves(MST)
    # order leaves by weight: We will remove the largest-weighted leaves first.
    sortByWeight = lambda x: list(MST.edges(x, data='weight'))[0][2]
    leaves.sort(reverse=True, key=sortByWeight)

    # PRUNE leaves from MST.
    iters = -1
    num_times_pruned = 0 #
    removed = 1
    finishedPruningTree = False
    # num_times_looped = 0
    #We exit out this loop when the pruning method doesn't do anything to the MST (removes no leaves).
    #This means that if this loop runs ONCE, we know that we've found our pruned_MST.
    while removed > 0:
        # print(MST.edges())
        # print("Leaves: " + str(leaves))
        MST, leaves, removed = removeLeaves1(leaves, MST, G)
        # print("Finished prune_leaves call.")
        num_times_pruned += 1
        # print("Iters value after call: " + str(iters))
        # print("Leaves removed: " + str(removed))
        # num_times_looped += 1
    if num_times_pruned == 1:
        finishedPruningTree = True
    # print("Num times looped: " + str(num_times_looped))
    return MST, finishedPruningTree

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
            e = list(MST.edges(leaf, data='weight'))[0]
            #Remove the leaf from MST.
            MST.remove_node(leaf)

            #Check if removing the leaf would cause the DOMINATING SET requirement 
            #(all vertices in subgraph adjacent to all vertices in original G).
            # If it is, then add the leaf + edge back. 
            if (nx.is_dominating_set(G, MST.nodes) == False):
                MST.add_node(leaf)
                MST.add_edge(e[0], e[1], weight=e[2])
                kept_leaves.append(leaf)
            else:
                num_removed_leaves += 1
    return MST, kept_leaves, num_removed_leaves

# Handles removing appropriate leaves from
# the given minimum spanning tree.
def pruneMST2(MST, G):
    """Begin pruning leaf process considering decreasing of edge weights."""
    leaves = getLeaves(MST)
    iters = -1
    finishedPruningTree = False
    num_times_pruned = 0
    removed = 1
    while removed > 0:
        min_tree, leaves, removed = removeLeaves2(leaves, MST, G)
        iters += 1
        num_times_pruned += 1
    if num_times_pruned == 1:
        finishedPruningTree = True
    return min_tree, finishedPruningTree


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

# To run: python3 solver.py inputs

if __name__ == '__main__':
    assert len(sys.argv) == 2
    arg_path = sys.argv[1]
    alg_quality = {}
    total = 0
    input_graphs = os.listdir(arg_path)
    for graph_in in input_graphs:
        print("---------------")
        print("Calculating Minimal Tree for: " + graph_in)
        G = read_input_file(arg_path + '/' + graph_in)
        # foo()
        T, alg = solve(G)
        if (alg in alg_quality):
            alg_quality[alg] += 1 
        else:
            alg_quality[alg] = 1
        assert is_valid_network(G, T)
        print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        out = 'outputs/' + graph_in[:len(graph_in) - 3] + '.out'
        write_output_file(T, out)
        read_output_file(out, G)
        total += 1
    #Best method to solve is the one that produced the 
    optimal_algorithm = max(alg_quality, key=alg_quality.get)
    alg_score = (float) (alg_quality[optimal_algorithm] * 100) / total
    print("Best Method: " + str(optimal_algorithm))
    print("Score: "  + str(alg_score) + "%")