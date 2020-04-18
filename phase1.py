from parse import write_input_file, validate_file
import networkx as nx
import random

if __name__ == '__main__':
  G_small = nx.complete_graph(24)
  for (u, v) in G_small.edges():
    G_small.edges[u,v]['weight'] = round(random.uniform(0.0, 100.0), 3)
  
  G_medium = nx.complete_graph(48)
  for (u, v) in G_medium.edges():
    G_medium.edges[u,v]['weight'] = round(random.uniform(0.0, 100.0), 3)
  
  G_large = nx.complete_graph(99)
  for (u, v) in G_large.edges():
    G_large.edges[u,v]['weight'] = round(random.uniform(0.0, 100.0), 3)
  
  write_input_file(G_small, "/Users/narenyenuganti/Desktop/170Proj/25.in")
  write_input_file(G_medium, "/Users/narenyenuganti/Desktop/170Proj/50.in")
  write_input_file(G_large, "/Users/narenyenuganti/Desktop/170Proj/100.in") 

  if (validate_file("/Users/narenyenuganti/Desktop/170Proj/25.in") == False): 
    raise Exception("incorrect format for 25.in")
  if (validate_file("/Users/narenyenuganti/Desktop/170Proj/50.in") == False): 
    raise Exception("incorrect format for 50.in")
  if (validate_file("/Users/narenyenuganti/Desktop/170Proj/100.in") == False): 
    raise Exception("incorrect format for 100.in")