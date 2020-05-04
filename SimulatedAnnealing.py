from __future__ import print_function
import math
import random
from simanneal import Annealer
from utils import average_pairwise_distance
import networkx as nx

class SimulatedAnnealing(Annealer):
  # Test annealer with a travelling salesman problem.

  # pass extra data (the distance matrix) into the constructor
  def __init__(self, state, distance_matrix, graph):
      self.distance_matrix = distance_matrix
      self.graph = graph
      super(SimulatedAnnealing, self).__init__(state)  # important!

  def move(self):
      """Swaps two cities in the route."""
      # no efficiency gain, just proof of concept
      # demonstrates returning the delta energy (optional)
      initial_energy = self.energy()

      a = random.randint(0, len(self.state) - 1)
      b = random.randint(0, len(self.state) - 1)
      self.state[a], self.state[b] = self.state[b], self.state[a]

      return self.energy() - initial_energy

  def energy(self):
      """Calculates the length of the route."""
      e = 0
      for i in range(len(self.state)):
          e += self.distance_matrix[self.state[i-1]][self.state[i]]
      return e
    #   return average_pairwise_distance(self.graph)
