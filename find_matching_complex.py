'''

Finds matching complex of a given graph.

VPython is used to graph this complex

'''


import numpy as np
import networkx as nx


edge_list = [(1,2), (2,3), (3,4), (1,4), (1,3), (2,4)]


G = nx.Graph()
G.add_edges_from(edge_list)

nx.drawing.nx_pylab.draw(G)

print("done!")
