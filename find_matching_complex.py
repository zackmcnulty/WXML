'''

Finds matching complex of a given graph.

VPython is used to graph this complex

'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import more_itertools as mitl


edge_list = [(1,2), (2,3), (3,4), (1,4), (1,3), (2,4)]
edge_labels = {e:i+1 for i,e in enumerate(edge_list)}
print(edge_labels)


G = nx.Graph()
G.add_edges_from(edge_list)

# draw the graph given by edge list
pos=nx.spring_layout(G)
nx.drawing.nx_pylab.draw(G, pos)
nx.drawing.nx_pylab.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

matchings = set([])

# find all possible matchings: improve this code if necessary
for edge_set in mitl.powerset(edge_list):
    if nx.algorithms.matching.is_matching(G, edge_set):
        matchings.add(tuple([edge_labels[e] for e in edge_set]))

print("matchings: ", matchings)

maximal_matchings = set() 
for match in matchings:
    to_remove = set()
    for max_match in maximal_matchings:
        if set(match).issubset(set(max_match)): break
        if set(max_match).issubset(set(match)): to_remove.add(max_match)
    else:
        maximal_matchings.add(match)

    maximal_matchings = maximal_matchings - to_remove
        
print("maximal matchings: ", maximal_matchings)

max_dim = max([len(m) for m in maximal_matchings])

if max_dim > 3:
    print("Cannot draw matching complex. Dimension is > 3")
elif max_dim == 1 or max_dim == 0:
    print("Matching complex is trivial.")
elif max_dim == 2:
    draw_2D_matching_complex(maximal_matchings)
else: # dim == 3
    print("thats hard")


def draw_2D_matching_complex(max_matchings):
    #TODO: implement


print("done!")
