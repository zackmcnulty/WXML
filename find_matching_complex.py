'''

Finds matching complex of a given graph.

VPython is used to graph this complex

'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import more_itertools as mitl

def draw_2D_matching_complex(M_G, fill=[]):
    #TODO: implement
    pos = nx.spring_layout(M_G, dim=2)
#    pos = nx.planar_layout(M_G, dim=2)


    nx.drawing.nx_pylab.draw(M_G, pos)
    nx.drawing.nx_pylab.draw_networkx_labels(M_G, pos)

    ax = plt.gca()
    for V in fill:
        x = [pos[v][0] for v in V]
        y = [pos[v][1] for v in V]
        ax.fill(x,y, "blue")

    nx.drawing.nx_pylab.draw(M_G, pos)
    nx.drawing.nx_pylab.draw_networkx_labels(M_G, pos)

    plt.show()
    
# Define graph as edge list  ================================================================
#edge_list = [(1,2), (2,3), (3,4), (1,4), (1,3), (2,4)] # K_4
edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)] # C_6
# ==============================================================

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
M_G = nx.Graph()

if max_dim > 4:
    print("Cannot draw matching complex. Simplicial complex dimension is > 3")
elif max_dim == 1 or max_dim == 0:
    print("Matching complex is trivial.")
elif max_dim == 2:
    M_G.add_edges_from(maximal_matchings)
    draw_2D_matching_complex(M_G)

elif max_dim == 3:
    # NOTE: Larger weight = tries to force edge to be shorter
    tri_weight = 100 # how strong connection is between nodes in triangle
    other_weight = 50 # how strong other node connections are (for spring model)
    fill = []
    for match in maximal_matchings:
        if len(match) == 3:
            M_G.add_edge(match[0], match[1], weight=tri_weight)
            M_G.add_edge(match[0], match[2], weight=tri_weight)
            M_G.add_edge(match[2], match[1], weight=tri_weight)
            fill.append(match) 
        else:
            M_G.add_edge(match[0], match[1], weight=other_weight)
    print(fill)
    draw_2D_matching_complex(M_G,fill)

else: # dim == 3
    print("thats hard")



print("done!")
