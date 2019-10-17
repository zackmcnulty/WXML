'''

Finds matching complex of a given graph.

VPython is used to graph this complex

'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import more_itertools as mitl
import itertools as itl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_2D_matching_complex(M_G, fill=[]):
    pos = nx.spring_layout(M_G, dim=2, iterations=100)

    nx.drawing.nx_pylab.draw(M_G, pos)
    nx.drawing.nx_pylab.draw_networkx_labels(M_G, pos)

    ax = plt.gca()
    for V in fill:
        x = [pos[v][0] for v in V]
        y = [pos[v][1] for v in V]
        ax.fill(x,y, "blue")


    plt.show()

def draw_3D_matching_complex(M_G,fill=[]):
    pos = nx.spring_layout(M_G, dim=3, iterations=100)

    fig = plt.figure()
    ax = fig.add_subplot('111', projection='3d')

    for V in fill:
        vx = [pos[v][0] for v in V]
        vy = [pos[v][1] for v in V]
        vz = [pos[v][2] for v in V]
        verts = list(zip(vx, vy, vz))
        
        # linewidth sets width of lines in polyhedron. Alpha sets transparency of polyhedron faces
        poly = Poly3DCollection((verts,), linewidth=1, alpha=0.2)

        if len(verts) == 3: #triangle
            poly.set_color('g')
        elif len(verts) == 4: # tetrahedron
            poly.set_color('b')
        ax.add_collection3d(poly)

    for key in pos:
        point = pos[key]
        ax.scatter(point[0], point[1], point[2], s=120, c='r', alpha=0.4, edgecolor='k')
        ax.text(point[0], point[1], point[2], key, fontsize=15) #horizontalalignment='center', verticalalignment='center')


    for e in M_G.edges():
        ax.plot(xs=[pos[e[0]][0], pos[e[1]][0]], ys=[pos[e[0]][1], pos[e[1]][1]], zs=[pos[e[0]][2], pos[e[1]][2]], c='k') 

    plt.show()

    
# Define graph as edge list  ================================================================
#edge_list = [(1,2), (2,3), (3,4), (1,4), (1,3), (2,4)] # K_4
#edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)] # C_6
edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1)] # C_7
#edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,1)] # C_8
# ==============================================================

edge_labels = {e:i+1 for i,e in enumerate(edge_list)}


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

#print("matchings: ", matchings)

maximal_matchings = set() 
for match in matchings:
    to_remove = set()
    for max_match in maximal_matchings:
        if set(match).issubset(set(max_match)): break
        if set(max_match).issubset(set(match)): to_remove.add(max_match)
    else:
        maximal_matchings.add(match)

    maximal_matchings = maximal_matchings - to_remove
        

print("Matching Complex is :", maximal_matchings)
max_dim = max([len(m) for m in maximal_matchings])
M_G = nx.Graph()

# NOTE: Larger weight = tries to force edge to be shorter
tetra_weight = 2 # how strong connection is between nodes of tetrahedron
tri_weight = 1 # how strong connection is between nodes in triangle
other_weight = 0.5 # how strong other node connections are (for spring model)
fill = []
for match in maximal_matchings:
    if len(match) == 3:
        for v1, v2 in itl.combinations(match, 2):
            M_G.add_edge(v1, v2, weight=tri_weight)
        fill.append(match) 

    elif len(match) == 4:
        for v1, v2 in itl.combinations(match, 2):
            M_G.add_edge(v1, v2, weight=tetra_weight)
        fill.append(match) 

    else:
        M_G.add_edge(match[0], match[1], weight=other_weight)

if max_dim > 4:
    print("Cannot draw matching complex. Simplicial complex dimension is > 3")
elif max_dim == 1 or max_dim == 0:
    print("Matching complex is trivial.")
#elif max_dim == 2 or max_dim == 3:
#    draw_2D_matching_complex(M_G,fill)

else: # dim == 3

    draw_3D_matching_complex(M_G,fill)


print("done!")
