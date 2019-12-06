'''

This file is for finding and drawing the matching complex of a given graph. It will output to the terminal
a string representing the matching complex and attempt to plot the complex (if it is at most 3D).

'''

from matching_complexes import *
import networkx as nx

# NOTE: Set parameters and define graph as edge list  ================================================================


draw_2D = False # draws a 2D plot for matching complex rather than 3D


# edge weights to give edges in (filled in) tetrahedron, triangles, or just plain edges in the
# matching complex. These weights are used in the drawing algorithm. A higher weight makes a given
# edge want to be shorter. Hence giving tetrahedrons a higher weight may encourage those edges to
# be more structured and actually form a tetrahedron rather than just get flattened.
tetra_weight = 1 
tri_weight = 1
other_weight = 1

# Number of iterations to use in the drawing algorithm. The algorithm is stochastic (spring-force algorithm)
# so more iterations may improve performance.
iterations = 100

# whether to make a video rotating simplicial complex. Helps visualize 3D complex.
make_video = True
vid_path='./presentation/c7_video.mp4'



'''
This is where you can specify the graph G you want to find the matching complex of.
Specify it as an edge list. Some examples can be found below.
'''


#edge_list = [(1,2), (2,3), (3,4), (1,4), (1,3), (2,4)] # K_4
#edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1)] # C_6
edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,1)] # C_7
#edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,1)] # C_8
#edge_list=[(1,2), (3,4), (5,6), (7,8)] # 4 disjoint edges

#K_k,n
'''
k=4
n=3
edge_list =[]
for i in range(k):
    for j in range(k, k+n):
        edge_list.append((i,j))

print(edge_list)
'''

#K_n
'''
n=6
edge_list=[(i,j) for i,j in itl.combinations(list(range(n)), 2)]
'''

# ============================================================================================

# NOTE: you don't need to modify anything down here :D 


G = nx.Graph()
G.add_edges_from(edge_list)

edge_labels = {e:i+1 for i,e in enumerate(G.edges())}

# draw the graph given by edge list
draw_graph(G, edge_labels=edge_labels)

maximal_matchings = find_matching_complex(G, edge_labels)

print("Matching Complex is :", maximal_matchings)

max_dim = max([len(m) for m in maximal_matchings])


if max_dim > 4:
    print("Cannot draw matching complex. Simplicial complex dimension is > 3")
elif max_dim == 1 or max_dim == 0:
    print("Matching complex is trivial.")
else:
    M_G, fill = make_matching_complex(maximal_matchings, tetra_weight=tetra_weight, tri_weight=tri_weight, other_weight=other_weight)

    if draw_2D:
        draw_2D_matching_complex(M_G,fill)

    else:
        draw_3D_matching_complex(M_G,fill, iterations = iterations, make_video=make_video, vid_path=vid_path)

print("done!")
