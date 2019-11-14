'''

    Given a matching/simplicial complex (specified as a list of edges and filled in faces) find a graph that could of
    generated it. Namely, a graph whose matching complex is the given complex. This may not be possible of course and
    even if it is, the graph found may not be unique (consider the matching complexes of the 3-cycle and of the 4 vertex
    star graph).

'''

from matching_complexes import *



# NOTE: edit the following variables to specify your matching complex ===============================
# labels to give vertices in M_G (and thus labels for the corresponding edges in G)
#vertices = [1,2,3,4,5,6]
vertices = [1,2,3]


# edges in matching complex (NOT higher dimensional faces; simply specify the one-skeleton here)
#edges = [(2,5), (6,3), (4,1), (2,4), (2,6), (6,4), (3,5), (5,1), (1,3)]
edges = []

# Specify higher dimensional faces here as a list of tuples. For example, the 4-tuple
# (1,2,3,4) would specify a filled in tetrahedron between vertices 1,2,3,4 in the matching complex
faces = [(2,4,6), (1,3,5)]

#=====================================================================================================

G, edge_labels = graph_from_complex(edges, vertices, faces, simple_graph=True)
print('graph: ', G.edges())
draw_graph(G, edge_labels=edge_labels)

