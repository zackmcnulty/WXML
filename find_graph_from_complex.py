'''

    Given a matching/simplicial complex (specified as a list of edges and filled in faces) find a graph that could of
    generated it. Namely, a graph whose matching complex is the given complex. This may not be possible of course and
    even if it is, the graph found may not be unique (consider the matching complexes of the 3-cycle and of the 4 vertex
    star graph).

'''

from matching_complexes import *



# NOTE: edit the following variables to specify your matching complex ===============================

# Specify edges and higher dimensional faces here as a list of tuples. For example, the 4-tuple
# (1,2,3,4) would specify a filled in tetrahedron between vertices 1,2,3,4 in the matching complex
# while the 2-tuple (5,7) specifies an edge between vertex 5 and 7.
#faces = [(1,2), (2,3), (3,4), (4,5), (5,1)] # should be C5
faces = [(2,4,6), (1,5,3), (1,2), (3,4), (5,6)] # should be C6
faces = [(1,), (2,), (3,)]

simple_graph = True  # if true, only looks for simple graphs that generate given matching complex

# =====================================================================================================

G, edge_labels = graph_from_complex(faces, simple_graph=simple_graph)
print('graph: ', G.edges())
draw_graph(G, edge_labels=edge_labels)

