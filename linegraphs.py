'''
Finds and draws the line graph of a given graph G (specified by an edge list

'''
import networkx as nx
import matplotlib.pyplot as plt
import itertools as itl

# NOTE: define graph here =======================================================================

n = 7
#edge_list=[(i,i+1) for i in range(n)] # P_n
edge_list=[(i,i+1) for i in range(n)] + [(n, 0)]# C_n

# K_k,n
'''
k=8
n=3
edge_list =[]
for i in range(k):
    for j in range(k, k+n):
        edge_list.append((i,j))

print(edge_list)
'''

# K_n
'''
n=4
edge_list=[]
for v1,v2 in itl.combinations(list(range(n)), 2):
    edge_list.append((v1,v2))
'''


# ==============================================================================================

G = nx.Graph()
G.add_edges_from(edge_list)

L = nx.line_graph(G)

pos=nx.spring_layout(L)
nx.drawing.nx_pylab.draw(L, pos)
#nx.drawing.nx_pylab.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
nx.drawing.nx_pylab.draw_networkx_labels(L, pos)
plt.title('Graph G')
plt.show()

print("Diameter of G: ", nx.diameter(G), " , Diameter of L: ",  nx.diameter(L))
