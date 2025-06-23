import networkx as nx

G = nx.Graph()

# Add typed nodes
G.add_node(0, type="X")
G.add_node(1, type="X")
G.add_node(2, type="X")
G.add_node(3, type="X")

# Add typed edges
G.add_edge(0, 1, type="a")
G.add_edge(0, 2, type="b")
G.add_edge(0, 3, type="c")
G.add_edge(2, 3, type="a")
G.add_edge(1, 3, type="b")
G.add_edge(1, 2, type="c")
