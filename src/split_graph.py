from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import networkx as nx
from itertools import combinations

def is_connected_induced(G, nodes):
    return nx.is_connected(G.subgraph(nodes))

def typed_hash(Gsub):
    return weisfeiler_lehman_graph_hash(
        Gsub, 
        node_attr="type", 
        edge_attr="type"
    )

def all_unique_induced_splits(G):
    nodes = list(G.nodes)
    n = len(nodes)
    seen = set()
    
    for r in range(1, n // 2 + 1):
        for combo in combinations(nodes, r):
            A = set(combo)
            B = set(nodes) - A
            G1 = G.subgraph(A).copy()
            G2 = G.subgraph(B).copy()
            if not (nx.is_connected(G1) and nx.is_connected(G2)):
                continue
            h1, h2 = typed_hash(G1), typed_hash(G2)
            key = tuple(sorted((h1, h2)))  # symmetric key
            if key in seen:
                continue
            seen.add(key)
            yield (G1, G2)


if __name__ == "__main__":
    # Main block
    G = nx.Graph()
    G.add_node(0, type="X")
    G.add_node(1, type="X")
    G.add_node(2, type="X")
    G.add_node(3, type="X")

    G.add_edge(0, 1, type="a")
    G.add_edge(0, 2, type="b")
    G.add_edge(0, 3, type="c")
    G.add_edge(2, 3, type="a")
    G.add_edge(1, 3, type="b")
    G.add_edge(1, 2, type="c")

    for part1, part2 in all_unique_induced_splits(G):
        print("Split:")
        print("  Part 1:", part1.edges)
        print("  Part 2:", part2.edges)