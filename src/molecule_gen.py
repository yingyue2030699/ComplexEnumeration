import networkx as nx
from itertools import combinations, chain
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.components import connected_components

def powerset_connected_nodes(nodes):
    """Generate all non-empty subsets of nodes, up to len(nodes)"""
    nodes = list(nodes)
    for r in range(1, len(nodes) + 1):
        yield from combinations(nodes, r)

def get_unique_fully_connected_subgraphs_optimized(G):
    seen_hashes = set()
    unique_subgraphs = []

    full_degrees = dict(G.degree())  # cache full graph degrees

    for component_nodes in connected_components(G):
        for node_subset in powerset_connected_nodes(component_nodes):
            #H = G.subgraph(node_subset).copy() <- Avoid copy graph, 20% speed up
            H = G.subgraph(node_subset)

            if len(H) == 1:
                # Allow size-1 subgraphs only if connected in G
                node = node_subset[0]
                if full_degrees[node] == 0:
                    continue
            else:
                # Require every node in subgraph to have degree >= 1 in H
                if not nx.is_connected(H):
                    continue

            # Use canonical hash for deduplication
            H_relabel = nx.convert_node_labels_to_integers(H)
            wl_hash = weisfeiler_lehman_graph_hash(H_relabel, node_attr="type", edge_attr="type")

            if wl_hash not in seen_hashes:
                seen_hashes.add(wl_hash)
                unique_subgraphs.append(H)

    return unique_subgraphs



if __name__ == "__main__":
    import time
    
    G = nx.Graph()

    # Add typed nodes
    G.add_node(0, type="A")
    G.add_node(1, type="A")
    G.add_node(2, type="A")
    G.add_node(3, type="A")
    G.add_node(4, type="A")
    G.add_node(5, type="A")
    G.add_node(6, type="A")
    G.add_node(7, type="A")
    G.add_node(8, type="A")
    G.add_node(9, type="A")
    G.add_node(10, type="A")
    G.add_node(11, type="A")
    G.add_node(12, type="A")
    G.add_node(13, type="A")
    G.add_node(14, type="A")
    G.add_node(15, type="A")
    G.add_node(16, type="A")
    G.add_node(17, type="A")

    # Add typed edges
    G.add_edge(0, 1, type="hex")
    G.add_edge(1, 2, type="hex")
    G.add_edge(2, 3, type="hex")
    G.add_edge(3, 4, type="hex")
    G.add_edge(4, 5, type="hex")
    G.add_edge(5, 0, type="hex")
    G.add_edge(6, 7, type="tri")
    G.add_edge(7, 0, type="tri")
    G.add_edge(0, 6, type="tri")
    G.add_edge(7, 1, type="di")
    G.add_edge(8, 9, type="tri")
    G.add_edge(9, 1, type="tri")
    G.add_edge(1, 8, type="tri")
    G.add_edge(9, 2, type="di")
    G.add_edge(10, 11, type="tri")
    G.add_edge(11, 2, type="tri")
    G.add_edge(2, 10, type="tri")
    G.add_edge(11, 3, type="di")
    G.add_edge(12, 13, type="tri")
    G.add_edge(13, 3, type="tri")
    G.add_edge(3, 12, type="tri")
    G.add_edge(13, 4, type="di")
    G.add_edge(14, 15, type="tri")
    G.add_edge(15, 4, type="tri")
    G.add_edge(4, 14, type="tri")
    G.add_edge(15, 5, type="di")
    G.add_edge(16, 17, type="tri")
    G.add_edge(17, 5, type="tri")
    G.add_edge(5, 16, type="tri")
    G.add_edge(17, 0, type="di")

    import time
    start = time.time()


    unique_subgraphs = get_unique_fully_connected_subgraphs_optimized(G)
    
    print("Elapsed:", time.time() - start)
    
    print(len(unique_subgraphs))
