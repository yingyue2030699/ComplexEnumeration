import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from itertools import combinations
from multiprocessing import Pool, cpu_count
import time

def is_connected(G, nodes):
    """Check if nodes induce a connected subgraph in G."""
    return nx.is_connected(G.subgraph(nodes))

def typed_wl_hash(Gsub):
    """Get WL graph hash using node and edge types."""
    return weisfeiler_lehman_graph_hash(Gsub, node_attr="type", edge_attr="type")

def all_unique_induced_splits(G):
    """Yield all unique ways to split G into two connected, typed-isomorphic-aware induced subgraphs."""
    nodes = list(G.nodes)
    n = len(nodes)
    seen = set()
    
    for r in range(1, n // 2 + 1):
        for A in combinations(nodes, r):
            A = set(A)
            B = set(nodes) - A

            if not is_connected(G, A) or not is_connected(G, B):
                continue

            h1 = typed_wl_hash(G.subgraph(A))
            h2 = typed_wl_hash(G.subgraph(B))
            key = tuple(sorted((h1, h2)))

            if key in seen:
                continue
            seen.add(key)
            yield A, B

def deduplicate_species(species):
    """Filter out isomorphic species using WL hash."""
    seen_hashes = set()
    unique_species = []
    for sp in species:
        h = typed_wl_hash(sp)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_species.append(sp)
    return unique_species

def compute_reactions_for_species(specie):
    """Compute reactions (split pairs) for a single species."""
    reactions = []
    for part1, part2 in all_unique_induced_splits(specie):
        reactions.append((part1, part2, specie))
    return reactions

def compute_all_reactions(species, use_multiprocessing=False):
    """Compute all reactions across a list of species with optional multiprocessing."""
    species = deduplicate_species(species)

    if use_multiprocessing:
        with Pool(cpu_count()) as pool:
            results = pool.map(compute_reactions_for_species, species)
        reactions = [r for group in results for r in group]
    else:
        reactions = []
        for specie in species:
            reactions.extend(compute_reactions_for_species(specie))
    return reactions

# Sample test graph
def example_graph():
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
    return G

# Generate connected subgraphs of G
def get_unique_fully_connected_subgraphs(G):
    """Naive generation of all connected induced subgraphs."""
    nodes = list(G.nodes)
    connected_subgraphs = []
    for r in range(2, len(nodes) + 1):
        for combo in combinations(nodes, r):
            subG = G.subgraph(combo)
            if nx.is_connected(subG):
                connected_subgraphs.append(subG.copy())
    return connected_subgraphs

from example_complex_graph import ExampleGraph
# Run full pipeline
G = ExampleGraph.get_5l93()
t0 = time.time()
species = get_unique_fully_connected_subgraphs(G)
t1 = time.time()
reactions = compute_all_reactions(species, use_multiprocessing=True)
t2 = time.time()

import pandas as pd

reaction_df = pd.DataFrame([
    {"specie_nodes": list(r[2].nodes), "part1": list(r[0]), "part2": list(r[1])}
    for r in reactions
])

print(reaction_df)
print(t1 - t0, t2 - t1)
