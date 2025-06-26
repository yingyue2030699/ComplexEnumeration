# Re-import missing dependency
from itertools import combinations

# Re-run everything after adding missing import

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

from molecule_gen import get_unique_fully_connected_subgraphs
from split_graph import all_unique_induced_splits


# first of all, we get all unique products because if the product is
# different, the reaction has to be different

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

    species = get_unique_fully_connected_subgraphs(G)
    
    reactions = []
    for specie in species:
        for part in all_unique_induced_splits(specie):
            part1, part2 = part
            reactions.append((part1, part2, specie))

    for part1, part2, specie in reactions:
        print(f"[{part1.nodes}] + [{part2.nodes}] -> [{specie.nodes}]")
