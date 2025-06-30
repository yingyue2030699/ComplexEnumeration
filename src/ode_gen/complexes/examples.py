import networkx as nx

def graph_5l93():
    G = nx.Graph()
    G.add_nodes_from([(i, {"type": "A"}) for i in range(18)])
    G.add_edges_from([
        (0, 1, {"type": "hex"}), (1, 2, {"type": "hex"}), (2, 3, {"type": "hex"}),
        (3, 4, {"type": "hex"}), (4, 5, {"type": "hex"}), (5, 0, {"type": "hex"}),
        (6, 7, {"type": "tri"}), (7, 0, {"type": "tri"}), (0, 6, {"type": "tri"}),
        (7, 1, {"type": "di"}), (8, 9, {"type": "tri"}), (9, 1, {"type": "tri"}),
        (1, 8, {"type": "tri"}), (9, 2, {"type": "di"}), (10, 11, {"type": "tri"}),
        (11, 2, {"type": "tri"}), (2, 10, {"type": "tri"}), (11, 3, {"type": "di"}),
        (12, 13, {"type": "tri"}), (13, 3, {"type": "tri"}), (3, 12, {"type": "tri"}),
        (13, 4, {"type": "di"}), (14, 15, {"type": "tri"}), (15, 4, {"type": "tri"}),
        (4, 14, {"type": "tri"}), (15, 5, {"type": "di"}), (16, 17, {"type": "tri"}),
        (17, 5, {"type": "tri"}), (5, 16, {"type": "tri"}), (17, 0, {"type": "di"}),
    ])
    return G

def graph_8y7s():
    G = nx.Graph()
    G.add_nodes_from([(i, {"type": "X"}) for i in range(4)])
    G.add_edges_from([
        (0, 1, {"type": "a"}), (0, 2, {"type": "b"}), (0, 3, {"type": "c"}),
        (2, 3, {"type": "a"}), (1, 3, {"type": "b"}), (1, 2, {"type": "c"}),
    ])
    return G

def graph_asymmetry_4mer():
    G = nx.Graph()
    G.add_nodes_from([(0, {"type": "A"}), (1, {"type": "B"}), (2, {"type": "C"}), (3, {"type": "A"})])
    G.add_edges_from([
        (0, 1, {"type": "ab"}), (1, 2, {"type": "bc"}), (2, 3, {"type": "ca"}), (3, 1, {"type": "ab"})
    ])
    return G

################################
# Graph Registry
################################

graph_registry = {
    "5l93": graph_5l93,
    "8y7s": graph_8y7s,
    "asymmetry_4mer": graph_asymmetry_4mer,
}

def get_example(name: str):
    """
    Get example with a key
    """
    try:
        return graph_registry[name]()
    except KeyError:
        raise ValueError(f"Unknown example: {name}. Available: {list(graph_registry)}")
    
def show_available_examples(do_print = True):
    """
    Return the registry of examples
    """
    if do_print:
        print("Available examples:", list(graph_registry.keys()))
    return(list(graph_registry.keys()))
