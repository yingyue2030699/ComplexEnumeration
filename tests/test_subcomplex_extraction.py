import unittest
import networkx as nx
import time
import warnings

# Replace with actual import path
from ode_gen.complexes.subcomplexes import get_unique_fully_connected_subgraphs

class TestFullyConnectedSubgraphDetection8y7s(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_node(0, type="X")
        self.G.add_node(1, type="X")
        self.G.add_node(2, type="X")
        self.G.add_node(3, type="X")

        self.G.add_edge(0, 1, type="a")
        self.G.add_edge(0, 2, type="b")
        self.G.add_edge(0, 3, type="c")
        self.G.add_edge(2, 3, type="a")
        self.G.add_edge(1, 3, type="b")
        self.G.add_edge(1, 2, type="c")

    def test_unique_subgraphs(self):
        start = time.time()
        result = get_unique_fully_connected_subgraphs(self.G)
        elapsed = time.time() - start

        if elapsed > 10.0:
            self.fail(f"ERROR: Test terminated — elapsed time {elapsed:.2f}s exceeds 10 seconds.")
        elif elapsed > 0.01:
            warnings.warn(f"WARNING: Elapsed time {elapsed:.2f}s exceeds 0.1 second.")

        self.assertEqual(
            len(result), 6,
            f"ERROR: Expected 6 unique subgraphs, but got {len(result)}."
        )

        print(f"Test passed in {elapsed:.4f} seconds.")

class TestFullyConnectedSubgraphDetectionHetero8mer(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_node(0, type="A")
        self.G.add_node(1, type="B")
        self.G.add_node(2, type="C")
        self.G.add_node(3, type="D")
        self.G.add_node(4, type="E")
        self.G.add_node(5, type="F")
        self.G.add_node(6, type="G")
        self.G.add_node(7, type="H")

        self.G.add_edge(0, 1, type="ab")
        self.G.add_edge(1, 2, type="bc")
        self.G.add_edge(2, 3, type="cd")
        self.G.add_edge(3, 4, type="de")
        self.G.add_edge(4, 5, type="ef")
        self.G.add_edge(5, 6, type="fg")
        self.G.add_edge(6, 7, type="gh")
        self.G.add_edge(7, 0, type="ha")

    def test_unique_subgraphs(self):
        start = time.time()
        result = get_unique_fully_connected_subgraphs(self.G)
        elapsed = time.time() - start

        if elapsed > 10.0:
            self.fail(f"ERROR: Test terminated — elapsed time {elapsed:.2f}s exceeds 10 seconds.")
        elif elapsed > 0.1:
            warnings.warn(f"WARNING: Elapsed time {elapsed:.2f}s exceeds 0.1 second.")

        self.assertEqual(
            len(result), 57,
            f"ERROR: Expected 57 unique subgraphs, but got {len(result)}."
        ) # 7 * 8 + 1 = 57

        print(f"Test passed in {elapsed:.4f} seconds.")
        
class TestFullyConnectedSubgraphDetectionAsymmetric(unittest.TestCase):
    def setUp(self):
        self.G = nx.Graph()
        # Add typed nodes
        self.G.add_node(0, type="A")
        self.G.add_node(1, type="B")
        self.G.add_node(2, type="C")
        self.G.add_node(3, type="A")

        # Add typed edges
        self.G.add_edge(0, 1, type="ab")
        self.G.add_edge(1, 2, type="bc")
        self.G.add_edge(2, 3, type="ca")
        self.G.add_edge(1, 3, type="ab")

    def test_unique_subgraphs(self):
        start = time.time()
        result = get_unique_fully_connected_subgraphs(self.G)
        elapsed = time.time() - start

        if elapsed > 10.0:
            self.fail(f"ERROR: Test terminated — elapsed time {elapsed:.2f}s exceeds 10 seconds.")
        elif elapsed > 0.1:
            warnings.warn(f"WARNING: Elapsed time {elapsed:.2f}s exceeds 0.1 second.")

        self.assertEqual(
            len(result), 10,
            f"ERROR: Expected 10 unique subgraphs, but got {len(result)}."
        ) # 3 + 3 + 3 + 1 = 10

        print(f"Test passed in {elapsed:.4f} seconds.")

if __name__ == '__main__':
    unittest.main()
