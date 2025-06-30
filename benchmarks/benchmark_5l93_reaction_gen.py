import time
from datetime import datetime
from pathlib import Path
import platform
import os
import sys
import json
import pandas as pd

from ode_gen.complexes.examples import get_example
from ode_gen.complexes.subcomplexes import get_unique_fully_connected_subgraphs
from ode_gen.reactions.dimer import find_all_dimer_reactions, get_broken_edges
from ode_gen.reactions.transformation import find_all_transformable_subgraph_pairs

def capture_environment_info():
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu": platform.processor(),
    }

def benchmark_pipeline(graph_name="5l93"):
    print(f"Running benchmark for: {graph_name}")
    G = get_example(graph_name)

    results = {}
    env_info = capture_environment_info()

    t0 = time.time()
    species = get_unique_fully_connected_subgraphs(G)
    t1 = time.time()

    reactions = find_all_dimer_reactions(species, use_multiprocessing=True)
    t2 = time.time()

    reaction_df = pd.DataFrame([
        {
            "product": list(r[2].nodes),
            "part1": list(r[0]),
            "part2": list(r[1]),
            "bonds_broken": get_broken_edges(r[2], r[0], r[1])
        }
        for r in reactions
    ])

    transformations = find_all_transformable_subgraph_pairs(G, species)
    t3 = time.time()

    transformations_df = pd.DataFrame([
        {
            "monomer_1_nodes": list(t1.nodes),
            "monomer_2_nodes": list(t2.nodes),
            "diff": list(set(t1.edges(data="type")) ^ set(t2.edges(data="type")))
        }
        for t1, t2 in transformations
    ])

    # Timing summary
    results["timing"] = {
        "subgraph_enumeration": round(t1 - t0, 4),
        "dimer_reactions": round(t2 - t1, 4),
        "transformable_pairs": round(t3 - t2, 4),
        "total": round(t3 - t0, 4),
    }
    results["n_species"] = len(species)
    results["n_reactions"] = len(reaction_df)
    results["n_transformations"] = len(transformations_df)
    results["environment"] = env_info

    return reaction_df, transformations_df, results

def save_results(reaction_df, transformations_df, metadata, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    reaction_df.to_csv(out_dir / "reactions.csv", index=False)
    transformations_df.to_csv(out_dir / "transformations.csv", index=False)

    with open(out_dir / "benchmark_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved benchmark results to: {out_dir}")

if __name__ == "__main__":
    graph_name = "5l93"
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("benchmarks") / f"{graph_name}_benchmark_{now}"
    reaction_df, transformations_df, metadata = benchmark_pipeline(graph_name)
    save_results(reaction_df, transformations_df, metadata, out_dir)
