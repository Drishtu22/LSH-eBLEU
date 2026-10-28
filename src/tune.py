"""Choose delta, k and L on the WMT19 dev set only (WMT20 stays untouched).

1. delta: sweep with exact all-pairs soft matching, keep the best average Pearson.
2. (k, L): with delta fixed, keep the configuration with the fewest similarity
   computations whose average Pearson and Kendall are both within TOLERANCE of
   exact soft matching.
"""

import json
import time

import numpy as np
import pandas as pd

from embeddings import Embedder, target_language
from evaluation import correlations
from soft_bleu import counts_exact, counts_lsh, hyperplanes, prepare


DEV_FILE = "data/processed/wmt19_dev.csv"
DELTAS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
BITS = [4, 6, 8, 10, 12, 14, 16]
TABLES = [1, 2, 4, 8, 16, 32]
TOLERANCE = 0.005
SEED = 42


def prepare_split(data, planes):
    embedders = {lang: Embedder(lang) for lang in data["lp"].map(target_language).unique()}
    return [
        prepare(mt, ref, embedders[target_language(lp)], planes)
        for mt, ref, lp in zip(data["mt"], data["ref"], data["lp"])
    ]


def evaluate(data, segments, scorer):
    """Average correlations plus total similarity / hash cost for one setting."""
    results = [scorer(segment) for segment in segments]
    scored = data.assign(metric=[r[0] for r in results])
    table = correlations(scored, ["metric"])
    avg = table[table["lp"] == "avg"].iloc[0]
    return {
        "pearson": avg["pearson"],
        "kendall": avg["kendall"],
        "similarity_computations": sum(r[1] for r in results),
        "hash_projections": sum(r[2] for r in results),
        "scores": scored["metric"].to_numpy(),
        "table": table,
    }


if __name__ == "__main__":

    data = pd.read_csv(DEV_FILE)
    planes = hyperplanes(SEED)

    start = time.perf_counter()
    segments = prepare_split(data, planes)
    print(f"Prepared {len(segments)} dev segments in {time.perf_counter() - start:.1f}s")

    # Step 1: delta with exact soft matching
    delta_rows = []
    for delta in DELTAS:
        result = evaluate(data, segments, lambda s: counts_exact(s, delta))
        delta_rows.append({
            "delta": delta,
            "pearson": result["pearson"],
            "kendall": result["kendall"],
            "similarity_computations": result["similarity_computations"],
        })
        print(f"delta={delta:.2f}  pearson={result['pearson']:.4f}  kendall={result['kendall']:.4f}")

    delta_table = pd.DataFrame(delta_rows)
    delta_table.to_csv("results/dev_delta_sweep.csv", index=False)
    best_delta = float(delta_table.loc[delta_table["pearson"].idxmax(), "delta"])

    exact = evaluate(data, segments, lambda s: counts_exact(s, best_delta))
    exact["table"].assign(metric="soft_exact").to_csv(
        "results/dev_soft_exact_correlations.csv", index=False
    )
    print(f"\nChosen delta = {best_delta}  (exact soft: pearson={exact['pearson']:.4f}, "
          f"kendall={exact['kendall']:.4f}, sims={exact['similarity_computations']})")

    # Step 2: (k, L) with LSH at the chosen delta
    grid_rows = []
    for k in BITS:
        for L in TABLES:
            start = time.perf_counter()
            result = evaluate(data, segments, lambda s: counts_lsh(s, best_delta, k, L))
            grid_rows.append({
                "k": k,
                "L": L,
                "pearson": result["pearson"],
                "kendall": result["kendall"],
                "similarity_computations": result["similarity_computations"],
                "hash_projections": result["hash_projections"],
                "sim_fraction": result["similarity_computations"] / exact["similarity_computations"],
                "agreement_with_exact": float(np.corrcoef(result["scores"], exact["scores"])[0, 1]),
                "mean_abs_diff_from_exact": float(np.abs(result["scores"] - exact["scores"]).mean()),
            })
            row = grid_rows[-1]
            print(f"k={k:2d} L={L:2d}  pearson={row['pearson']:.4f}  kendall={row['kendall']:.4f}  "
                  f"sims={row['sim_fraction']:.3f} of exact  ({time.perf_counter() - start:.1f}s)")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv("results/dev_lsh_grid.csv", index=False)

    eligible = grid[
        (grid["pearson"] >= exact["pearson"] - TOLERANCE)
        & (grid["kendall"] >= exact["kendall"] - TOLERANCE)
    ]
    best = eligible.sort_values(["similarity_computations", "hash_projections"]).iloc[0]

    selected = {
        "delta": best_delta,
        "k": int(best["k"]),
        "L": int(best["L"]),
        "seed": SEED,
        "tolerance": TOLERANCE,
        "dev_exact_soft": {
            "pearson": exact["pearson"],
            "kendall": exact["kendall"],
            "similarity_computations": int(exact["similarity_computations"]),
        },
        "dev_lsh": {
            "pearson": float(best["pearson"]),
            "kendall": float(best["kendall"]),
            "similarity_computations": int(best["similarity_computations"]),
            "hash_projections": int(best["hash_projections"]),
            "sim_fraction": float(best["sim_fraction"]),
        },
    }
    with open("results/selected_params.json", "w") as f:
        json.dump(selected, f, indent=2)

    print("\nSelected:", json.dumps(selected, indent=2))
