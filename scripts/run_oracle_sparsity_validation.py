#!/usr/bin/env python3
"""
FedSUCP Oracle Sparsity Validation Study.

Evaluates oracle personalization actions P0–P5 across multiple datasets and
federated settings to validate sparsity hypothesis.

Usage:
  python scripts/run_oracle_sparsity_validation.py --quick
  python scripts/run_oracle_sparsity_validation.py --datasets ptbxl,cifar100,femnist
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.config import (  # noqa: E402
    DEFAULT_LAMBDA,
    LAMBDA_VALUES,
    PROJECT_ROOT as CFG_ROOT,
)
from fedpbg.data.registry import load_registered_dataset
from fedpbg.oracle.experiment import run_oracle_client_eval
from fedpbg.oracle.sparsity_metrics import (
    build_experiment_summary_row,
    build_oracle_client_rows,
)
from fedpbg.oracle.viz import generate_validation_figures
from fedpbg.reporting.oracle_sparsity_summary import write_oracle_sparsity_summary

RESULTS_DIR = CFG_ROOT / "results" / "oracle_sparsity_validation"

# Full study grid (FedSUCP ICML validation)
DATASETS_DEFAULT = ["ptbxl", "cifar100", "femnist"]
SEEDS_DEFAULT = [42, 43, 44]
ALPHAS_DEFAULT = [0.1, 0.5, 1.0]
CLIENT_COUNTS_DEFAULT = [20, 100]
LAMBDAS = LAMBDA_VALUES


def parse_args():
    p = argparse.ArgumentParser(description="FedSUCP oracle sparsity validation")
    p.add_argument("--quick", action="store_true", help="Reduced grid for smoke testing")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--alphas", type=str, default=None)
    p.add_argument("--client-counts", type=str, default=None)
    p.add_argument("--global-rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--max-records", type=int, default=None, help="Cap PTB-XL records (e.g. 5000)")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--resume", action="store_true", help="Skip configs with cached client CSV")
    return p.parse_args()


def config_id(dataset: str, seed: int, alpha: float, num_clients: int) -> str:
    return f"{dataset}_s{seed}_a{alpha}_n{num_clients}"


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else CFG_ROOT / "data"

    if args.quick:
        datasets = (args.datasets or "ptbxl,cifar100,femnist").split(",")
        seeds = [int(x) for x in (args.seeds or "42").split(",")]
        alphas = [float(x) for x in (args.alphas or "0.1").split(",")]
        client_counts = [int(x) for x in (args.client_counts or "20").split(",")]
        global_rounds = args.global_rounds or 2
        local_epochs = args.local_epochs or 1
        quick = True
    else:
        datasets = (args.datasets or ",".join(DATASETS_DEFAULT)).split(",")
        seeds = [int(x) for x in (args.seeds or ",".join(map(str, SEEDS_DEFAULT))).split(",")]
        alphas = [float(x) for x in (args.alphas or ",".join(map(str, ALPHAS_DEFAULT))).split(",")]
        client_counts = [
            int(x) for x in (args.client_counts or ",".join(map(str, CLIENT_COUNTS_DEFAULT))).split(",")
        ]
        global_rounds = args.global_rounds or 5
        local_epochs = args.local_epochs or 3
        quick = False

    device = torch.device(args.device)
    np.random.seed(seeds[0])

    print("=" * 70)
    print("FedSUCP Oracle Sparsity Validation")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds} | Alphas: {alphas} | Clients: {client_counts}")
    print(f"Global rounds: {global_rounds} | Local epochs: {local_epochs}")
    print(f"Results: {results_dir}")

    all_client_rows = []
    summary_rows = []
    oracle_client_rows = []

    grid = list(product(datasets, seeds, alphas, client_counts))
    for dataset_name, seed, alpha, num_clients in tqdm(grid, desc="Configurations"):
        cid = config_id(dataset_name, seed, alpha, num_clients)
        cache_path = cache_dir / f"{cid}_client_level.csv"

        try:
            if args.resume and cache_path.exists():
                client_df = pd.read_csv(cache_path)
                spec = None
                resolved = client_df["dataset"].iloc[0] if "dataset" in client_df.columns else dataset_name
            else:
                spec, pytorch_ds, labels, resolved = load_registered_dataset(
                    dataset_name,
                    data_root,
                    quick=quick,
                    max_records=args.max_records or (800 if quick else None),
                )
                client_df = run_oracle_client_eval(
                    spec=spec,
                    dataset=pytorch_ds,
                    labels=labels,
                    num_clients=num_clients,
                    dirichlet_alpha=alpha,
                    seed=seed,
                    global_rounds=global_rounds,
                    local_epochs=local_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                )
                client_df["dataset"] = resolved
                client_df["dataset_requested"] = dataset_name
                client_df["seed"] = seed
                client_df["dirichlet_alpha"] = alpha
                client_df["num_clients"] = num_clients
                client_df.to_csv(cache_path, index=False)
        except FileNotFoundError as e:
            print(f"[SKIP] {cid}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] {cid}: {e}")
            continue

        meta = {
            "dataset": client_df["dataset"].iloc[0],
            "dataset_requested": dataset_name,
            "seed": seed,
            "dirichlet_alpha": alpha,
            "num_clients": num_clients,
        }

        for lam in LAMBDAS:
            summary_rows.append(
                build_experiment_summary_row(client_df, lam, meta)
            )

        oracle_part = build_oracle_client_rows(client_df, LAMBDAS, meta)
        oracle_client_rows.append(oracle_part)

        for lam in LAMBDAS:
            expanded = client_df.copy()
            expanded["lambda"] = lam
            all_client_rows.append(expanded)

    if not summary_rows:
        print("No experiments completed. Check data availability.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    oracle_df = pd.concat(oracle_client_rows, ignore_index=True)

    # Full per-configuration results
    sparsity_results = summary_df.copy()

    # Dataset-level aggregation (mean/std across seeds, alphas, client counts)
    dataset_summary = (
        summary_df.groupby(["dataset", "lambda"], as_index=False)
        .agg(
            p0_ratio=("p0_ratio", "mean"),
            p1_p2_ratio=("p1_p2_ratio", "mean"),
            deep_ratio=("deep_ratio", "mean"),
            best_uniform_utility=("best_uniform_utility", "mean"),
            oracle_utility=("oracle_utility", "mean"),
            personalize_all_utility=("personalize_all_utility", "mean"),
            oracle_vs_uniform_rel_gap=("oracle_vs_uniform_rel_gap", "mean"),
            oracle_vs_personalize_all_rel_gap=("oracle_vs_personalize_all_rel_gap", "mean"),
            num_configs=("seed", "count"),
        )
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    sparsity_results.to_csv(results_dir / "oracle_sparsity_results.csv", index=False)
    dataset_summary.to_csv(results_dir / "dataset_level_summary.csv", index=False)
    oracle_df.to_csv(results_dir / "oracle_client_assignments.csv", index=False)

    generate_validation_figures(summary_df, oracle_df, results_dir, DEFAULT_LAMBDA)
    write_oracle_sparsity_summary(summary_df, results_dir / "oracle_sparsity_summary.md", DEFAULT_LAMBDA)

    print(f"\n[Output] {results_dir / 'oracle_sparsity_results.csv'}")
    print(f"[Output] {results_dir / 'dataset_level_summary.csv'}")
    print(f"[Output] {results_dir / 'oracle_sparsity_summary.md'}")
    print(f"[Output] 5 figures in {results_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
