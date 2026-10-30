#!/usr/bin/env python3
"""
FedSUCP Sparsity Phase Diagram Study.

Maps oracle personalization over (Dirichlet α, cost λ) phase space.

Usage:
  python scripts/run_sparsity_phase_diagram.py --quick
  python scripts/run_sparsity_phase_diagram.py --resume
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from fedpbg.data.registry import load_registered_dataset
from fedpbg.oracle.experiment import run_oracle_client_eval
from fedpbg.oracle.sparsity_metrics import build_experiment_summary_row
from fedpbg.phase_diagram.viz import generate_phase_diagram_figures
from fedpbg.reporting.phase_diagram_summary import write_phase_diagram_summary

RESULTS_DIR = CFG_ROOT / "results" / "sparsity_phase_diagram"

# Phase diagram grids
ALPHAS_FULL = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 10.0]
LAMBDAS_FULL = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
SEEDS_FULL = [42, 43, 44]
DATASETS_DEFAULT = ["cifar100", "ptbxl", "cifar10"]


def parse_args():
    p = argparse.ArgumentParser(description="FedSUCP sparsity phase diagram")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--alphas", type=str, default=None)
    p.add_argument("--lambdas", type=str, default=None)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--global-rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--max-records", type=int, default=None, help="Cap PTB-XL records (e.g. 5000)")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--reuse-oracle-cache",
        action="store_true",
        help="Also check results/oracle_sparsity_validation/cache for matching configs",
    )
    return p.parse_args()


def alpha_cache_id(dataset: str, seed: int, alpha: float, num_clients: int) -> str:
    return f"{dataset}_s{seed}_a{alpha}_n{num_clients}"


def find_cached_client_csv(
    cache_dirs: list[Path],
    cid: str,
) -> Path | None:
    for d in cache_dirs:
        p = d / f"{cid}_client_level.csv"
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else CFG_ROOT / "data"

    if args.quick:
        datasets = (args.datasets or "ptbxl,cifar100").split(",")
        seeds = [int(x) for x in (args.seeds or "42").split(",")]
        alphas = [float(x) for x in (args.alphas or "0.1,0.5,1.0").split(",")]
        lambdas = [float(x) for x in (args.lambdas or "0.0,0.3,1.0").split(",")]
        global_rounds = args.global_rounds or 2
        local_epochs = args.local_epochs or 1
        quick = True
    else:
        datasets = (args.datasets or ",".join(DATASETS_DEFAULT)).split(",")
        seeds = [int(x) for x in (args.seeds or ",".join(map(str, SEEDS_FULL))).split(",")]
        alphas = [float(x) for x in (args.alphas or ",".join(map(str, ALPHAS_FULL))).split(",")]
        lambdas = [float(x) for x in (args.lambdas or ",".join(map(str, LAMBDAS_FULL))).split(",")]
        global_rounds = args.global_rounds or 5
        local_epochs = args.local_epochs or 3
        quick = False

    device = torch.device(args.device)
    num_clients = args.num_clients

    cache_dirs = [cache_dir]
    if args.reuse_oracle_cache:
        cache_dirs.append(CFG_ROOT / "results" / "oracle_sparsity_validation" / "cache")

    print("=" * 70)
    print("FedSUCP Sparsity Phase Diagram Study")
    print("=" * 70)
    print(f"Framing: When is personalization worth its cost?")
    print(f"Datasets: {datasets}")
    print(f"Alphas: {alphas}")
    print(f"Lambdas: {lambdas} (post-hoc from cached client evals)")
    print(f"Seeds: {seeds} | Clients: {num_clients}")
    print(f"Results: {results_dir}")

    phase_rows = []
    alpha_grid = list(product(datasets, seeds, alphas))

    for dataset_name, seed, alpha in tqdm(alpha_grid, desc="Alpha configs"):
        cid = alpha_cache_id(dataset_name, seed, alpha, num_clients)
        cache_path = cache_dir / f"{cid}_client_level.csv"

        try:
            if args.resume and cache_path.exists():
                client_df = pd.read_csv(cache_path)
            else:
                reused = find_cached_client_csv(cache_dirs, cid)
                if reused and reused != cache_path:
                    client_df = pd.read_csv(reused)
                    client_df.to_csv(cache_path, index=False)
                elif cache_path.exists():
                    client_df = pd.read_csv(cache_path)
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

        for lam in lambdas:
            phase_rows.append(build_experiment_summary_row(client_df, lam, meta))

    if not phase_rows:
        print("No phase diagram cells completed.")
        sys.exit(1)

    results_df = pd.DataFrame(phase_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "phase_diagram_results.csv", index=False)

    generate_phase_diagram_figures(results_df, results_dir, alphas, lambdas)
    write_phase_diagram_summary(results_df, results_dir / "phase_diagram_summary.md")

    print(f"\n[Output] {results_dir / 'phase_diagram_results.csv'}")
    print(f"[Output] {results_dir / 'phase_diagram_summary.md'}")
    print(f"[Output] 6 figures in {results_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
