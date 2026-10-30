#!/usr/bin/env python3
"""
Oracle Regret Study — final ICML scientific validation.

Evaluates ORACLE vs RANDOM, BEST_UNIFORM, PERSONALIZE_ALL, GLOBAL_ONLY
from cached client-level oracle results. No new algorithms.

Usage:
  python scripts/run_oracle_regret_study.py
  python scripts/run_oracle_regret_study.py --lambdas 0.0,0.3,1.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from fedpbg.final_validation.viz import generate_regret_figures
from fedpbg.oracle.policy_eval import POLICIES, evaluate_all_policies
from fedpbg.oracle.regret_stats import (
    BASELINE_POLICIES,
    aggregate_significance,
    build_regret_table,
    paired_oracle_test,
    robustness_by_dataset,
)
from fedpbg.reporting.oracle_regret_summary import write_oracle_regret_summary

RESULTS_DIR = CFG_ROOT / "results" / "final_validation"
LAMBDAS_DEFAULT = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

CACHE_PATTERN = re.compile(r"^(?P<dataset>\w+)_s(?P<seed>\d+)_a(?P<alpha>[\d.]+)_n(?P<clients>\d+)_client_level\.csv$")


def parse_args():
    p = argparse.ArgumentParser(description="Oracle regret final validation")
    p.add_argument("--lambdas", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--cache-dirs", type=str, default=None, help="Comma-separated cache directories")
    return p.parse_args()


def discover_cache_files(cache_dirs: list[Path]) -> list[Path]:
    files = []
    for d in cache_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*_client_level.csv")))
    # Deduplicate by filename (same config in multiple dirs)
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique


def parse_cache_meta(path: Path, client_df: pd.DataFrame) -> dict:
    m = CACHE_PATTERN.match(path.name)
    if m:
        return {
            "dataset": client_df["dataset"].iloc[0] if "dataset" in client_df.columns else m.group("dataset"),
            "dataset_requested": client_df["dataset_requested"].iloc[0] if "dataset_requested" in client_df.columns else m.group("dataset"),
            "seed": int(m.group("seed")),
            "dirichlet_alpha": float(m.group("alpha")),
            "num_clients": int(m.group("clients")),
        }
    return {
        "dataset": client_df["dataset"].iloc[0],
        "dataset_requested": (
            client_df["dataset_requested"].iloc[0]
            if "dataset_requested" in client_df.columns
            else client_df["dataset"].iloc[0]
        ),
        "seed": int(client_df["seed"].iloc[0]),
        "dirichlet_alpha": float(client_df["dirichlet_alpha"].iloc[0]),
        "num_clients": int(client_df["num_clients"].iloc[0]),
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [float(x) for x in (args.lambdas or ",".join(map(str, LAMBDAS_DEFAULT))).split(",")]

    if args.cache_dirs:
        cache_dirs = [Path(x) for x in args.cache_dirs.split(",")]
    else:
        cache_dirs = [
            CFG_ROOT / "results" / "sparsity_phase_diagram" / "cache",
            CFG_ROOT / "results" / "oracle_sparsity_validation" / "cache",
        ]

    cache_files = discover_cache_files(cache_dirs)
    if not cache_files:
        print("No cached client-level CSVs found. Run phase diagram or oracle validation first.")
        sys.exit(1)

    print("=" * 70)
    print("Oracle Regret Study — Final Validation")
    print("=" * 70)
    print(f"Cache files: {len(cache_files)}")
    print(f"Lambdas: {lambdas}")
    print(f"Policies: {POLICIES}")

    regret_rows = []
    sig_inputs = []

    for cache_path in tqdm(cache_files, desc="Configs"):
        client_df = pd.read_csv(cache_path)
        meta = parse_cache_meta(cache_path, client_df)

        for lam in lambdas:
            rows, results = evaluate_all_policies(client_df, lam, meta["seed"], meta)
            regret_rows.extend(rows)

            oracle_utils = results["ORACLE"]["client_utilities"]
            policy_utils = {
                pol: results[pol]["client_utilities"] for pol in BASELINE_POLICIES
            }
            sig_inputs.append(
                {
                    "meta": meta,
                    "lambda": lam,
                    "oracle_utils": oracle_utils,
                    "policy_utils": policy_utils,
                }
            )

    regret_df = pd.DataFrame(regret_rows)
    regret_table = build_regret_table(regret_df)
    sig_df = aggregate_significance(sig_inputs)

    regret_df.to_csv(results_dir / "oracle_regret_results.csv", index=False)
    regret_table.to_csv(results_dir / "regret_table.csv", index=False)
    if not sig_df.empty:
        sig_df.to_csv(results_dir / "significance_summary.csv", index=False)

    generate_regret_figures(regret_df, results_dir)
    write_oracle_regret_summary(
        regret_df,
        sig_df,
        results_dir / "oracle_regret_summary.md",
        n_configs=len(cache_files),
        n_datasets=regret_df["dataset"].nunique(),
    )

    print(f"\n[Output] {results_dir / 'oracle_regret_results.csv'}")
    print(f"[Output] {results_dir / 'regret_table.csv'}")
    print(f"[Output] {results_dir / 'oracle_regret_summary.md'}")
    print(f"[Output] 7 figures in {results_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
