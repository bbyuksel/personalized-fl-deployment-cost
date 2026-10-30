#!/usr/bin/env python3
"""
Run Ditto baseline on Paper-1 configurations (cifar10, cifar100, ptbxl).

Produces:
  results/ditto/cache/*_ditto.csv
  results/ditto/ditto_vs_oracle.csv
  results/ditto/ditto_regret_summary.md
  results/ditto/ditto_positioning_report.md
  results/ditto/figures/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from fedpbg.data.registry import load_registered_dataset
from fedpbg.oracle.ditto_eval import LAMBDAS_REGRET, run_ditto_for_config, utilities_for_config
from fedpbg.oracle.policy_eval import evaluate_policy
from fedpbg.oracle.regret_stats import paired_oracle_test
from fedpbg.reporting.ditto_analysis import (
    build_ditto_vs_oracle_rows,
    plot_ditto_positioning,
    positioning_analysis,
    regret_summary_for_policy,
    write_positioning_report_md,
    write_regret_summary_md,
)

TARGET_DATASETS = {"cifar10", "cifar100", "ptbxl"}
CACHE_PATTERN = re.compile(
    r"^(?P<dataset>\w+)_s(?P<seed>\d+)_a(?P<alpha>[\d.]+)_n(?P<clients>\d+)_client_level\.csv$"
)
RESULTS_DIR = CFG_ROOT / "results" / "ditto"
LAMBDAS = LAMBDAS_REGRET


def parse_args():
    p = argparse.ArgumentParser(description="Ditto evaluation under Paper-1 utility U")
    p.add_argument("--datasets", type=str, default="cifar10,cifar100,ptbxl")
    p.add_argument("--max-configs", type=int, default=None)
    p.add_argument("--skip-training", action="store_true", help="Only analyze existing ditto caches")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ptbxl-max-records", type=int, default=5000)
    return p.parse_args()


def discover_configs(cache_dirs: list[Path], datasets: set[str]) -> list[tuple[Path, dict]]:
    seen = set()
    configs = []
    for d in cache_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*_client_level.csv")):
            if f.name in seen:
                continue
            m = CACHE_PATTERN.match(f.name)
            if not m:
                continue
            ds_req = m.group("dataset")
            if ds_req not in datasets:
                continue
            seen.add(f.name)
            meta = {
                "cache_path": f,
                "dataset_requested": ds_req,
                "seed": int(m.group("seed")),
                "dirichlet_alpha": float(m.group("alpha")),
                "num_clients": int(m.group("clients")),
            }
            configs.append((f, meta))
    return configs


def dataset_name_from_cache(cache_df: pd.DataFrame, requested: str) -> str:
    if "dataset" in cache_df.columns:
        resolved = str(cache_df["dataset"].iloc[0])
        if resolved in TARGET_DATASETS:
            return requested
    return requested


def main() -> None:
    args = parse_args()
    datasets = set(args.datasets.split(","))
    results_dir = RESULTS_DIR
    cache_out = results_dir / "cache"
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data_root = CFG_ROOT / "data"

    oracle_cache_dirs = [
        CFG_ROOT / "results" / "sparsity_phase_diagram" / "cache",
        CFG_ROOT / "results" / "oracle_sparsity_validation" / "cache",
    ]
    configs = discover_configs(oracle_cache_dirs, datasets)
    if args.max_configs:
        configs = configs[: args.max_configs]

    print(f"Configs: {len(configs)} | Device: {device} | Datasets: {datasets}")

    comparison_rows = []
    sig_inputs: list[dict] = []

    for cache_path, meta in tqdm(configs, desc="Ditto configs"):
        ditto_cache = cache_out / cache_path.name.replace("_client_level.csv", "_ditto.csv")
        oracle_df = pd.read_csv(cache_path)
        ds_name = meta["dataset_requested"]

        if ds_name not in dataset_cache:
            dataset_cache[ds_name] = load_registered_dataset(
                ds_name,
                data_root,
                max_records=args.ptbxl_max_records if ds_name == "ptbxl" else None,
            )

        if not args.skip_training and not ditto_cache.exists():
            ditto_df = run_ditto_for_config(
                dataset_name=ds_name,
                seed=meta["seed"],
                dirichlet_alpha=meta["dirichlet_alpha"],
                num_clients=meta["num_clients"],
                device=device,
                data_root=data_root,
                max_records=args.ptbxl_max_records if ds_name == "ptbxl" else None,
                oracle_cache_df=oracle_df,
                preloaded=dataset_cache[ds_name],
            )
            ditto_df.to_csv(ditto_cache, index=False)
        elif ditto_cache.exists():
            ditto_df = pd.read_csv(ditto_cache)
        else:
            print(f"Missing ditto cache: {ditto_cache}")
            continue

        resolved = str(ditto_df["dataset"].iloc[0])
        for lam in LAMBDAS:
            utils = utilities_for_config(ditto_df, oracle_df, lam, meta["seed"])
            random_res = evaluate_policy(oracle_df, lam, "RANDOM", meta["seed"])
            p5_res = evaluate_policy(oracle_df, lam, "PERSONALIZE_ALL", meta["seed"])
            comparison_rows.append(
                {
                    "dataset": resolved,
                    "dataset_requested": ds_name,
                    "seed": meta["seed"],
                    "dirichlet_alpha": meta["dirichlet_alpha"],
                    "num_clients": meta["num_clients"],
                    "lambda": lam,
                    "oracle_utility": utils["ORACLE"]["mean_utility"],
                    "best_uniform_utility": utils["BEST_UNIFORM"]["mean_utility"],
                    "global_only_utility": utils["GLOBAL_ONLY"]["mean_utility"],
                    "ditto_utility": utils["DITTO"]["mean_utility"],
                    "random_utility": random_res["mean_utility"],
                    "personalize_all_utility": p5_res["mean_utility"],
                    "best_uniform_level": utils["BEST_UNIFORM"]["chosen_levels"][0]
                    if utils["BEST_UNIFORM"]["chosen_levels"]
                    else "",
                }
            )
            sig_inputs.append(
                {
                    "oracle_utils": utils["ORACLE"]["client_utilities"],
                    "policy_utils": {
                        "DITTO": utils["DITTO"]["client_utilities"],
                        "BEST_UNIFORM": utils["BEST_UNIFORM"]["client_utilities"],
                        "GLOBAL_ONLY": utils["GLOBAL_ONLY"]["client_utilities"],
                        "RANDOM": random_res["client_utilities"],
                        "PERSONALIZE_ALL": p5_res["client_utilities"],
                    },
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(results_dir / "ditto_vs_oracle.csv", index=False)

    regret_df = build_ditto_vs_oracle_rows(comparison_df)
    # Add random and personalize-all from comparison_df
    extra = []
    for _, r in comparison_df.iterrows():
        oracle_u = r["oracle_utility"]
        for pol, col in [("RANDOM", "random_utility"), ("PERSONALIZE_ALL", "personalize_all_utility")]:
            pu = r[col]
            regret = oracle_u - pu
            rel = regret / max(abs(oracle_u), 0.05, 1e-8)
            extra.append(
                {
                    **{k: r[k] for k in [
                        "dataset", "dataset_requested", "seed", "dirichlet_alpha",
                        "num_clients", "lambda",
                    ]},
                    "policy": pol,
                    "policy_utility": pu,
                    "oracle_utility": oracle_u,
                    "regret": regret,
                    "relative_regret": rel,
                    "oracle_wins": regret > 1e-8,
                }
            )
    regret_df = pd.concat([regret_df, pd.DataFrame(extra)], ignore_index=True)
    regret_df.to_csv(results_dir / "ditto_regret_details.csv", index=False)

    import numpy as np

    sig_records = []
    for policy in ["DITTO", "BEST_UNIFORM", "GLOBAL_ONLY", "RANDOM", "PERSONALIZE_ALL"]:
        pvals, dzs = [], []
        for item in sig_inputs:
            o_util = item["oracle_utils"]
            p_util = item["policy_utils"][policy]
            _, p = paired_oracle_test(o_util, p_util)
            pvals.append(p)
            diff = np.array(o_util) - np.array(p_util)
            dzs.append(
                float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-8)) if len(diff) > 1 else 0.0
            )
        if pvals:
            sig_records.append(
                {
                    "policy": policy,
                    "sig": int(np.sum(np.array(pvals) < 0.05)),
                    "n": len(pvals),
                    "frac": float(np.mean(np.array(pvals) < 0.05)),
                    "median_p": float(np.median(pvals)),
                    "median_dz": float(np.median(dzs)),
                }
            )

    write_regret_summary_md(
        regret_df,
        comparison_df,
        sig_records,
        results_dir / "ditto_regret_summary.md",
    )

    pos = positioning_analysis(comparison_df)
    write_positioning_report_md(pos, comparison_df, results_dir / "ditto_positioning_report.md")
    plot_ditto_positioning(comparison_df, fig_dir)

    print(f"\n[OK] {results_dir / 'ditto_vs_oracle.csv'}")
    print(f"[OK] {results_dir / 'ditto_regret_summary.md'}")
    print(f"[OK] {results_dir / 'ditto_positioning_report.md'}")
    print(f"[OK] figures in {fig_dir}")
    print(f"Positioning: closest baseline = {pos['closest_baseline']}")


if __name__ == "__main__":
    main()
