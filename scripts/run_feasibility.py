#!/usr/bin/env python3
"""
FedPBG Feasibility Study — main entry point.

Runs oracle personalization budget discovery across simulated federated clients.

Usage:
  python scripts/run_feasibility.py --quick          # synthetic data, fast smoke test
  python scripts/run_feasibility.py                  # PTB-XL (requires download)
  python scripts/run_feasibility.py --alpha 0.3      # alternate Dirichlet alpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running as script from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fedpbg.config import (  # noqa: E402
    CLIENT_TRAIN_RATIO,
    CLIENT_VAL_RATIO,
    DEFAULT_LAMBDA,
    DIRICHLET_ALPHA_OPTIONS,
    LAMBDA_VALUES,
    PERSONALIZATION_LEVELS,
    ExperimentConfig,
)
from fedpbg.data.federated_split import build_client_splits
from fedpbg.data.ptbxl import ECGDataset, build_ptbxl_arrays, build_synthetic_arrays
from fedpbg.federated.client_trainer import train_and_evaluate_level
from fedpbg.federated.fedavg import run_fedavg
from fedpbg.hardware.profiles import HARDWARE_PROFILES, assign_hardware_profiles
from fedpbg.metrics.cost import build_cost_record, normalize_costs_across_levels
from fedpbg.metrics.heterogeneity import heterogeneity_scores
from fedpbg.metrics.utility import build_oracle_table, compute_utility
from fedpbg.models.ecg_resnet1d import ECGResNet1D, estimate_forward_flops
from fedpbg.reporting.summary import write_feasibility_summary
from fedpbg.viz.plots import generate_all_plots


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="FedPBG feasibility study")
    parser.add_argument("--quick", action="store_true", help="Synthetic data, reduced scale")
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for non-IID split")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--global-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        quick=args.quick,
        dirichlet_alpha=args.alpha,
        num_clients=args.num_clients,
        random_seed=args.seed,
        device=args.device,
    )
    if args.global_rounds is not None:
        cfg.num_global_rounds = args.global_rounds
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.max_records is not None:
        cfg.max_records = args.max_records
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.results_dir:
        cfg.results_dir = Path(args.results_dir)

    if cfg.quick:
        cfg.num_global_rounds = min(cfg.num_global_rounds, 2)
        cfg.local_epochs = min(cfg.local_epochs, 2)
        cfg.max_records = cfg.max_records or 800

    return cfg


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FedPBG Feasibility Study — Oracle Budget Discovery")
    print("=" * 60)

    # --- Data ---
    lead_indices = (cfg.lead_index,)
    if cfg.quick:
        print("[Data] Using synthetic ECG-like signals (--quick)")
        X, y, _meta = build_synthetic_arrays(
            n_samples=cfg.max_records or 2000,
            n_leads=cfg.input_channels,
            target_length=cfg.signal_length,
            random_seed=cfg.random_seed,
        )
    else:
        print(f"[Data] Loading PTB-XL from {cfg.data_dir}")
        X, y, _meta = build_ptbxl_arrays(
            cfg.data_dir,
            lead_indices=lead_indices,
            target_length=cfg.signal_length,
            max_records=cfg.max_records,
        )

    dataset = ECGDataset(X, y)
    print(f"[Data] {len(dataset)} samples, class balance: {np.bincount(y)}")

    clients = build_client_splits(
        y,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        train_ratio=CLIENT_TRAIN_RATIO,
        val_ratio=CLIENT_VAL_RATIO,
        random_seed=cfg.random_seed,
    )
    print(f"[Split] {len(clients)} clients, Dirichlet alpha={cfg.dirichlet_alpha}")

    hardware_assignments = assign_hardware_profiles(cfg.num_clients, cfg.random_seed)

    # --- Global model (FedAvg) ---
    model = ECGResNet1D(
        in_channels=cfg.input_channels,
        num_classes=cfg.num_classes,
    ).to(device)

    train_indices = {c.client_id: c.train_idx for c in clients}
    print(f"[FedAvg] {cfg.num_global_rounds} rounds, {cfg.local_epochs} local epochs")
    global_model = run_fedavg(
        model,
        dataset,
        train_indices,
        num_rounds=cfg.num_global_rounds,
        local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        device=device,
        random_seed=cfg.random_seed,
    )

    # --- Heterogeneity scores (global encoder) ---
    het_map = heterogeneity_scores(
        global_model,
        dataset,
        {c.client_id: c.train_idx for c in clients},
        device,
        batch_size=cfg.batch_size,
    )

    flops_est = estimate_forward_flops(
        global_model, cfg.signal_length, batch_size=1
    )

    # --- Per-client, per-level oracle search ---
    all_rows = []
    for client in clients:
        cid = client.client_id
        hw_type = hardware_assignments[cid]
        hw_profile = HARDWARE_PROFILES[hw_type]

        level_costs_raw = []
        level_results = []

        for level in PERSONALIZATION_LEVELS:
            metrics, cost_info = train_and_evaluate_level(
                global_model=global_model,
                level=level,
                dataset=dataset,
                train_idx=client.train_idx,
                val_idx=client.val_idx,
                test_idx=client.test_idx,
                local_epochs=cfg.local_epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                device=device,
                flops_estimate=flops_est,
                memory_scale=cfg.signal_length,
            )
            raw = build_cost_record(
                trainable_params=cost_info["trainable_params"],
                flops=cost_info["flops"],
                train_time_sec=cost_info["train_time_sec"],
                comm_bytes=cost_info["comm_bytes"],
                batch_size=cfg.batch_size,
                signal_length=cost_info["memory_scale"],
                hardware=hw_profile,
                normalized_cost=0.0,  # filled after normalization
            )
            level_costs_raw.append(raw.raw_cost)
            level_results.append((level, metrics, raw))

        norm_costs = normalize_costs_across_levels(level_costs_raw)

        for (level, metrics, raw), norm_c in zip(level_results, norm_costs):
            all_rows.append(
                {
                    "client_id": cid,
                    "level": level,
                    "hardware": hw_type.value,
                    "heterogeneity": het_map[cid],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "auroc": metrics["auroc"],
                    "trainable_params": raw.trainable_params,
                    "flops": raw.flops,
                    "train_time_sec": raw.train_time_sec,
                    "comm_bytes": raw.comm_bytes,
                    "memory_proxy_mb": raw.memory_proxy_mb,
                    "hardware_multiplier": raw.hardware_multiplier,
                    "raw_cost": raw.raw_cost,
                    "normalized_cost": norm_c,
                    "utility_lam_0.1": compute_utility(metrics["macro_f1"], norm_c, 0.1),
                    "utility_lam_0.3": compute_utility(metrics["macro_f1"], norm_c, 0.3),
                    "utility_lam_0.5": compute_utility(metrics["macro_f1"], norm_c, 0.5),
                    "utility_lam_1.0": compute_utility(metrics["macro_f1"], norm_c, 1.0),
                    "dirichlet_alpha": cfg.dirichlet_alpha,
                }
            )
        print(f"  Client {cid:02d} ({hw_type.value}): done")

    client_df = pd.DataFrame(all_rows)
    client_csv = cfg.results_dir / "client_level_results.csv"
    client_df.to_csv(client_csv, index=False)
    print(f"[Output] {client_csv}")

    oracle_df = build_oracle_table(client_df, LAMBDA_VALUES)
    oracle_csv = cfg.results_dir / "oracle_budget_results.csv"
    oracle_df.to_csv(oracle_csv, index=False)
    print(f"[Output] {oracle_csv}")

    generate_all_plots(client_df, oracle_df, cfg.results_dir, DEFAULT_LAMBDA)
    print(f"[Output] Figures saved to {cfg.results_dir}")

    summary_path = cfg.results_dir / "feasibility_summary.md"
    write_feasibility_summary(
        client_df,
        oracle_df,
        summary_path,
        default_lambda=DEFAULT_LAMBDA,
        dirichlet_alpha=cfg.dirichlet_alpha,
        num_clients=cfg.num_clients,
    )
    print(f"[Output] {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
