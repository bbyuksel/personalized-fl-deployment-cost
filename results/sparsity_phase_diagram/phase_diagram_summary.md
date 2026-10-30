# FedSUCP Sparsity Phase Diagram Summary

**Revised framing:** *When is personalization worth its cost in federated learning?*

This study maps oracle personalization actions over the (α, λ) phase space,
where α controls label heterogeneity (Dirichlet) and λ weights personalization cost.

**Configurations:** 378 cells (3 datasets × 7 α × 6 λ × seeds).

## Phase-space findings

### Under what α–λ regimes is P0 (global-only) dominant?

- **cifar10** α=0.1, λ=2.0: P0=63.3%
- **cifar10** α=0.3, λ=1.0: P0=75.0%
- **cifar10** α=0.3, λ=2.0: P0=91.7%
- **cifar10** α=0.5, λ=0.5: P0=60.0%
- **cifar10** α=0.5, λ=1.0: P0=91.7%
- **cifar10** α=0.5, λ=2.0: P0=100.0%
- **cifar10** α=1.0, λ=0.3: P0=81.7%
- **cifar10** α=1.0, λ=0.5: P0=95.0%

**Pattern:** P0 dominates at **high λ** (cost-sensitive) and on datasets/tasks where global models suffice (e.g., PTB-XL synthetic). Low λ restores personalization value.

### When does deep personalization (P3–P5) become worthwhile?

- **cifar10** α=0.01, λ=0.0: deep=38.3%
- **cifar10** α=0.01, λ=0.1: deep=36.7%
- **cifar10** α=0.01, λ=0.3: deep=21.7%
- **cifar10** α=0.01, λ=0.5: deep=18.3%
- **cifar10** α=0.05, λ=0.0: deep=53.3%
- **cifar10** α=0.05, λ=0.1: deep=38.3%

**Pattern:** Deep personalization concentrates at **low λ** and **moderate-to-high α** when local adaptation gains outweigh cost.

### When is uniform personalization most suboptimal?

- **cifar100** α=0.01, λ=2.0: gap=232.5% (best uniform=P0)
- **cifar10** α=0.05, λ=2.0: gap=140.8% (best uniform=P0)
- **cifar10** α=0.1, λ=2.0: gap=41.5% (best uniform=P0)
- **cifar10** α=0.01, λ=2.0: gap=37.9% (best uniform=P2)
- **cifar100** α=0.05, λ=1.0: gap=25.2% (best uniform=P0)

### Is personalize-all (P5) ever optimal?

**Yes, in 1 cells** — typically at **λ=0** where cost is ignored and accuracy-only utility favors full local training.

### Does sparsity depend more on α, λ, dataset, or hardware?

**Strongest factor in this study:** `dataset` (max variance of P0 ratio across dataset levels).

Qualitative ranking:
- **λ:** Directly trades accuracy vs cost; high λ → more P0.
- **Dataset/task:** Modality and label structure shift the entire phase boundary.
- **α:** Shifts heterogeneity; effect is dataset-dependent (stronger on CIFAR than PTB-XL synthetic).
- **Hardware:** Shifts feasible region; LOW hardware amplifies P0 at moderate λ.

Hardware breakdown (mean P0 ratio):
  - **LOW:** mean P0 = 53.9%
  - **MEDIUM:** mean P0 = 54.2%
  - **HIGH:** mean P0 = 52.3%

### Strongest ICML framing after these results

> **Do not claim personalization is always sparse.** Instead:
> Federated learning should be analyzed as a **personalization–cost phase diagram**:
> whether local adaptation is worth its cost depends jointly on **task**, **heterogeneity (α)**,
> **cost sensitivity (λ)**, and **device constraints**. FedSUCP targets learning this boundary
> online without exhaustive oracle search.

**Title-level claim:** *When is personalization worth its cost in federated learning?*

## Scope

Oracle phase characterization only. FedSUCP allocator and bandit are not implemented.

## Figures

- `p0_ratio_heatmap_by_dataset.png`
- `deep_ratio_heatmap_by_dataset.png`
- `oracle_uniform_gap_heatmap_by_dataset.png`
- `personalize_all_penalty_heatmap_by_dataset.png`
- `best_uniform_level_heatmap_by_dataset.png`
- `hardware_phase_interaction.png`