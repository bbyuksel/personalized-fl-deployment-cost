# Oracle Regret Study — Final Validation Summary

**Paper title (working):** *Phase Transitions in Personalized Federated Learning:
A Utility-Based Characterization of the Personalization–Cost Tradeoff*

**Configurations analyzed:** 99 (dataset × α × seed), 
**4 datasets**, 
**6 λ values**.

> Conservative assessment. Claims below are tied directly to observed regret statistics.

## Global regret vs ORACLE (mean across all cells)

| Policy | Mean regret | Mean relative regret (%) | Oracle win rate |
|--------|-------------|--------------------------|-----------------|
| RANDOM | 0.3181 | 189.3% | 100% |
| BEST_UNIFORM | 0.0273 | 6.5% | 82% |
| PERSONALIZE_ALL | 0.6407 | 409.0% | 99% |
| GLOBAL_ONLY | 0.1223 | 28.2% | 83% |

## Scientific Questions

### 1. Is Oracle meaningfully better than Random?

**Strong** — mean relative regret 189.3%, oracle wins in 100% of configurations.
 Paired Wilcoxon (client-level): significant in 99% of configs (median p=0.000).

### 2. Is Oracle meaningfully better than Uniform?

**Moderate** — mean relative regret 6.5%, oracle wins in 82% of configurations.
 Paired Wilcoxon (client-level): significant in 72% of configs (median p=0.002).

### 3. Is Oracle meaningfully better than Personalize-All?

**Strong** — mean relative regret 409.0%, oracle wins in 99% of configurations.
 Paired Wilcoxon (client-level): significant in 98% of configs (median p=0.000).

### 4. Is Oracle meaningfully better than Global-Only?

**Strong** — mean relative regret 28.2%, oracle wins in 83% of configurations.
 Paired Wilcoxon (client-level): significant in 72% of configs (median p=0.001).

### 5. Are the observed personalization phase transitions robust?

**Insufficient evidence in this run** to confirm robust phase transitions; expand α/λ grid and seeds.

### 6. Strongest defensible scientific claim

> Under a utility objective U = Performance − λ·Cost, **fixed global policies** 
> (uniform personalization, personalize-all, global-only) incur **non-zero regret** 
> relative to per-client oracle allocation in most configurations; the magnitude and 
> which baseline is worst **depend on λ, dataset, and α** — supporting a 
> **phase-transition view** rather than a universal sparsity law.

### 7. Claims that should NOT be made

- Do **not** claim personalization is always sparse (CIFAR-100 at λ=0 shows deep personalization).
- Do **not** claim oracle is always far above random (random regret can be small when λ is high).
- Do **not** claim a new algorithm — this is **characterization + oracle upper bound** only.
- Do **not** claim cross-device hardware generalization without measured device costs.
- Do **not** overstate statistical power with few seeds or synthetic PTB-XL.

### 8. Would an ICML reviewer consider the evidence sufficient?

**Insufficient alone for ICML main track** — regret vs uniform is moderate but needs broader benchmarks, more seeds, and real (non-synthetic) PTB-XL. Suitable as strong motivation section; method paper still required.

## Per-dataset regret (BEST_UNIFORM baseline)

- **cifar10:** mean regret=0.0250, rel=6.4%, win rate=82%
- **cifar100:** mean regret=0.0051, rel=5.5%, win rate=64%
- **cifar10_fallback:** mean regret=0.0267, rel=7.1%, win rate=91%
- **ptbxl:** mean regret=0.0515, rel=7.2%, win rate=97%

## Scope

Oracle regret study only. No allocator, bandit, or RL implemented.