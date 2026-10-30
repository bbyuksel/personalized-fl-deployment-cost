# Ditto Regret Summary

Ditto evaluated under $U = \mathrm{Macro\text{-}F1} - \lambda \cdot \mathrm{Cost}$ (P5-normalized cost from oracle cache; Ditto trains full model).

## Relative regret vs oracle (all λ cells)

| Policy | Mean rel. regret (%) | Std (%) | 95% boot CI | Oracle win rate (%) |
|--------|---------------------|---------|-------------|---------------------|
| DITTO | 454.9 | 875.8 | [380.7, 534.7] | 94.0 |
| BEST_UNIFORM | 6.4 | 8.0 | [5.7, 7.1] | 80.7 |
| GLOBAL_ONLY | 27.9 | 29.7 | [25.3, 30.6] | 81.7 |

## Wilcoxon (oracle > policy, client-level per cell)

| Policy | Sig. cells | Fraction | Median p | Median d_z |
|--------|------------|----------|----------|------------|
| DITTO | 435/486 | 89.5% | 9.537e-07 | 2.810 |
| BEST_UNIFORM | 342/486 | 70.4% | 0.002531 | 0.486 |
| GLOBAL_ONLY | 339/486 | 69.8% | 0.001673 | 0.624 |
| RANDOM | 480/486 | 98.8% | 0.0001437 | 1.077 |
| PERSONALIZE_ALL | 472/486 | 97.1% | 9.537e-07 | 5.758 |

## Positioning preview

- Closest baseline by mean |ΔU|: **global_only**
- Ditto beats best uniform: 7.2% of cells
- Ditto beats global-only: 25.5% of cells