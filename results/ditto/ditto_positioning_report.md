# Ditto Positioning Report

## Key question

Does Ditto behave closer to Oracle, Best Uniform, or Global-only under utility U?

## Answer: **C) Global-only**

Evidence (mean absolute utility distance across all config–λ cells):

| Baseline | Mean |ΔU| vs Ditto |
|----------|-------------------|
| Oracle | 0.6481 |
| Best uniform | 0.6225 |
| Global-only | 0.6121 |

## Head-to-head win rates (Ditto utility > baseline)

- vs best uniform: **7.2%**
- vs global-only: **25.5%**
- vs oracle (ties allowed): **6.0%**

## Per-dataset (λ=0.3)

- **cifar10**: |ΔU| to uniform 0.3053, to oracle 0.3292
- **cifar100**: |ΔU| to uniform 0.2566, to oracle 0.2613
- **ptbxl**: |ΔU| to uniform 0.2687, to oracle 0.3170

## Interpretation

Ditto is a published PFL trainer (full local personalization with proximal regularization). Under Paper-1's utility U with P5-equivalent cost, positioning relative to the oracle ladder indicates whether 'how to personalize' (Ditto) aligns with the utility-optimal allocation identified by the oracle.