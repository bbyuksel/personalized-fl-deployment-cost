"""Regenerate UBMK final figures: Fig1 as 2x2, Fig2 with large fonts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PHASE_CSV = ROOT / "results" / "sparsity_phase_diagram" / "phase_diagram_results.csv"
REGRET_CSV = ROOT / "results" / "final_validation" / "regret_table.csv"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
    }
)

NAME_MAP = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ptbxl": "PTB-XL"}
DATASETS = ["CIFAR-10", "CIFAR-100", "PTB-XL"]
POLICIES = ["RANDOM", "BEST_UNIFORM", "GLOBAL_ONLY", "PERSONALIZE_ALL"]
COLORS = {
    "RANDOM": "#e74c3c",
    "BEST_UNIFORM": "#f39c12",
    "PERSONALIZE_ALL": "#9b59b6",
    "GLOBAL_ONLY": "#3498db",
}


def _agg_phase() -> tuple[pd.DataFrame, list, list, list]:
    df = pd.read_csv(PHASE_CSV)
    df["dataset_label"] = df["dataset"].map(NAME_MAP).fillna(df["dataset"])
    present = [d for d in DATASETS if d in set(df["dataset_label"])]
    alphas = sorted(df["dirichlet_alpha"].unique())
    lambdas = sorted(df["lambda"].unique())
    agg = df.groupby(["dataset_label", "dirichlet_alpha", "lambda"], as_index=False).mean(
        numeric_only=True
    )
    agg["uniform_gap_pct"] = 100 * agg["oracle_vs_uniform_rel_gap"]
    agg["p5_penalty"] = 100 * agg["oracle_vs_personalize_all_rel_gap"]
    return agg, present, alphas, lambdas


def _heat_ax(ax, agg, ds, alphas, lambdas, value_col, cmap, title, as_percent=False, fmt=".1f", cbar=True):
    sub = agg[agg["dataset_label"] == ds]
    pivot = sub.pivot_table(index="lambda", columns="dirichlet_alpha", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=lambdas, columns=alphas)
    if as_percent:
        annot = pivot.map(lambda x: f"{100 * x:.0f}%" if pd.notna(x) else "")
        vmin, vmax = 0, 1
    else:
        annot = pivot.map(lambda x: format(x, fmt) if pd.notna(x) else "")
        vmin = vmax = None
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt="",
        annot_kws={"size": 9},
        cbar=cbar,
        linewidths=0.4,
    )
    ax.set_title(title, fontsize=13, pad=4)
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel(r"$\lambda$", fontsize=12)
    ax.tick_params(labelsize=10)


def make_fig1() -> None:
    """Four panels as 2x2: each panel averages/overlays by showing CIFAR-10 as canonical + note?
    Better: each panel = mean over datasets OR pick one dataset strip.
    Paper needs all three datasets. Use 2x2 grid where each cell is a 1x3 mini-row is too wide.
    Compromise: 2x2 panels, each panel shows three datasets as columns inside via Facet-like subgrids.
    Simpler: 2 rows x 2 cols of multi-dataset heatmaps (each panel is 1x3 small heatmaps).
    """
    agg, present, alphas, lambdas = _agg_phase()
    specs = [
        ("p0_ratio", "Blues", True, ".0f", "(a) Oracle mass on $P_0$", True),
        ("deep_ratio", "Oranges", True, ".0f", "(b) Oracle mass on $P_3$–$P_5$", True),
        ("uniform_gap_pct", "YlOrRd", False, ".1f", "(c) Oracle gain over uniform (%)", False),
        ("p5_penalty", "Purples", False, ".0f", "(d) Penalty of $P_5$-for-all (%)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    for ax, (col, cmap, as_pct, fmt, title, _) in zip(axes.ravel(), specs):
        # nested 1x3 inside each panel
        ax.set_axis_off()
        # Build nested gridspec-like by dividing the axes bbox
        # Cleaner approach: use GridSpec 2x2 with each cell further as 1x3
        pass
    plt.close(fig)

    from matplotlib import gridspec

    fig = plt.figure(figsize=(12.0, 8.4))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)
    for oi, (col, cmap, as_pct, fmt, title, _) in enumerate(specs):
        inner = gridspec.GridSpecFromSubplotSpec(1, len(present), subplot_spec=outer[oi], wspace=0.18)
        for j, ds in enumerate(present):
            ax = fig.add_subplot(inner[j])
            _heat_ax(
                ax,
                agg,
                ds,
                alphas,
                lambdas,
                col,
                cmap,
                ds if j > 0 else f"{title}\n{ds}",
                as_percent=as_pct,
                fmt=fmt,
                cbar=(j == len(present) - 1),
            )
            if j > 0:
                ax.set_ylabel("")
            if j == 0 and oi in (0, 1):
                # keep ylabel
                pass
    out = PAPER / "figures" / "fig_phase_grid.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # Also write individual panels still used? final tex will use fig_phase_grid.png only.


def make_fig2() -> None:
    rdf = pd.read_csv(REGRET_CSV)
    long = rdf.melt(
        id_vars=["dataset", "dirichlet_alpha", "lambda", "seed"],
        value_vars=POLICIES,
        var_name="policy",
        value_name="regret",
    )
    long["dataset_label"] = long["dataset"].map(NAME_MAP).fillna(long["dataset"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    agg_ds = long.groupby(["dataset_label", "policy"], as_index=False)["regret"].mean()
    order_ds = [d for d in DATASETS if d in set(agg_ds["dataset_label"])]
    sns.barplot(
        data=agg_ds,
        x="dataset_label",
        y="regret",
        hue="policy",
        order=order_ds,
        hue_order=POLICIES,
        palette=COLORS,
        ax=axes[0],
    )
    axes[0].set_xlabel("Dataset", fontsize=12)
    axes[0].set_ylabel("Mean regret", fontsize=12)
    axes[0].set_title("(a) By dataset", fontsize=13)
    axes[0].legend(title="Policy", fontsize=9, title_fontsize=10, loc="upper right")
    axes[0].tick_params(labelsize=11)

    agg_l = long.groupby(["lambda", "policy"], as_index=False)["regret"].mean()
    for policy in POLICIES:
        psub = agg_l[agg_l["policy"] == policy].sort_values("lambda")
        axes[1].plot(
            psub["lambda"],
            psub["regret"],
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=policy,
            color=COLORS[policy],
        )
    axes[1].set_xlabel(r"$\lambda$", fontsize=12)
    axes[1].set_ylabel("Mean regret", fontsize=12)
    axes[1].set_title(r"(b) By $\lambda$", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(labelsize=11)
    fig.tight_layout()
    out = PAPER / "figures" / "fig_regret_panel.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    make_fig1()
    make_fig2()

