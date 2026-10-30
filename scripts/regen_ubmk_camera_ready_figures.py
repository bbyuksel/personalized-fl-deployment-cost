"""Camera-ready UBMK figures: publication labels, 300 dpi, readable fonts."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "figures"
PHASE_CSV = ROOT / "results" / "sparsity_phase_diagram" / "phase_diagram_results.csv"
REGRET_CSV = ROOT / "results" / "final_validation" / "regret_table.csv"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    }
)

NAME_MAP = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "ptbxl": "PTB-XL"}
DATASETS = ["CIFAR-10", "CIFAR-100", "PTB-XL"]
POLICY_LABELS = {
    "RANDOM": "Random",
    "BEST_UNIFORM": "Best-uniform",
    "GLOBAL_ONLY": "Global-only",
    "PERSONALIZE_ALL": "Personalize-all",
}
POLICIES = list(POLICY_LABELS.keys())
COLORS = {
    "RANDOM": "#c0392b",
    "BEST_UNIFORM": "#d68910",
    "PERSONALIZE_ALL": "#7d3c98",
    "GLOBAL_ONLY": "#2874a6",
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


def _fmt_cell(x, as_percent: bool, fmt: str) -> str:
    """Short cell labels: ratios as integer %, large penalties as '12k'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    if as_percent:
        return f"{100 * float(x):.0f}"
    v = float(x)
    av = abs(v)
    if av >= 1000:
        return f"{v / 1000:.0f}k"
    if av >= 100:
        return f"{v:.0f}"
    return format(v, fmt)


def _heat_ax(
    ax,
    agg,
    ds,
    alphas,
    lambdas,
    value_col,
    cmap,
    title,
    as_percent=False,
    fmt=".0f",
    cbar=True,
    annotate=True,
):
    """Compact cell annotations so values do not overlap in narrow heatmap cells."""
    sub = agg[agg["dataset_label"] == ds]
    pivot = sub.pivot_table(index="lambda", columns="dirichlet_alpha", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=lambdas, columns=alphas)
    pivot = pivot.dropna(axis=1, how="all")
    annot_vals = pivot.map(lambda x: _fmt_cell(x, as_percent, fmt))
    plot_data = pivot
    if as_percent:
        vmin, vmax = 0.0, 1.0
        cbar_label = "%"
    else:
        vmin = vmax = None
        cbar_label = ""

    hm = sns.heatmap(
        plot_data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar=cbar,
        cbar_kws={"shrink": 0.80, "label": cbar_label, "pad": 0.02} if cbar else None,
        linewidths=0.10,
        linecolor="0.95",
        square=False,
    )

    if annotate:
        import matplotlib.colors as mcolors

        if as_percent:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        else:
            finite = plot_data.values[np.isfinite(plot_data.values)]
            norm = mcolors.Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
        cmap_obj = plt.get_cmap(cmap)

        # Scale annotation size with number of columns (narrower cells → smaller type)
        ncols = max(plot_data.shape[1], 1)
        # Keep labels strictly inside cells; prefer readability over large type
        base = 5.5 if ncols <= 4 else 4.5

        for y in range(plot_data.shape[0]):
            for x in range(plot_data.shape[1]):
                txt = annot_vals.iloc[y, x]
                if not txt:
                    continue
                val = plot_data.iloc[y, x]
                if not np.isfinite(val):
                    continue
                r, g, b, _ = cmap_obj(norm(val))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if lum < 0.55 else "black"
                n = len(txt)
                fsize = base if n <= 2 else (base - 0.4 if n == 3 else base - 0.9)
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=max(fsize, 3.5),
                    color=color,
                    clip_on=True,
                )
    ax.set_title(title, fontsize=11, pad=3)
    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel(r"$\lambda$", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    return hm


def make_fig1() -> None:
    """Four full-width rows (one metric each) so heatmap cells are wide enough for labels."""
    agg, present, alphas, lambdas = _agg_phase()
    specs = [
        ("p0_ratio", "Blues", True, ".0f", "(a) $P_0$ oracle mass (%)", True),
        ("deep_ratio", "Oranges", True, ".0f", "(b) Deep-action mass ($P_3$--$P_5$) (%)", True),
        ("uniform_gap_pct", "YlOrRd", False, ".0f", "(c) Oracle gain over best-uniform (%)", True),
        # Extreme magnitudes: colorbar only (avoids 4-digit overflow in cells)
        ("p5_penalty", "Purples", False, ".0f", "(d) $P_5$-for-all penalty (%)", False),
    ]
    fig = plt.figure(figsize=(13.5, 12.0))
    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)
    for oi, (col, cmap, as_pct, fmt, panel_title, do_annot) in enumerate(specs):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(present), subplot_spec=outer[oi], wspace=0.18, width_ratios=[1.15, 1.15, 0.85]
        )
        for j, ds in enumerate(present):
            ax = fig.add_subplot(inner[j])
            title = f"{panel_title} — {ds}" if j == 0 else ds
            _heat_ax(
                ax,
                agg,
                ds,
                alphas,
                lambdas,
                col,
                cmap,
                title,
                as_percent=as_pct,
                fmt=fmt,
                cbar=(j == len(present) - 1),
                annotate=do_annot,
            )
            if j > 0:
                ax.set_ylabel("")
    for ext in ("pdf", "png"):
        out = PAPER / f"fig_phase_grid.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white", pad_inches=0.05)
        print("wrote", out)
    plt.close(fig)


def make_fig2() -> None:
    rdf = pd.read_csv(REGRET_CSV)
    long = rdf.melt(
        id_vars=["dataset", "dirichlet_alpha", "lambda", "seed"],
        value_vars=POLICIES,
        var_name="policy",
        value_name="regret",
    )
    long["dataset_label"] = long["dataset"].map(NAME_MAP).fillna(long["dataset"])
    long["policy_label"] = long["policy"].map(POLICY_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    agg_ds = long.groupby(["dataset_label", "policy"], as_index=False)["regret"].mean()
    agg_ds["policy_label"] = agg_ds["policy"].map(POLICY_LABELS)
    order_ds = [d for d in DATASETS if d in set(agg_ds["dataset_label"])]
    hue_order = [POLICY_LABELS[p] for p in POLICIES]
    sns.barplot(
        data=agg_ds,
        x="dataset_label",
        y="regret",
        hue="policy_label",
        order=order_ds,
        hue_order=hue_order,
        palette=[COLORS[p] for p in POLICIES],
        ax=axes[0],
    )
    axes[0].set_xlabel("Dataset", fontsize=13)
    axes[0].set_ylabel("Mean regret", fontsize=13)
    axes[0].set_title("(a) By dataset", fontsize=15)
    axes[0].legend(title="Policy", fontsize=10, title_fontsize=11, loc="upper right", frameon=False)
    axes[0].tick_params(labelsize=11)

    agg_l = long.groupby(["lambda", "policy"], as_index=False)["regret"].mean()
    for policy in POLICIES:
        psub = agg_l[agg_l["policy"] == policy].sort_values("lambda")
        axes[1].plot(
            psub["lambda"],
            psub["regret"],
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
    axes[1].set_xlabel(r"Cost sensitivity $\lambda$", fontsize=13)
    axes[1].set_ylabel("Mean regret", fontsize=13)
    axes[1].set_title(r"(b) By $\lambda$", fontsize=15)
    axes[1].legend(fontsize=10, frameon=False)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(labelsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = PAPER / f"fig_regret_panel.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
