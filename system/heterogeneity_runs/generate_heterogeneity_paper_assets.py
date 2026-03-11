from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "Cifar10_FedAvg_heterogeneity_summary.csv"
FIG_PNG_PATH = ROOT / "heterogeneity_paper_figure.png"
FIG_PDF_PATH = ROOT / "heterogeneity_paper_figure.pdf"
TABLE_TEX_PATH = ROOT / "heterogeneity_paper_table.tex"

LEVEL_ORDER = ["mild", "moderate", "severe"]
METHOD_ORDER = ["fu", "fedau", "fedcsa", "fedosd", "retrain"]
METHOD_LABELS = {
    "fu": "FU",
    "fedau": "FedAU",
    "fedcsa": "FedCSA",
    "fedosd": "FedOSD",
    "retrain": "Retrain",
}
METHOD_COLORS = {
    "fu": "#1f3b73",
    "fedau": "#2a9d8f",
    "fedcsa": "#8c6d1f",
    "fedosd": "#b35c1e",
    "retrain": "#6c757d",
}


def prepare_dataframe():
    df = pd.read_csv(SUMMARY_PATH)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["display_name"] = df["method"].map(METHOD_LABELS)
    df["privacy_auc_after"] = df["mia_post_auc"].fillna(df["final_mia_auc"])
    df["attack_success_after"] = df["backdoor_post_acc"].fillna(df["final_backdoor_acc"])
    return df.sort_values(["level", "method"]).reset_index(drop=True)


def make_figure(df):
    x = np.arange(len(LEVEL_ORDER))
    width = 0.16
    offsets = np.linspace(-2, 2, len(METHOD_ORDER)) * width

    metrics = [
        ("final_avg_acc", "Final Avg Accuracy", True),
        ("final_target_acc", "Target Client Accuracy After Unlearning", False),
        ("privacy_auc_after", "Post-Unlearning MIA AUC", False),
        ("attack_success_after", "Post-Unlearning Backdoor Accuracy", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, higher_is_better) in zip(axes, metrics):
        for offset, method in zip(offsets, METHOD_ORDER):
            sub = df[df["method"] == method].sort_values("level")
            ax.bar(
                x + offset,
                sub[metric].values,
                width=width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["Mild", "Moderate", "Severe"])
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if higher_is_better:
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Lower is better")
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Comparison Under Mild, Moderate, and Severe Heterogeneity", fontsize=13, y=1.07)
    fig.savefig(FIG_PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def make_latex_table(df):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Results under different heterogeneity levels on CIFAR-10. Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, while higher average accuracy indicates better utility retention.}",
        "\\label{tab:heterogeneity_results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Level & Method & Final Avg. Acc. $\\uparrow$ & Target Acc. $\\downarrow$ & Retain Avg. Acc. $\\uparrow$ & Post MIA AUC $\\downarrow$ & Post Backdoor Acc. $\\downarrow$ \\\\",
        "\\midrule",
    ]

    for level in LEVEL_ORDER:
        sub = df[df["level"] == level].sort_values("method")
        for idx, (_, row) in enumerate(sub.iterrows()):
            level_label = level.capitalize() if idx == 0 else ""
            lines.append(
                f"{level_label} & {METHOD_LABELS[row['method']]} & "
                f"{row['final_avg_acc']:.4f} & "
                f"{row['final_target_acc']:.4f} & "
                f"{row['final_retain_avg_acc']:.4f} & "
                f"{row['privacy_auc_after']:.4f} & "
                f"{row['attack_success_after']:.4f} \\\\"
            )
        if level != LEVEL_ORDER[-1]:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    TABLE_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = prepare_dataframe()
    make_figure(df)
    make_latex_table(df)
    print(f"Figure written to: {FIG_PNG_PATH}")
    print(f"Figure written to: {FIG_PDF_PATH}")
    print(f"LaTeX table written to: {TABLE_TEX_PATH}")


if __name__ == "__main__":
    main()
