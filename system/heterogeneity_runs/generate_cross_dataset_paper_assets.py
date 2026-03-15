import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = ROOT / "FedAvg_cross_dataset_comparison_multidataset_summary.csv"
DEFAULT_FIG_PNG_PATH = ROOT / "cross_dataset_paper_figure.png"
DEFAULT_FIG_PDF_PATH = ROOT / "cross_dataset_paper_figure.pdf"
DEFAULT_TABLE_TEX_PATH = ROOT / "cross_dataset_paper_table.tex"
DEFAULT_ANALYSIS_MD_PATH = ROOT / "cross_dataset_paper_analysis.md"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-ready cross-dataset assets from a summary CSV.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--figure-png", type=Path, default=DEFAULT_FIG_PNG_PATH)
    parser.add_argument("--figure-pdf", type=Path, default=DEFAULT_FIG_PDF_PATH)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_TEX_PATH)
    parser.add_argument("--analysis-md", type=Path, default=DEFAULT_ANALYSIS_MD_PATH)
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Cross-dataset federated unlearning results under severe Dirichlet heterogeneity. "
            "Higher final average accuracy indicates better utility retention, while lower target "
            "accuracy, post-unlearning MIA AUC, and post-backdoor accuracy indicate stronger forgetting."
        ),
    )
    return parser.parse_args()


def prepare_dataframe(summary_path):
    df = pd.read_csv(summary_path)
    df = df[df["status"].astype(str).str.lower() == "complete"].copy()
    if df.empty:
        raise ValueError(f"No complete rows found in {summary_path}.")

    if "dataset_display" not in df.columns:
        df["dataset_display"] = df.get("dataset_base", df["dataset"])

    level_suffix = df["level"].fillna("").astype(str)
    if df["level"].nunique(dropna=True) > 1:
        df["group_label"] = df["dataset_display"] + " (" + level_suffix.str.capitalize() + ")"
    else:
        df["group_label"] = df["dataset_display"]

    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["privacy_auc_after"] = df["mia_post_auc"].fillna(df["final_mia_auc"])
    df["attack_success_after"] = df["backdoor_post_acc"].fillna(df["final_backdoor_acc"])
    return df.sort_values(["group_label", "method"]).reset_index(drop=True)


def make_figure(df, fig_png_path, fig_pdf_path):
    groups = list(dict.fromkeys(df["group_label"].tolist()))
    x = np.arange(len(groups))
    width = 0.16
    offsets = np.linspace(-2, 2, len(METHOD_ORDER)) * width

    metrics = [
        ("final_avg_acc", "Final Avg Accuracy", True),
        ("final_target_acc", "Target Client Accuracy", False),
        ("privacy_auc_after", "Post-Unlearning MIA AUC", False),
        ("attack_success_after", "Post-Backdoor Accuracy", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, metric_title, higher_is_better) in zip(axes, metrics):
        for offset, method in zip(offsets, METHOD_ORDER):
            sub = (
                df[df["method"] == method]
                .set_index("group_label")
                .reindex(groups)
            )
            ax.bar(
                x + offset,
                sub[metric].astype(float).values,
                width=width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(metric_title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Higher is better" if higher_is_better else "Lower is better")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.savefig(fig_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_latex_table(df, table_tex_path, caption):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:cross_dataset_results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Method & Final Avg. Acc. $\\uparrow$ & Target Acc. $\\downarrow$ & Retain Avg. Acc. $\\uparrow$ & Post MIA AUC $\\downarrow$ & Post Backdoor Acc. $\\downarrow$ \\\\",
        "\\midrule",
    ]

    groups = list(dict.fromkeys(df["group_label"].tolist()))
    for group_idx, group in enumerate(groups):
        sub = df[df["group_label"] == group].sort_values("method")
        for row_idx, (_, row) in enumerate(sub.iterrows()):
            group_label = group if row_idx == 0 else ""
            lines.append(
                f"{group_label} & {METHOD_LABELS[row['method']]} & "
                f"{float(row['final_avg_acc']):.4f} & "
                f"{float(row['final_target_acc']):.4f} & "
                f"{float(row['final_retain_avg_acc']):.4f} & "
                f"{float(row['privacy_auc_after']):.4f} & "
                f"{float(row['attack_success_after']):.4f} \\\\"
            )
        if group_idx != len(groups) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    table_tex_path.write_text("\n".join(lines), encoding="utf-8")


def make_analysis(df, analysis_md_path):
    groups = list(dict.fromkeys(df["group_label"].tolist()))
    lines = [
        "# Cross-Dataset Paper Analysis",
        "",
        "## Key observations",
        "",
    ]

    for group in groups:
        sub = df[df["group_label"] == group].copy()
        best_utility = sub.sort_values(["final_avg_acc", "final_target_acc"], ascending=[False, True]).iloc[0]
        best_forgetting = sub.sort_values(["final_target_acc", "final_avg_acc"], ascending=[True, False]).iloc[0]
        fu_row = sub[sub["method"] == "fu"]
        lines.append(f"### {group}")
        lines.append(
            f"- Best utility: `{METHOD_LABELS[best_utility['method']]}` with `final_avg_acc={float(best_utility['final_avg_acc']):.4f}`."
        )
        lines.append(
            f"- Strongest forgetting: `{METHOD_LABELS[best_forgetting['method']]}` with `final_target_acc={float(best_forgetting['final_target_acc']):.4f}`."
        )
        if not fu_row.empty:
            fu = fu_row.iloc[0]
            lines.append(
                f"- FU achieves `final_avg_acc={float(fu['final_avg_acc']):.4f}` and `final_target_acc={float(fu['final_target_acc']):.4f}`."
            )
        lines.append("")

    analysis_md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    df = prepare_dataframe(args.summary)
    make_figure(df, args.figure_png, args.figure_pdf)
    make_latex_table(df, args.table, args.caption)
    make_analysis(df, args.analysis_md)
    print(f"Figure written to: {args.figure_png}")
    print(f"Figure written to: {args.figure_pdf}")
    print(f"LaTeX table written to: {args.table}")
    print(f"Analysis written to: {args.analysis_md}")


if __name__ == "__main__":
    main()
