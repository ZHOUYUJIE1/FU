import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = ROOT / "Cifar10_FedAvg_client_count_severe_summary.csv"
DEFAULT_FIG_PNG_PATH = ROOT / "client_count_paper_figure.png"
DEFAULT_FIG_PDF_PATH = ROOT / "client_count_paper_figure.pdf"
DEFAULT_TABLE_TEX_PATH = ROOT / "client_count_paper_table.tex"
DEFAULT_ANALYSIS_MD_PATH = ROOT / "client_count_paper_analysis.md"

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
METRICS = [
    ("final_avg_acc", "Final Avg Accuracy", True),
    ("final_target_acc", "Target Client Accuracy", False),
    ("privacy_auc_after", "Post-Unlearning MIA AUC", False),
    ("attack_success_after", "Post-Unlearning Backdoor Accuracy", False),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figure/table assets from client-count comparison summary CSV."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--figure-png", type=Path, default=DEFAULT_FIG_PNG_PATH)
    parser.add_argument("--figure-pdf", type=Path, default=DEFAULT_FIG_PDF_PATH)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_TEX_PATH)
    parser.add_argument("--analysis-md", type=Path, default=DEFAULT_ANALYSIS_MD_PATH)
    parser.add_argument(
        "--title",
        type=str,
        default="Comparison Under Severe Heterogeneity with Varying Client Counts",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Results under severe Dirichlet heterogeneity on CIFAR-10 with different numbers of clients. "
            "Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy "
            "protection, while higher average and retain-client accuracy indicate better utility retention."
        ),
    )
    return parser.parse_args()


def _to_numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def _series_or_nan(df, column):
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index, dtype=float)


def prepare_dataframe(summary_path):
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    for col in [
        "client_count",
        "num_clients",
        "final_avg_acc",
        "final_target_acc",
        "final_retain_avg_acc",
        "mia_post_auc",
        "final_mia_auc",
        "backdoor_post_acc",
        "final_backdoor_acc",
    ]:
        if col in df.columns:
            df[col] = _to_numeric(df, col)

    if "client_count" not in df.columns or df["client_count"].isna().all():
        if "num_clients" in df.columns:
            df["client_count"] = df["num_clients"]
        else:
            raise ValueError("Summary CSV must contain either client_count or num_clients column.")

    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["display_name"] = df["method"].map(METHOD_LABELS)
    mia_post = _series_or_nan(df, "mia_post_auc")
    mia_final = _series_or_nan(df, "final_mia_auc")
    backdoor_post = _series_or_nan(df, "backdoor_post_acc")
    backdoor_final = _series_or_nan(df, "final_backdoor_acc")
    df["privacy_auc_after"] = mia_post.fillna(mia_final)
    df["attack_success_after"] = backdoor_post.fillna(backdoor_final)
    df = df[df["status"] == "complete"].copy()
    df["client_count"] = df["client_count"].astype(int)
    return df.sort_values(["client_count", "method"]).reset_index(drop=True)


def make_figure(df, fig_png_path, fig_pdf_path, title):
    client_counts = sorted(df["client_count"].unique().tolist())
    if not client_counts:
        raise ValueError("No complete rows found in summary; cannot plot figure.")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, metric_title, higher_is_better) in zip(axes, METRICS):
        for method in METHOD_ORDER:
            sub = df[df["method"] == method].sort_values("client_count")
            if sub.empty or sub[metric].isna().all():
                continue
            ax.plot(
                sub["client_count"].values,
                sub[metric].values,
                marker="o",
                linewidth=1.8,
                markersize=4.8,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_title(metric_title, fontsize=11)
        ax.set_xlabel("Number of Clients")
        ax.set_xticks(client_counts)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(bottom=0.0)
        if higher_is_better:
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Lower is better")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle(title, fontsize=13, y=1.07)
    fig.savefig(fig_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_latex_table(df, table_tex_path, caption):
    client_counts = sorted(df["client_count"].unique().tolist())
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:client_count_results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Client \\# & Method & Final Avg. Acc. $\\uparrow$ & Target Acc. $\\downarrow$ & Retain Avg. Acc. $\\uparrow$ & Post MIA AUC $\\downarrow$ & Post Backdoor Acc. $\\downarrow$ \\\\",
        "\\midrule",
    ]

    for idx, client_count in enumerate(client_counts):
        sub = df[df["client_count"] == client_count].sort_values("method")
        for row_idx, (_, row) in enumerate(sub.iterrows()):
            count_label = str(client_count) if row_idx == 0 else ""
            lines.append(
                f"{count_label} & {METHOD_LABELS[row['method']]} & "
                f"{row['final_avg_acc']:.4f} & "
                f"{row['final_target_acc']:.4f} & "
                f"{row['final_retain_avg_acc']:.4f} & "
                f"{row['privacy_auc_after']:.4f} & "
                f"{row['attack_success_after']:.4f} \\\\"
            )
        if idx != len(client_counts) - 1:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    table_tex_path.write_text("\n".join(lines), encoding="utf-8")


def make_analysis(df, analysis_md_path):
    client_counts = sorted(df["client_count"].unique().tolist())
    lines = ["# Client Count Comparison Analysis", ""]
    lines.append(f"- Available client counts: {client_counts}")
    lines.append("- Methods: " + ", ".join(METHOD_LABELS[m] for m in METHOD_ORDER))
    lines.append("")

    for metric, metric_title, higher_is_better in METRICS:
        lines.append(f"## {metric_title}")
        grouped = df[["client_count", "method", metric]].dropna().copy()
        if grouped.empty:
            lines.append("- No complete data.")
            lines.append("")
            continue
        best_rows = []
        for client_count in client_counts:
            sub = grouped[grouped["client_count"] == client_count]
            if sub.empty:
                continue
            if higher_is_better:
                row = sub.loc[sub[metric].idxmax()]
            else:
                row = sub.loc[sub[metric].idxmin()]
            best_rows.append((int(client_count), row["method"], float(row[metric])))
        for client_count, method, value in best_rows:
            lines.append(f"- clients={client_count}: best={METHOD_LABELS[method]} ({value:.4f})")
        lines.append("")

    analysis_md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    df = prepare_dataframe(args.summary)
    make_figure(df, args.figure_png, args.figure_pdf, args.title)
    make_latex_table(df, args.table, args.caption)
    make_analysis(df, args.analysis_md)
    print(f"Figure written to: {args.figure_png}")
    print(f"Figure written to: {args.figure_pdf}")
    print(f"LaTeX table written to: {args.table}")
    print(f"Analysis written to: {args.analysis_md}")


if __name__ == "__main__":
    main()
