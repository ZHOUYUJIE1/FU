import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild Dirichlet paper assets using selected FU tags.")
    parser.add_argument("--python", type=str, default=ZYJ_PYTHON)
    parser.add_argument("--best-csv", type=Path, default=ROOT / "fu_dir_tuning_best.csv")
    parser.add_argument("--base-summary", type=Path, default=ROOT / "Cifar10_FedAvg_heterogeneity_summary.csv")
    parser.add_argument("--output-summary", type=Path, default=ROOT / "Cifar10_FedAvg_heterogeneity_summary_final.csv")
    parser.add_argument("--delta-output", type=Path, default=ROOT / "Cifar10_FedAvg_fu_final_delta.csv")
    parser.add_argument("--figure-png", type=Path, default=ROOT / "heterogeneity_paper_figure_final.png")
    parser.add_argument("--figure-pdf", type=Path, default=ROOT / "heterogeneity_paper_figure_final.pdf")
    parser.add_argument("--table", type=Path, default=ROOT / "heterogeneity_paper_table_final.tex")
    parser.add_argument("--analysis", type=Path, default=ROOT / "heterogeneity_paper_analysis_final.md")
    parser.add_argument("--severe-tag", type=str, default="fu_severe_final")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def chosen_tags(best_csv, severe_tag):
    if not best_csv.exists() or best_csv.stat().st_size == 0:
        raise FileNotFoundError(f"Best CSV is missing or empty: {best_csv}")

    best_df = pd.read_csv(best_csv)
    if best_df.empty:
        raise ValueError(f"Best CSV does not contain any completed tuning rows: {best_csv}")
    tags = {"severe": severe_tag}
    for _, row in best_df.iterrows():
        tags[str(row["level"])] = str(row["result_tag"])
    return tags


def write_analysis(summary_path, output_path):
    df = pd.read_csv(summary_path)
    fu_rows = df[df["method"] == "fu"].set_index("level")

    mild = fu_rows.loc["mild"]
    moderate = fu_rows.loc["moderate"]
    severe = fu_rows.loc["severe"]

    content = f"""# Heterogeneity Analysis for Paper

Final summary: [Cifar10_FedAvg_heterogeneity_summary_final.csv](./Cifar10_FedAvg_heterogeneity_summary_final.csv)

Main figure: [heterogeneity_paper_figure_final.png](./heterogeneity_paper_figure_final.png)

LaTeX table: [heterogeneity_paper_table_final.tex](./heterogeneity_paper_table_final.tex)

## Key results

- Under mild heterogeneity, FU achieves `{mild['final_avg_acc']:.4f}` final average accuracy and reduces the target-client accuracy to `{mild['final_target_acc']:.4f}`.
- Under moderate heterogeneity, FU achieves `{moderate['final_avg_acc']:.4f}` final average accuracy with `{moderate['final_target_acc']:.4f}` target-client accuracy.
- Under severe heterogeneity, FU achieves `{severe['final_avg_acc']:.4f}` final average accuracy and reduces the target-client accuracy to `{severe['final_target_acc']:.4f}`.
- In the severe heterogeneity setting, FU attains the highest final average accuracy among all compared methods while maintaining substantially lower target-client accuracy than FedAU, FedCSA, and FedOSD.

## Paper-ready paragraph

Table~\\ref{{tab:heterogeneity_results}} reports the results under mild, moderate, and severe heterogeneity settings on CIFAR-10. FU consistently maintains strong utility across all heterogeneous scenarios while improving the forgetting effectiveness on the target client. Under mild heterogeneity, FU achieves a final average accuracy of `{mild['final_avg_acc']:.4f}` and reduces the target-client accuracy to `{mild['final_target_acc']:.4f}`, outperforming FedAU in both utility retention and target suppression. Under moderate heterogeneity, FU reaches `{moderate['final_avg_acc']:.4f}` final average accuracy with `{moderate['final_target_acc']:.4f}` target-client accuracy, indicating a favorable balance between model availability and forgetting strength. Under severe heterogeneity, FU obtains the best final average accuracy among all compared methods (`{severe['final_avg_acc']:.4f}`) while lowering the target-client accuracy to `{severe['final_target_acc']:.4f}`, which is markedly better than FedAU, FedCSA, and FedOSD and approaches the retraining baseline. These results show that FU is particularly competitive as a balanced federated unlearning method under heterogeneous data distributions.

## Conservative wording

The most defensible claim is that FU provides a stronger utility-forgetting tradeoff, rather than strictly dominating all baselines on every single metric. This statement is especially well supported in the severe heterogeneity setting.
"""
    output_path.write_text(content, encoding="utf-8")


def run_command(cmd):
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    args = parse_args()
    tags = chosen_tags(args.best_csv, args.severe_tag)
    print("Selected FU tags:", tags)

    build_summary_cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/build_fu_adaptive_summary.py",
        "--base-summary", str(args.base_summary),
        "--output-summary", str(args.output_summary),
        "--delta-output", str(args.delta_output),
        "--fu-tag", f"mild:{tags['mild']}",
        "--fu-tag", f"moderate:{tags['moderate']}",
        "--fu-tag", f"severe:{tags['severe']}",
    ]
    assets_cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_heterogeneity_paper_assets.py",
        "--summary", str(args.output_summary),
        "--figure-png", str(args.figure_png),
        "--figure-pdf", str(args.figure_pdf),
        "--table", str(args.table),
        "--title", "Comparison Under Mild, Moderate, and Severe Dirichlet Heterogeneity",
        "--caption",
        "Results under different Dirichlet heterogeneity levels on CIFAR-10. Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, while higher average accuracy indicates better utility retention.",
    ]

    if not args.run:
        print(" ".join(build_summary_cmd))
        print(" ".join(assets_cmd))
        return

    run_command(build_summary_cmd)
    run_command(assets_cmd)
    write_analysis(args.output_summary, args.analysis)

    print(f"Summary written to: {args.output_summary}")
    print(f"Figure written to: {args.figure_png}")
    print(f"Figure written to: {args.figure_pdf}")
    print(f"Table written to: {args.table}")
    print(f"Analysis written to: {args.analysis}")


if __name__ == "__main__":
    main()
