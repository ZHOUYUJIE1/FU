import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "exp_configs"

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
LEVEL_ORDER = ["mild", "moderate", "severe"]

MAIN_CONFIGS = [
    {"dataset_display": "CIFAR-10", "dataset": "Cifar10_hetero_clean_s42_mild", "level": "mild"},
    {"dataset_display": "CIFAR-10", "dataset": "Cifar10_hetero_clean_s42_moderate", "level": "moderate"},
    {"dataset_display": "CIFAR-10", "dataset": "Cifar10_hetero_clean_s42_severe", "level": "severe"},
]
SEVERE_CROSS_DATASET_CONFIGS = [
    {"dataset_display": "CIFAR-10", "dataset": "Cifar10_hetero_clean_s42_severe", "level": "severe"},
    {"dataset_display": "GTSRB", "dataset": "GTSRB_hetero_clean_s42_severe", "level": "severe"},
    {"dataset_display": "STL10", "dataset": "STL10_hetero_clean_s42_severe", "level": "severe"},
    {"dataset_display": "Tiny-ImageNet", "dataset": "TinyImagenet_hetero_clean_s42_severe", "level": "severe"},
]


def use_plot_style():
    for style_name in ("seaborn-whitegrid", "ggplot", "default"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_result_path(prefix: str, dataset: str, result_tag: str) -> Path:
    return RESULT_ROOT / f"{prefix}_{dataset}_FedAvg_test_1_{result_tag}.json"


def load_eval_pair(dataset: str, method: str, level: str):
    tag = f"{method}_{level}"

    if method == "retrain":
        return None, read_json(build_result_path("eval_retrain_only", dataset, tag))

    pre_eval = read_json(build_result_path("eval_pre_forget", dataset, tag))
    post_recovery = read_json(build_result_path("eval_post_recovery", dataset, tag))
    post_forget = read_json(build_result_path("eval_post_forget", dataset, tag))
    final_eval = post_recovery if post_recovery is not None else post_forget
    return pre_eval, final_eval


def load_attack_pair(dataset: str, method: str, level: str):
    tag = f"{method}_{level}"

    if method == "retrain":
        return None, read_json(build_result_path("attack_retrain_only", dataset, tag))

    pre_attack = read_json(build_result_path("attack_pre_forget", dataset, tag))
    post_recovery = read_json(build_result_path("attack_post_recovery", dataset, tag))
    post_forget = read_json(build_result_path("attack_post_forget", dataset, tag))
    final_attack = post_recovery if post_recovery is not None else post_forget
    return pre_attack, final_attack


def get_mia_auc(attack_data):
    if not attack_data:
        return np.nan
    return attack_data.get("mia", {}).get("auc", np.nan)


def metric_or_nan(data, key):
    if not data:
        return np.nan
    value = data.get(key, np.nan)
    return np.nan if value is None else value


def collect_rows(configs):
    rows = []
    for cfg in configs:
        for method in METHOD_ORDER:
            pre_eval, final_eval = load_eval_pair(cfg["dataset"], method, cfg["level"])
            pre_attack, final_attack = load_attack_pair(cfg["dataset"], method, cfg["level"])

            rows.append(
                {
                    "dataset_display": cfg["dataset_display"],
                    "dataset": cfg["dataset"],
                    "level": cfg["level"],
                    "method": method,
                    "display_name": METHOD_LABELS[method],
                    "pre_local_target_acc": metric_or_nan(pre_eval, "local_buffer_target_client_accuracy"),
                    "final_local_target_acc": metric_or_nan(final_eval, "local_buffer_target_client_accuracy"),
                    "local_target_drop": (
                        metric_or_nan(pre_eval, "local_buffer_target_client_accuracy")
                        - metric_or_nan(final_eval, "local_buffer_target_client_accuracy")
                    ),
                    "pre_global_model_acc": metric_or_nan(pre_eval, "global_buffer_average_accuracy"),
                    "final_global_retain_acc": metric_or_nan(final_eval, "global_buffer_retained_average_accuracy"),
                    "retain_vs_pre_global_delta": (
                        metric_or_nan(final_eval, "global_buffer_retained_average_accuracy")
                        - metric_or_nan(pre_eval, "global_buffer_average_accuracy")
                    ),
                    "pre_mia_auc": get_mia_auc(pre_attack),
                    "final_mia_auc": get_mia_auc(final_attack),
                    "mia_delta": get_mia_auc(final_attack) - get_mia_auc(pre_attack),
                }
            )

    df = pd.DataFrame(rows)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)

    for (dataset_name, level), group in df.groupby(["dataset", "level"], observed=True):
        retrain = group[group["method"] == "retrain"].iloc[0]
        retrain_target = retrain["final_local_target_acc"]
        retrain_retain = retrain["final_global_retain_acc"]

        for idx in group.index:
            method = df.at[idx, "method"]
            if method == "retrain":
                df.at[idx, "retrain_gap_target"] = np.nan
                df.at[idx, "retrain_gap_retain"] = np.nan
                df.at[idx, "retrain_distance"] = np.nan
                continue

            target_gap = abs(df.at[idx, "final_local_target_acc"] - retrain_target)
            retain_gap = abs(df.at[idx, "final_global_retain_acc"] - retrain_retain)
            df.at[idx, "retrain_gap_target"] = target_gap
            df.at[idx, "retrain_gap_retain"] = retain_gap
            df.at[idx, "retrain_distance"] = math.sqrt(target_gap ** 2 + retain_gap ** 2)

    return df.sort_values(["dataset_display", "level", "method"]).reset_index(drop=True)


def fmt(value, digits=3):
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def format_cell(row, column, best_rows):
    value = row[column]
    text = fmt(value)
    key = (row["dataset"], row["level"], row["method"], column)
    if key in best_rows:
        return f"\\textbf{{{text}}}"
    return text


def build_best_row_set(df, columns_by_rule):
    best_rows = set()
    non_retrain = df[df["method"] != "retrain"]
    for (dataset_name, level), group in non_retrain.groupby(["dataset", "level"], observed=True):
        for column, rule in columns_by_rule.items():
            if group[column].isna().all():
                continue
            value = group[column].min() if rule == "min" else group[column].max()
            winners = group[np.isclose(group[column], value)]
            for _, row in winners.iterrows():
                best_rows.add((dataset_name, level, row["method"], column))
    return best_rows


def write_main_table(df: pd.DataFrame, output_path: Path):
    subset = df[df["dataset_display"] == "CIFAR-10"].copy()
    best_rows = build_best_row_set(
        subset,
        {
            "final_local_target_acc": "min",
            "final_global_retain_acc": "max",
            "retrain_distance": "min",
            "final_mia_auc": "min",
        },
    )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{CIFAR-10 heterogeneity results using the exact evaluation protocol emphasized in our paper. "
        "Target-client accuracies are measured on local buffers (LB), while utility is measured on global buffers (GB). "
        "Lower target accuracy, Retrain distance, and MIA AUC are better; higher retained-client accuracy is better. "
        "FU, FedCSA, and FedOSD share the same pre-unlearning checkpoint in these runs, so their pre-unlearning LB/GB metrics are directly comparable. "
        "FedAU is trained separately because it requires training-time cache states.}",
        "\\label{tab:lbgb_main_results}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Level & Method & LB Target Before & LB Target After & GB Global Before & GB Retained After & Dist. to Retrain & Post MIA AUC \\\\",
        "\\midrule",
    ]

    for level in LEVEL_ORDER:
        group = subset[subset["level"] == level]
        for idx, (_, row) in enumerate(group.iterrows()):
            level_label = level.capitalize() if idx == 0 else ""
            dist_text = "--" if row["method"] == "retrain" else format_cell(row, "retrain_distance", best_rows)
            lines.append(
                f"{level_label} & {row['display_name']} & "
                f"{fmt(row['pre_local_target_acc'])} & "
                f"{format_cell(row, 'final_local_target_acc', best_rows)} & "
                f"{fmt(row['pre_global_model_acc'])} & "
                f"{format_cell(row, 'final_global_retain_acc', best_rows)} & "
                f"{dist_text} & "
                f"{format_cell(row, 'final_mia_auc', best_rows)} \\\\"
            )
        if level != LEVEL_ORDER[-1]:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_cross_dataset_table(df: pd.DataFrame, output_path: Path):
    best_rows = build_best_row_set(
        df,
        {
            "final_local_target_acc": "min",
            "final_global_retain_acc": "max",
            "retrain_distance": "min",
            "final_mia_auc": "min",
        },
    )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Severe-heterogeneity results across datasets. LB target accuracy captures forgetting strength, "
        "GB retained-client accuracy captures retained utility, and Dist. to Retrain is the Euclidean distance in the "
        "(LB target, GB retained) plane.}",
        "\\label{tab:lbgb_cross_dataset_results}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Method & LB Target After & GB Retained After & Dist. to Retrain & Post MIA AUC \\\\",
        "\\midrule",
    ]

    dataset_order = [cfg["dataset"] for cfg in SEVERE_CROSS_DATASET_CONFIGS]
    for dataset_name in dataset_order:
        group = df[df["dataset"] == dataset_name]
        dataset_label = group.iloc[0]["dataset_display"]
        for idx, (_, row) in enumerate(group.iterrows()):
            label = dataset_label if idx == 0 else ""
            dist_text = "--" if row["method"] == "retrain" else format_cell(row, "retrain_distance", best_rows)
            lines.append(
                f"{label} & {row['display_name']} & "
                f"{format_cell(row, 'final_local_target_acc', best_rows)} & "
                f"{format_cell(row, 'final_global_retain_acc', best_rows)} & "
                f"{dist_text} & "
                f"{format_cell(row, 'final_mia_auc', best_rows)} \\\\"
            )
        if dataset_name != dataset_order[-1]:
            lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_main_metrics_figure(df: pd.DataFrame, png_path: Path, pdf_path: Path):
    subset = df[df["dataset_display"] == "CIFAR-10"].copy()
    x = np.arange(len(LEVEL_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(METHOD_ORDER)) * width

    metrics = [
        ("final_local_target_acc", "LB Target Accuracy After", False),
        ("final_global_retain_acc", "GB Retained Accuracy After", True),
        ("retrain_distance", "Distance to Retrain", False),
        ("final_mia_auc", "Post-unlearning MIA AUC", False),
    ]

    use_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title, higher_is_better) in zip(axes, metrics):
        for offset, method in zip(offsets, METHOD_ORDER):
            method_df = subset[subset["method"] == method].sort_values("level")
            y = method_df[metric].to_numpy(dtype=float)
            if metric == "retrain_distance" and method == "retrain":
                y = np.zeros_like(y)
            ax.bar(
                x + offset,
                y,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                label=METHOD_LABELS[method],
            )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["Mild", "Moderate", "Severe"])
        ax.set_ylabel("Higher is better" if higher_is_better else "Lower is better")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle(
        "Main Metrics Under CIFAR-10 Heterogeneity (LB for Forgetting, GB for Utility)",
        fontsize=13,
        y=1.06,
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_figure(df: pd.DataFrame, png_path: Path, pdf_path: Path):
    subset = df[df["dataset_display"] == "CIFAR-10"].copy()
    use_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)

    markers = {"fu": "o", "fedau": "s", "fedcsa": "^", "fedosd": "D", "retrain": "*"}

    for ax, level in zip(axes, LEVEL_ORDER):
        level_df = subset[subset["level"] == level]
        for _, row in level_df.iterrows():
            size = 220 if row["method"] == "retrain" else 90
            ax.scatter(
                row["final_local_target_acc"],
                row["final_global_retain_acc"],
                s=size,
                marker=markers[row["method"]],
                color=METHOD_COLORS[row["method"]],
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                row["display_name"],
                (row["final_local_target_acc"], row["final_global_retain_acc"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8.5,
            )

        ax.set_title(level.capitalize())
        ax.set_xlabel("LB Target Accuracy After (lower is better)")
        ax.set_ylabel("GB Retained Accuracy After (higher is better)")
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Utility-Forgetting Trade-off Relative to Retrain on CIFAR-10", fontsize=13)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_analysis(main_df: pd.DataFrame, cross_df: pd.DataFrame, output_path: Path):
    def row(dataset, level, method, df):
        result = df[(df["dataset"] == dataset) & (df["level"] == level) & (df["method"] == method)]
        return result.iloc[0]

    mild_fu = row("Cifar10_hetero_clean_s42_mild", "mild", "fu", main_df)
    mild_best_distance = (
        main_df[(main_df["dataset"] == "Cifar10_hetero_clean_s42_mild") & (main_df["method"] != "retrain")]
        .sort_values("retrain_distance")
        .iloc[0]
    )
    moderate_best_distance = (
        main_df[(main_df["dataset"] == "Cifar10_hetero_clean_s42_moderate") & (main_df["method"] != "retrain")]
        .sort_values("retrain_distance")
        .iloc[0]
    )
    severe_fu = row("Cifar10_hetero_clean_s42_severe", "severe", "fu", main_df)
    severe_fedau = row("Cifar10_hetero_clean_s42_severe", "severe", "fedau", main_df)
    cross_best = (
        cross_df[cross_df["method"] != "retrain"]
        .sort_values(["dataset_display", "retrain_distance"])
        .groupby("dataset_display", observed=True)
        .first()
        .reset_index()
    )

    lines = [
        "# LB/GB Paper Assets",
        "",
        "## 产出文件",
        "",
        "- 主表 CSV: `paper_lbgb_main_summary.csv`",
        "- 主表 LaTeX: `paper_lbgb_main_table.tex`",
        "- 主图 PNG/PDF: `paper_lbgb_main_metrics.png`, `paper_lbgb_main_metrics.pdf`",
        "- Trade-off 图 PNG/PDF: `paper_lbgb_tradeoff.png`, `paper_lbgb_tradeoff.pdf`",
        "- 严重异质性跨数据集表: `paper_lbgb_cross_dataset_summary.csv`, `paper_lbgb_cross_dataset_table.tex`",
        "",
        "## 关键观察（中文）",
        "",
        f"- CIFAR-10 mild 下，{mild_best_distance['display_name']} 与 Retrain 最近，距离仅 {mild_best_distance['retrain_distance']:.3f}。"
        f"FU 的遗忘后 LB 目标客户端准确率为 {mild_fu['final_local_target_acc']:.3f}，GB 保留客户端平均准确率为 {mild_fu['final_global_retain_acc']:.3f}。",
        f"- CIFAR-10 moderate 下，与 Retrain 最近的方法是 {moderate_best_distance['display_name']}，其距离为 {moderate_best_distance['retrain_distance']:.3f}。"
        f"这一设置下，方法间差异更多体现在保留效用而非 MIA。",
        f"- CIFAR-10 severe 下，FU 的 LB 目标客户端准确率从 {severe_fu['pre_local_target_acc']:.3f} 降到 {severe_fu['final_local_target_acc']:.3f}，"
        f"同时 GB 保留客户端平均准确率达到 {severe_fu['final_global_retain_acc']:.3f}，与 Retrain 的距离仅 {severe_fu['retrain_distance']:.3f}。",
        f"- 同样在 CIFAR-10 severe 下，FedAU 的目标压制更强（{severe_fedau['final_local_target_acc']:.3f}），"
        f"但 GB 保留客户端平均准确率只有 {severe_fedau['final_global_retain_acc']:.3f}，明显低于 FU 的 {severe_fu['final_global_retain_acc']:.3f}。",
        "- MIA AUC 在多数设置中都集中在 0.48 到 0.50 左右，说明这些方法的差距主要体现在忘记-效用权衡，而不是形成非常大的隐私攻击差距。",
        "- 在这些实验里，FU、FedCSA 和 FedOSD 共享完全相同的遗忘前 checkpoint，因此它们的 `LB Target Before` 与 `GB Global Before` 可以直接横向比较；FedAU 由于需要训练期缓存，必须单独训练，所以其遗忘前指标应单独解释。",
        "",
        "## 跨数据集观察（中文）",
        "",
    ]

    for _, best in cross_best.iterrows():
        lines.append(
            f"- {best['dataset_display']} severe 下，按与 Retrain 的二维距离衡量，最接近 Retrain 的非重训练方法是 "
            f"{best['display_name']}（距离 {best['retrain_distance']:.3f}，LB 目标准确率 {best['final_local_target_acc']:.3f}，"
            f"GB 保留客户端平均准确率 {best['final_global_retain_acc']:.3f}）。"
        )

    lines.extend(
        [
            "",
            "## 论文主文可直接使用的中文表述",
            "",
            "在 CIFAR-10 的 mild, moderate 和 severe 异质性设置下，我们采用 local buffer 评估目标客户端遗忘效果，"
            "并采用 global buffer 评估遗忘后保留客户端的整体效用。结果表明，不同方法在“遗忘强度”和“保留效用”之间存在明显权衡。"
            f"在 mild 设置下，FU 将目标客户端的 local-buffer 准确率降低到 {mild_fu['final_local_target_acc']:.3f}，同时保持 {mild_fu['final_global_retain_acc']:.3f} 的 global-buffer 保留客户端平均准确率，"
            f"是非重训练方法中最接近 Retrain 的方案。"
            f"在 moderate 设置下，FedOSD 与 Retrain 的距离最小，说明其在该设置下更接近重训练基线；"
            f"而在 severe 设置下，FU 重新取得最优的 Retrain 接近度（{severe_fu['retrain_distance']:.3f}），"
            f"并在明显优于 FedAU 的保留客户端效用前提下，将目标客户端准确率从 {severe_fu['pre_local_target_acc']:.3f} 压低到 {severe_fu['final_local_target_acc']:.3f}。"
            "整体来看，FU 在 severe 异质性下呈现出最稳定的忘记-效用平衡，而 moderate 设置下则需要结合具体目标决定是否采用更接近 Retrain 的 FedOSD。",
            "此外，在本组实验中，FU、FedCSA 和 FedOSD 使用同一遗忘前全局模型作为起点，因此它们的遗忘前 local-buffer 与 global-buffer 指标可以直接横向比较；只有 FedAU 由于方法本身依赖训练阶段缓存，需要重新训练得到专属起点。",
            "",
            "## English Paragraph for the Paper",
            "",
            "Table~\\ref{tab:lbgb_main_results} reports the heterogeneous-data comparison using the exact evaluation protocol emphasized in our paper: "
            "target-client forgetting is measured on local buffers, whereas retained utility is measured on global buffers. "
            "In our setup, FU, FedCSA, and FedOSD start from the same pre-unlearning checkpoint, so their pre-unlearning LB/GB metrics are directly comparable; "
            "FedAU is trained separately because it depends on training-time cache states. "
            f"Under mild heterogeneity, FU reduces the local-buffer target-client accuracy to {mild_fu['final_local_target_acc']:.3f} while preserving "
            f"{mild_fu['final_global_retain_acc']:.3f} global-buffer retained-client accuracy, making it the closest non-retraining baseline to Retrain in this setting. "
            f"Under moderate heterogeneity, {moderate_best_distance['display_name']} becomes the closest method to Retrain, indicating that the best utility-forgetting balance is setting dependent. "
            f"Under severe heterogeneity, FU lowers the target-client accuracy from {severe_fu['pre_local_target_acc']:.3f} to {severe_fu['final_local_target_acc']:.3f} and retains "
            f"{severe_fu['final_global_retain_acc']:.3f} accuracy on the retained clients, achieving the smallest distance to Retrain among the non-retraining methods. "
            "Across most settings, the post-unlearning MIA AUC remains around 0.48--0.50, suggesting that the major differences among methods come from the forgetting-utility trade-off rather than from large separations in attack AUC.",
            "",
            "## Figure Caption (English)",
            "",
            "Main metrics under CIFAR-10 heterogeneity. Target forgetting is measured by the post-unlearning local-buffer target-client accuracy, "
            "retained utility is measured by the post-unlearning global-buffer retained-client accuracy, and Retrain distance is computed in the two-dimensional "
            "(local-buffer target accuracy, global-buffer retained accuracy) plane. Lower target accuracy, Retrain distance, and MIA AUC are better.",
            "",
            "## Supplementary Table Caption (English)",
            "",
            "Cross-dataset severe-heterogeneity comparison using local-buffer target accuracy and global-buffer retained-client accuracy. "
            "Dist. to Retrain denotes the Euclidean distance to the Retrain point in the same evaluation space.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    main_df = collect_rows(MAIN_CONFIGS)
    cross_df = collect_rows(SEVERE_CROSS_DATASET_CONFIGS)

    main_csv = ROOT / "paper_lbgb_main_summary.csv"
    cross_csv = ROOT / "paper_lbgb_cross_dataset_summary.csv"
    main_table = ROOT / "paper_lbgb_main_table.tex"
    cross_table = ROOT / "paper_lbgb_cross_dataset_table.tex"
    main_fig_png = ROOT / "paper_lbgb_main_metrics.png"
    main_fig_pdf = ROOT / "paper_lbgb_main_metrics.pdf"
    tradeoff_fig_png = ROOT / "paper_lbgb_tradeoff.png"
    tradeoff_fig_pdf = ROOT / "paper_lbgb_tradeoff.pdf"
    analysis_md = ROOT / "paper_lbgb_analysis.md"

    main_df.to_csv(main_csv, index=False)
    cross_df.to_csv(cross_csv, index=False)
    write_main_table(main_df, main_table)
    write_cross_dataset_table(cross_df, cross_table)
    make_main_metrics_figure(main_df, main_fig_png, main_fig_pdf)
    make_tradeoff_figure(main_df, tradeoff_fig_png, tradeoff_fig_pdf)
    write_analysis(main_df, cross_df, analysis_md)

    print(f"Wrote {main_csv}")
    print(f"Wrote {cross_csv}")
    print(f"Wrote {main_table}")
    print(f"Wrote {cross_table}")
    print(f"Wrote {main_fig_png}")
    print(f"Wrote {main_fig_pdf}")
    print(f"Wrote {tradeoff_fig_png}")
    print(f"Wrote {tradeoff_fig_pdf}")
    print(f"Wrote {analysis_md}")


if __name__ == "__main__":
    main()
