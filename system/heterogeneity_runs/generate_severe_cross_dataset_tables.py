import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUTS = {
    "CIFAR-10": ROOT / "Cifar10_FedAvg_heterogeneity_clean_seed42_r100_nobd_split_summary.csv",
    "GTSRB": ROOT / "GTSRB_FedAvg_heterogeneity_clean_seed42_severe_summary.csv",
    "STL10": ROOT / "STL10_FedAvg_heterogeneity_clean_seed42_severe_summary.csv",
    "Tiny-ImageNet": ROOT / "TinyImagenet_FedAvg_heterogeneity_clean_seed42_severe_summary.csv",
}

METHOD_ORDER = ["fu", "fedau", "fedcsa", "fedosd", "retrain"]
METHOD_NAMES = {
    "fu": "FU",
    "fedau": "FedAU",
    "fedcsa": "FedCSA",
    "fedosd": "FedOSD",
    "retrain": "Retrain",
}

MAIN_COLUMNS = [
    ("final_avg_acc", "Final Avg Acc", "max"),
    ("final_target_acc", "Final Target Acc", "min"),
    ("final_retain_avg_acc", "Final Retain Avg Acc", "max"),
    ("final_mia_auc", "Final MIA AUC", "min"),
]

EFFECT_COLUMNS = [
    ("target_acc_change", "Target Acc Change", "min"),
    ("others_acc_change_avg", "Others Acc Change Avg", "max"),
    ("mia_auc_drop", "MIA AUC Drop", "max"),
]


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def load_rows():
    rows = []
    for dataset, path in INPUTS.items():
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("level") != "severe":
                    continue
                if row.get("status") != "complete":
                    continue
                parsed = {
                    "dataset_display": dataset,
                    "method": row["method"],
                    "display_name": METHOD_NAMES[row["method"]],
                }
                for key, _, _ in MAIN_COLUMNS + EFFECT_COLUMNS:
                    parsed[key] = parse_float(row.get(key))
                rows.append(parsed)
    rows.sort(key=lambda row: (list(INPUTS.keys()).index(row["dataset_display"]), METHOD_ORDER.index(row["method"])))
    return rows


def winners(rows, columns, exclude_methods=None):
    exclude_methods = set(exclude_methods or [])
    result = {}
    for dataset in INPUTS.keys():
        dataset_rows = [row for row in rows if row["dataset_display"] == dataset and row["method"] not in exclude_methods]
        result[dataset] = {}
        for key, _, mode in columns:
            values = [row[key] for row in dataset_rows if row[key] is not None]
            if not values:
                result[dataset][key] = set()
                continue
            target = min(values) if mode == "min" else max(values)
            result[dataset][key] = {
                row["method"]
                for row in dataset_rows
                if row[key] is not None and abs(row[key] - target) < 1e-12
            }
    return result


def fmt(value):
    if value is None:
        return "--"
    return f"{value:.3f}"


def fmt_tex(value, highlight=False):
    text = fmt(value)
    if highlight and text != "--":
        return f"\\textbf{{{text}}}"
    return text


def write_combined_csv(rows):
    path = ROOT / "severe_cross_dataset_combined_summary.csv"
    fieldnames = [
        "dataset_display",
        "method",
        "display_name",
        *[key for key, _, _ in MAIN_COLUMNS],
        *[key for key, _, _ in EFFECT_COLUMNS],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_main_table(rows):
    best = winners(rows, MAIN_COLUMNS)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Severe Dirichlet heterogeneity ($\\alpha=0.1$) across four image datasets. Higher Final Avg Acc and Final Retain Avg Acc are better; lower Final Target Acc and Final MIA AUC are better. Best value in each dataset/metric block is boldfaced.}",
        "\\label{tab:severe_cross_dataset_main}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Method & Final Avg Acc $\\uparrow$ & Final Target Acc $\\downarrow$ & Final Retain Avg Acc $\\uparrow$ & Final MIA AUC $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for dataset in INPUTS.keys():
        dataset_rows = [row for row in rows if row["dataset_display"] == dataset]
        for idx, row in enumerate(dataset_rows):
            dataset_label = dataset if idx == 0 else ""
            cells = [dataset_label, row["display_name"]]
            for key, _, _ in MAIN_COLUMNS:
                cells.append(fmt_tex(row[key], row["method"] in best[dataset][key]))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table*}"])
    path = ROOT / "severe_cross_dataset_main_table.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_effect_table(rows):
    best = winners(rows, EFFECT_COLUMNS, exclude_methods={"retrain"})
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Forgetting-impact view under severe heterogeneity. Retrain is omitted because pre/post deltas are not directly defined in the exported summaries. More negative Target Acc Change means stronger forgetting on the target client; larger Others Acc Change Avg and MIA AUC Drop indicate better retain preservation and stronger privacy gain, respectively.}",
        "\\label{tab:severe_cross_dataset_effect}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Dataset & Method & Target Acc Change $\\downarrow$ & Others Acc Change Avg $\\uparrow$ & MIA AUC Drop $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for dataset in INPUTS.keys():
        dataset_rows = [row for row in rows if row["dataset_display"] == dataset and row["method"] != "retrain"]
        for idx, row in enumerate(dataset_rows):
            dataset_label = dataset if idx == 0 else ""
            cells = [dataset_label, row["display_name"]]
            for key, _, _ in EFFECT_COLUMNS:
                cells.append(fmt_tex(row[key], row["method"] in best[dataset][key]))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table*}"])
    path = ROOT / "severe_cross_dataset_effect_table.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dataset_commentary(rows, dataset):
    dataset_rows = [row for row in rows if row["dataset_display"] == dataset]
    by_method = {row["method"]: row for row in dataset_rows}

    utility_best = max(dataset_rows, key=lambda row: row["final_avg_acc"])
    forgetting_best_fast = min(
        [row for row in dataset_rows if row["method"] != "retrain"],
        key=lambda row: row["final_target_acc"],
    )
    privacy_best = min(dataset_rows, key=lambda row: row["final_mia_auc"])

    comments = []
    if dataset == "CIFAR-10":
        comments.append(
            f"On {dataset}, {utility_best['display_name']} achieved the best utility "
            f"({fmt(utility_best['final_avg_acc'])}) and retain accuracy ({fmt(utility_best['final_retain_avg_acc'])}), "
            f"while {forgetting_best_fast['display_name']} pushed the target client lower among fast methods "
            f"({fmt(forgetting_best_fast['final_target_acc'])})."
        )
        comments.append(
            f"Retrain remained the strongest forgetting baseline with final target accuracy {fmt(by_method['retrain']['final_target_acc'])}, "
            f"but its utility ({fmt(by_method['retrain']['final_avg_acc'])}) stayed well below FU."
        )
    elif dataset == "GTSRB":
        comments.append(
            f"On {dataset}, FedCSA almost preserved the original model ({fmt(by_method['fedcsa']['final_avg_acc'])} final average accuracy), "
            f"but forgetting was weak ({fmt(by_method['fedcsa']['final_target_acc'])} target accuracy)."
        )
        comments.append(
            f"FedAU delivered the most aggressive fast forgetting ({fmt(by_method['fedau']['final_target_acc'])}), "
            f"whereas FU provided the best trade-off between strong utility ({fmt(by_method['fu']['final_avg_acc'])}) and meaningful forgetting ({fmt(by_method['fu']['final_target_acc'])})."
        )
    elif dataset == "STL10":
        comments.append(
            f"On {dataset}, FU and FedCSA were essentially tied on utility ({fmt(by_method['fu']['final_avg_acc'])} vs. {fmt(by_method['fedcsa']['final_avg_acc'])}), "
            f"but FU forgot the target client much more strongly ({fmt(by_method['fu']['final_target_acc'])} vs. {fmt(by_method['fedcsa']['final_target_acc'])})."
        )
        comments.append(
            f"FedAU was the most aggressive fast method on both forgetting ({fmt(by_method['fedau']['final_target_acc'])}) and privacy ({fmt(by_method['fedau']['final_mia_auc'])}), "
            f"at the cost of a sharp utility drop to {fmt(by_method['fedau']['final_avg_acc'])}."
        )
    elif dataset == "Tiny-ImageNet":
        comments.append(
            f"On {dataset}, FU and FedCSA clearly dominated utility ({fmt(by_method['fu']['final_avg_acc'])} and {fmt(by_method['fedcsa']['final_avg_acc'])}), "
            f"with FU holding a slight edge on both target forgetting ({fmt(by_method['fu']['final_target_acc'])}) and privacy ({fmt(by_method['fu']['final_mia_auc'])})."
        )
        comments.append(
            f"FedAU still forgot more aggressively ({fmt(by_method['fedau']['final_target_acc'])}), but its utility dropped to {fmt(by_method['fedau']['final_avg_acc'])}; "
            f"FedOSD was dominated on both utility and privacy."
        )

    comments.append(
        f"Within {dataset}, the lowest final MIA AUC came from {privacy_best['display_name']} "
        f"({fmt(privacy_best['final_mia_auc'])})."
    )
    return comments


def write_analysis(rows):
    utility_winners = {dataset: max([row for row in rows if row["dataset_display"] == dataset], key=lambda row: row["final_avg_acc"])["method"] for dataset in INPUTS.keys()}
    fast_forgetting_winners = {
        dataset: min([row for row in rows if row["dataset_display"] == dataset and row["method"] != "retrain"], key=lambda row: row["final_target_acc"])["method"]
        for dataset in INPUTS.keys()
    }

    utility_counts = {method: sum(1 for winner in utility_winners.values() if winner == method) for method in METHOD_ORDER}
    fast_forgetting_counts = {method: sum(1 for winner in fast_forgetting_winners.values() if winner == method) for method in METHOD_ORDER}

    lines = [
        "# Severe Cross-Dataset Analysis",
        "",
        "## Overall trends",
        "",
        f"- FU won utility on {utility_counts['fu']} of the 4 datasets and was a near-tie on STL10, making it the most consistent utility-preserving unlearning method in this suite.",
        f"- FedAU was the strongest fast forgetting method on all 4 datasets when measured by final target-client accuracy, but it repeatedly paid for that strength with the largest utility drop.",
        "- FedCSA was most attractive when the goal was preserving the original model, especially on GTSRB, but its forgetting effect was often too mild.",
        "- FedOSD usually sat in the middle: it forgot more than FedCSA, but it rarely dominated on utility, retain preservation, or privacy.",
        "- Retrain remained the strongest absolute forgetting baseline, but it was not uniformly the best privacy baseline and often left noticeable utility on the table relative to FU or FedCSA.",
        "",
        "## Dataset-wise observations",
        "",
    ]

    for dataset in INPUTS.keys():
        for comment in dataset_commentary(rows, dataset):
            lines.append(f"- {comment}")
        lines.append("")

    lines.extend([
        "## Writing suggestions",
        "",
        "- If you want a single headline statement for the main paper, the cleanest one is: `FU achieves the most stable utility-forgetting trade-off across datasets, while FedAU is the most aggressive fast unlearning method and FedCSA is the most conservative.`",
        "- If you want to emphasize dataset dependence, the strongest contrast is GTSRB vs. Tiny-ImageNet: GTSRB exposes FedCSA's utility advantage and forgetting weakness most clearly, while Tiny-ImageNet highlights FU's balance under a harder 200-class regime.",
        "- STL10 is useful as the small-data case: it shows that aggressive forgetting can still be achieved, but the utility gap between FU/FedCSA and FedAU widens quickly.",
    ])

    path = ROOT / "severe_cross_dataset_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    rows = load_rows()
    csv_path = write_combined_csv(rows)
    main_table_path = write_main_table(rows)
    effect_table_path = write_effect_table(rows)
    analysis_path = write_analysis(rows)

    print(f"Wrote combined summary to: {csv_path}")
    print(f"Wrote main table to: {main_table_path}")
    print(f"Wrote effect table to: {effect_table_path}")
    print(f"Wrote analysis to: {analysis_path}")


if __name__ == "__main__":
    main()
