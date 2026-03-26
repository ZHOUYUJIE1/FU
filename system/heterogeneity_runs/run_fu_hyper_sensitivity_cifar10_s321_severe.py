import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "exp_configs"
OUTPUT_ROOT = PROJECT_ROOT / "system" / "heterogeneity_runs" / "fu_hyper_sensitivity_cifar10_s321_severe"
DEFAULT_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"
DEFAULT_DATASET = "Cifar10_hetero_clean_s321_severe"
DEFAULT_SAVED_MODEL = RESULT_ROOT / "global_model_Cifar10_hetero_clean_s321_severe_FedAvg_test_1_fu_severe.pt"
DEFAULT_LAMBDAS = [0.15, 0.2, 0.3, 0.4, 0.5]
DEFAULT_MASKS = [0.05, 0.08, 0.1, 0.12, 0.15]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FU hyperparameter sensitivity experiments on CIFAR-10 severe heterogeneity "
            "for seed=321 by loading a saved checkpoint and directly unlearning."
        )
    )
    parser.add_argument("--run", action="store_true", help="Execute the experiments. Otherwise only print commands.")
    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON) else sys.executable,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--eval-gap", type=int, default=10)
    parser.add_argument("--default-lambda", type=float, default=0.3)
    parser.add_argument("--default-mask-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--mask-values", nargs="+", type=float, default=DEFAULT_MASKS)
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose output JSONs already exist.")
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--saved-model-path", type=Path, default=DEFAULT_SAVED_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=OUTPUT_ROOT / "summary.csv")
    parser.add_argument("--summary-md", type=Path, default=OUTPUT_ROOT / "summary.md")
    return parser.parse_args()


def slug_float(value):
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def result_tag(lambda_value, mask_ratio):
    return f"fu_sens_lam{slug_float(lambda_value)}_mask{slug_float(mask_ratio)}"


def artifact_path(prefix, dataset, tag, extension):
    return RESULT_ROOT / f"{prefix}_{dataset}_FedAvg_test_1_{tag}.{extension}"


def build_rows(args):
    rows = []

    for lambda_value in sorted(set(args.lambda_values)):
        rows.append(
            {
                "sweep": "lambda_reversal",
                "sweep_label": "基础反转系数",
                "sweep_value": float(lambda_value),
                "lambda_value": float(lambda_value),
                "mask_ratio": float(args.default_mask_ratio),
                "result_tag": result_tag(lambda_value, args.default_mask_ratio),
            }
        )

    for mask_ratio in sorted(set(args.mask_values)):
        rows.append(
            {
                "sweep": "initial_mask_ratio",
                "sweep_label": "初始掩码比例",
                "sweep_value": float(mask_ratio),
                "lambda_value": float(args.default_lambda),
                "mask_ratio": float(mask_ratio),
                "result_tag": result_tag(args.default_lambda, mask_ratio),
            }
        )

    return rows


def build_unique_runs(rows):
    unique = []
    seen = set()
    for row in rows:
        key = (row["result_tag"], row["lambda_value"], row["mask_ratio"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def build_command(args, row):
    return [
        args.python,
        "-u",
        "system/main.py",
        "--dataset",
        args.dataset,
        "--goal",
        "test",
        "--algorithm",
        "FedAvg",
        "--model",
        "ResNet18",
        "--device",
        args.device,
        "--device_id",
        args.device_id,
        "--global_rounds",
        str(args.global_rounds),
        "--eval_gap",
        str(args.eval_gap),
        "--random_seed",
        str(args.seed),
        "--target_client_id",
        str(args.target_client_id),
        "--forget_strategy",
        "gradient_reversal",
        "--load_saved_model",
        "true",
        "--saved_model_path",
        str(args.saved_model_path),
        "--result_tag",
        row["result_tag"],
        "--use_optimized_forgetting",
        "true",
        "--enable_backdoor_attack",
        "true",
        "--fu_lambda_reversal",
        str(row["lambda_value"]),
        "--fu_initial_mask_ratio",
        str(row["mask_ratio"]),
    ]


def run_logged_command(args, cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not args.keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_row(args, row):
    eval_post = read_json(artifact_path("eval_post_forget", args.dataset, row["result_tag"], "json"))
    forget_effect = read_json(artifact_path("forget_effect", args.dataset, row["result_tag"], "json"))
    log_path = args.output_dir / args.dataset / f"{row['result_tag']}.log"

    collected = dict(row)
    collected["log_path"] = str(log_path)
    collected["status"] = "missing"

    if not eval_post or not forget_effect:
        return collected

    attack_post = forget_effect.get("attack_post_forget", {})
    metadata = forget_effect.get("method_metadata", {})
    collected.update(
        {
            "status": "complete",
            "final_avg_acc": safe_float(eval_post.get("average_accuracy")),
            "final_target_acc": safe_float(eval_post.get("target_client_accuracy")),
            "final_retain_avg_acc": safe_float(eval_post.get("retained_average_accuracy")),
            "global_buffer_avg_acc": safe_float(eval_post.get("global_buffer_average_accuracy")),
            "global_buffer_target_acc": safe_float(eval_post.get("global_buffer_target_client_accuracy")),
            "global_buffer_retain_avg_acc": safe_float(eval_post.get("global_buffer_retained_average_accuracy")),
            "mia_post_auc": safe_float(attack_post.get("mia_auc")),
            "backdoor_post_acc": safe_float(attack_post.get("backdoor_acc")),
            "target_abs_change": safe_float(forget_effect.get("target_abs_change")),
            "target_rel_change": safe_float(forget_effect.get("target_rel_change")),
            "others_abs_change_avg": safe_float(forget_effect.get("others_abs_change_avg")),
            "others_rel_change_avg": safe_float(forget_effect.get("others_rel_change_avg")),
            "recovery_stage": str(forget_effect.get("recovery_stage", "")),
            "actual_lambda_reversal": safe_float(metadata.get("base_lambda_reversal")),
            "actual_initial_mask_ratio": safe_float(metadata.get("initial_mask_ratio")),
            "retain_target_gap": safe_float(eval_post.get("retained_average_accuracy"))
            - safe_float(eval_post.get("target_client_accuracy")),
        }
    )
    return collected


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(value, digits=4):
    if value is None:
        return "missing"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "missing"
    if np.isnan(value):
        return "missing"
    return f"{value:.{digits}f}"


def describe_trend(values, metrics, lower_is_better=False):
    if len(values) < 2:
        return "样本不足，无法判断趋势。"

    values_arr = np.array(values, dtype=float)
    metrics_arr = np.array(metrics, dtype=float)
    if np.std(values_arr) < 1e-12 or np.std(metrics_arr) < 1e-12:
        return "整体变化较平，趋势不明显。"

    corr = float(np.corrcoef(values_arr, metrics_arr)[0, 1])
    strength = "较强" if abs(corr) >= 0.6 else "中等" if abs(corr) >= 0.3 else "较弱"

    if lower_is_better:
        if corr < -0.3:
            return f"随着参数增大，目标客户端准确率整体下降，遗忘更强，趋势{strength}。"
        if corr > 0.3:
            return f"随着参数增大，目标客户端准确率整体上升，遗忘变弱，趋势{strength}。"
        return "目标客户端准确率随参数变化有波动，但单调趋势较弱。"

    if corr > 0.3:
        return f"随着参数增大，保留客户端性能整体上升，趋势{strength}。"
    if corr < -0.3:
        return f"随着参数增大，保留客户端性能整体下降，趋势{strength}。"
    return "保留客户端性能随参数变化有波动，但单调趋势较弱。"


def build_analysis_lines(rows, sweep, args):
    complete_rows = [row for row in rows if row["sweep"] == sweep and row.get("status") == "complete"]
    if not complete_rows:
        return ["- 当前 sweep 还没有完整结果。"]

    complete_rows = sorted(complete_rows, key=lambda item: item["sweep_value"])
    default_value = args.default_lambda if sweep == "lambda_reversal" else args.default_mask_ratio
    baseline = next((row for row in complete_rows if abs(row["sweep_value"] - default_value) < 1e-12), None)

    best_forgetting = min(complete_rows, key=lambda item: item["final_target_acc"])
    best_utility = max(complete_rows, key=lambda item: item["final_retain_avg_acc"])
    best_tradeoff = max(complete_rows, key=lambda item: item["retain_target_gap"])

    lines = [
        (
            f"- 最强遗忘点: `{best_forgetting['sweep_value']}`，"
            f"目标客户端准确率 `{best_forgetting['final_target_acc']:.4f}`，"
            f"保留客户端平均准确率 `{best_forgetting['final_retain_avg_acc']:.4f}`。"
        ),
        (
            f"- 最佳保留性能点: `{best_utility['sweep_value']}`，"
            f"保留客户端平均准确率 `{best_utility['final_retain_avg_acc']:.4f}`，"
            f"目标客户端准确率 `{best_utility['final_target_acc']:.4f}`。"
        ),
        (
            f"- 综合折中最优点: `{best_tradeoff['sweep_value']}`，"
            f"`retain-target gap = {best_tradeoff['retain_target_gap']:.4f}`。"
        ),
    ]

    if baseline is not None:
        lines.append(
            (
                f"- 默认值 `{default_value}` 的基线结果: "
                f"target `{baseline['final_target_acc']:.4f}`，"
                f"retain `{baseline['final_retain_avg_acc']:.4f}`，"
                f"MIA AUC `{baseline['mia_post_auc']:.4f}`。"
            )
        )

    values = [row["sweep_value"] for row in complete_rows]
    target_metrics = [row["final_target_acc"] for row in complete_rows]
    retain_metrics = [row["final_retain_avg_acc"] for row in complete_rows]
    lines.append(f"- 遗忘趋势: {describe_trend(values, target_metrics, lower_is_better=True)}")
    lines.append(f"- 保留趋势: {describe_trend(values, retain_metrics, lower_is_better=False)}")

    return lines


def write_markdown_summary(rows, args):
    lines = [
        "# FU Hyperparameter Sensitivity Summary",
        "",
        "## Setup",
        "",
        f"- dataset: `{args.dataset}`",
        f"- seed: `{args.seed}`",
        f"- target_client_id: `{args.target_client_id}`",
        f"- saved_model_path: `{args.saved_model_path}`",
        f"- default lambda_reversal: `{args.default_lambda}`",
        f"- default initial_mask_ratio: `{args.default_mask_ratio}`",
        "",
    ]

    sweep_order = [
        ("lambda_reversal", "基础反转系数"),
        ("initial_mask_ratio", "初始掩码比例"),
    ]
    for sweep, label in sweep_order:
        sweep_rows = [row for row in rows if row["sweep"] == sweep]
        sweep_rows = sorted(sweep_rows, key=lambda item: item["sweep_value"])
        lines.extend(
            [
                f"## {label}",
                "",
                "| Value | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc | Gap | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )

        for row in sweep_rows:
            lines.append(
                "| "
                f"{fmt_metric(row['sweep_value'])} | "
                f"{fmt_metric(row.get('final_avg_acc'))} | "
                f"{fmt_metric(row.get('final_target_acc'))} | "
                f"{fmt_metric(row.get('final_retain_avg_acc'))} | "
                f"{fmt_metric(row.get('global_buffer_retain_avg_acc'))} | "
                f"{fmt_metric(row.get('mia_post_auc'))} | "
                f"{fmt_metric(row.get('backdoor_post_acc'))} | "
                f"{fmt_metric(row.get('retain_target_gap'))} | "
                f"{row.get('status', 'missing')} |"
            )

        lines.extend(["", "### Analysis", ""])
        lines.extend(build_analysis_lines(rows, sweep, args))
        lines.append("")

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")


def print_plan(rows):
    print("FU sensitivity plan:")
    for row in build_unique_runs(rows):
        print(
            "  "
            f"tag={row['result_tag']} lambda={row['lambda_value']:.4f} "
            f"mask_ratio={row['mask_ratio']:.4f}"
        )


def results_exist(args, row):
    eval_path = artifact_path("eval_post_forget", args.dataset, row["result_tag"], "json")
    forget_path = artifact_path("forget_effect", args.dataset, row["result_tag"], "json")
    return eval_path.exists() and forget_path.exists()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    unique_runs = build_unique_runs(rows)
    print_plan(rows)

    if args.run:
        for row in unique_runs:
            log_path = args.output_dir / args.dataset / f"{row['result_tag']}.log"
            if args.skip_existing and results_exist(args, row):
                print(f"\n===== Skip existing: {row['result_tag']} =====")
                continue

            print(
                "\n===== Running FU sensitivity experiment: "
                f"tag={row['result_tag']} lambda={row['lambda_value']:.4f} "
                f"mask_ratio={row['mask_ratio']:.4f} ====="
            )
            cmd = build_command(args, row)
            run_logged_command(args, cmd, log_path)

    collected_rows = [collect_row(args, row) for row in rows]
    write_csv(args.summary_csv, collected_rows)
    write_markdown_summary(collected_rows, args)

    print(f"\nSummary CSV written to: {args.summary_csv}")
    print(f"Summary Markdown written to: {args.summary_md}")

    for sweep, label in [
        ("lambda_reversal", "基础反转系数"),
        ("initial_mask_ratio", "初始掩码比例"),
    ]:
        print(f"\n[{label}]")
        for line in build_analysis_lines(collected_rows, sweep, args):
            print(line)


if __name__ == "__main__":
    main()
