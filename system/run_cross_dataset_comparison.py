import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = PROJECT_ROOT / "system"
HETERO_ROOT = SYSTEM_ROOT / "heterogeneity_runs"
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

import run_heterogeneity_comparison as hetero


DATASET_ALIASES = {
    "cinic-10": "CINIC10",
    "cinic10": "CINIC10",
    "tiny-imagenet": "TinyImagenet",
    "tinyimagenet": "TinyImagenet",
    "stl-10": "STL10",
    "stl10": "STL10",
    "gtsrb": "GTSRB",
}

DATASET_DISPLAY_NAMES = {
    "CINIC10": "CINIC-10",
    "TinyImagenet": "Tiny-ImageNet",
    "STL10": "STL-10",
    "GTSRB": "GTSRB",
}

DATASET_NUM_CLASSES = {
    "CINIC10": 10,
    "TinyImagenet": 200,
    "STL10": 10,
    "GTSRB": 43,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FU/FedAU/FedOSD/FedCSA/Retrain comparisons across multiple datasets "
            "(CINIC-10, Tiny-ImageNet, STL-10, GTSRB) while preserving the existing "
            "heterogeneity workflow."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GTSRB"],
        help="Dataset names. Supports aliases like CINIC-10/Tiny-ImageNet/STL-10/GTSRB.",
    )
    parser.add_argument("--experiment-name", type=str, default="cross_dataset_comparison")
    parser.add_argument("--dataset-prefix", type=str, default="crossds")
    parser.add_argument("--goal", type=str, default="test")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument(
        "--python",
        type=str,
        default=ZYJ_PYTHON if os.path.exists(ZYJ_PYTHON) else sys.executable,
    )
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--paper-assets", action="store_true",
                        help="Generate figure/table/analysis assets for the combined summary.")

    parser.add_argument("--levels", nargs="+", choices=hetero.LEVEL_ORDER, default=["severe"])
    parser.add_argument("--methods", nargs="+", choices=hetero.METHOD_ORDER, default=list(hetero.METHOD_ORDER))
    parser.add_argument("--partition", choices=["dir", "pat", "exdir"], default="dir")
    parser.add_argument("--alpha-mild", type=float, default=1.0)
    parser.add_argument("--alpha-moderate", type=float, default=0.3)
    parser.add_argument("--alpha-severe", type=float, default=0.1)
    parser.add_argument("--class-per-client-mild", type=int, default=5)
    parser.add_argument("--class-per-client-moderate", type=int, default=3)
    parser.add_argument("--class-per-client-severe", type=int, default=2)

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Optional override for all datasets. By default use dataset-specific class counts.")
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--top-cnt", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-learning-rate", type=float, default=0.005)
    parser.add_argument("--learning-rate-decay", type=str, default="false")
    parser.add_argument("--learning-rate-decay-gamma", type=float, default=0.99)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--eval-gap", type=int, default=1)
    parser.add_argument("--auto-break", type=str, default="false")
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-rounds", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=4)

    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--balance", dest="balance", action="store_true")
    balance_group.add_argument("--unbalance", dest="balance", action="store_false")
    parser.set_defaults(balance=True)

    return parser.parse_args()


def normalize_dataset_names(dataset_tokens):
    normalized = []
    seen = set()
    for raw in dataset_tokens:
        key = raw.strip()
        if not key:
            continue
        canonical = DATASET_ALIASES.get(key.lower(), key)
        if canonical not in DATASET_NUM_CLASSES:
            supported = ", ".join(sorted(DATASET_NUM_CLASSES.keys()))
            raise ValueError(f"Unsupported dataset '{raw}'. Supported canonical names: {supported}.")
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    if not normalized:
        raise ValueError("No valid datasets specified.")
    return normalized


def num_classes_for(args, dataset_base):
    if args.num_classes is not None:
        return int(args.num_classes)
    return int(DATASET_NUM_CLASSES[dataset_base])


def summary_path_for(args, dataset_base):
    return HETERO_ROOT / f"{dataset_base}_{args.algorithm}_{args.experiment_name}_summary.csv"


def command_script_path(args):
    return HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_multidataset_commands.sh"


def combined_summary_path(args):
    return HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_multidataset_summary.csv"


def analysis_path(args):
    return HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_analysis.md"


def wrap_command_for_shell(args, cmd):
    if args.keep_ld_library_path:
        return list(cmd)
    return ["env", "-u", "LD_LIBRARY_PATH", *cmd]


def build_child_command(args, dataset_base):
    cmd = [
        args.python,
        "-u",
        "system/run_heterogeneity_comparison.py",
        "--dataset-base", dataset_base,
        "--dataset-prefix", args.dataset_prefix,
        "--experiment-name", args.experiment_name,
        "--goal", args.goal,
        "--algorithm", args.algorithm,
        "--model", args.model,
        "--device", args.device,
        "--device-id", args.device_id,
        "--python", args.python,
        "--levels", *args.levels,
        "--methods", *args.methods,
        "--partition", args.partition,
        "--num-clients", str(args.num_clients),
        "--num-classes", str(num_classes_for(args, dataset_base)),
        "--global-rounds", str(args.global_rounds),
        "--top-cnt", str(args.top_cnt),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--local-learning-rate", str(args.local_learning_rate),
        "--learning-rate-decay", str(args.learning_rate_decay),
        "--learning-rate-decay-gamma", str(args.learning_rate_decay_gamma),
        "--join-ratio", str(args.join_ratio),
        "--times", str(args.times),
        "--eval-gap", str(args.eval_gap),
        "--auto-break", str(args.auto_break),
        "--target-client-id", str(args.target_client_id),
        "--seed", str(args.seed),
        "--recovery-rounds", str(args.recovery_rounds),
        "--max-batches", str(args.max_batches),
        "--num-processes", str(args.num_processes),
        "--alpha-mild", str(args.alpha_mild),
        "--alpha-moderate", str(args.alpha_moderate),
        "--alpha-severe", str(args.alpha_severe),
        "--class-per-client-mild", str(args.class_per_client_mild),
        "--class-per-client-moderate", str(args.class_per_client_moderate),
        "--class-per-client-severe", str(args.class_per_client_severe),
        "--balance" if args.balance else "--unbalance",
    ]

    if args.keep_ld_library_path:
        cmd.append("--keep-ld-library-path")
    if args.skip_prepare:
        cmd.append("--skip-prepare")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.resume_incomplete:
        cmd.append("--resume-incomplete")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.force_regenerate:
        cmd.append("--force-regenerate")

    if args.summary_only:
        cmd.append("--summary-only")
    elif args.run:
        cmd.append("--run")

    return cmd


def write_command_script(args, commands):
    HETERO_ROOT.mkdir(parents=True, exist_ok=True)
    script_path = command_script_path(args)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"cd {shlex.quote(str(PROJECT_ROOT))}\n\n")
        for _, cmd in commands:
            f.write(shlex.join(wrap_command_for_shell(args, cmd)) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def run_logged_command(args, cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not args.keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)

    with open(log_path, "w", encoding="utf-8") as log_file:
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


def read_summary_rows(summary_path):
    if not summary_path.exists():
        return []
    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_combined_summary(args, datasets):
    rows = []
    for dataset_base in datasets:
        summary_path = summary_path_for(args, dataset_base)
        display_name = DATASET_DISPLAY_NAMES[dataset_base]
        for row in read_summary_rows(summary_path):
            merged = dict(row)
            merged["dataset_base"] = dataset_base
            merged["dataset_display"] = display_name
            rows.append(merged)

    output_path = combined_summary_path(args)
    if not rows:
        if output_path.exists():
            output_path.unlink()
        return output_path, rows

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    HETERO_ROOT.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path, rows


def generate_analysis(args, rows):
    output_path = analysis_path(args)
    lines = [
        f"# 跨数据集对比分析（{args.algorithm}）",
        "",
        f"- 实验名称: `{args.experiment_name}`",
        f"- 数据集: {', '.join(DATASET_DISPLAY_NAMES[d] for d in sorted(set(row.get('dataset_base', '') for row in rows if row.get('dataset_base')))) if rows else '无'}",
        f"- 方法: {', '.join(args.methods)}",
        f"- 层级: {', '.join(args.levels)}",
        "",
    ]

    complete_rows = []
    for row in rows:
        if str(row.get("status", "")).strip().lower() != "complete":
            continue
        row_copy = dict(row)
        row_copy["_final_avg_acc"] = to_float(row.get("final_avg_acc"))
        row_copy["_final_target_acc"] = to_float(row.get("final_target_acc"))
        if row_copy["_final_avg_acc"] is None or row_copy["_final_target_acc"] is None:
            continue
        complete_rows.append(row_copy)

    if not complete_rows:
        lines.append("当前没有可分析的完整结果（`status=complete` 且关键指标可解析）。")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    lines.extend(["## 按数据集结论", ""])

    by_dataset_level = {}
    for row in complete_rows:
        key = (row["dataset_display"], row.get("level", "unknown"))
        by_dataset_level.setdefault(key, []).append(row)

    for (dataset_display, level) in sorted(by_dataset_level.keys()):
        group = by_dataset_level[(dataset_display, level)]
        best_utility = max(group, key=lambda r: r["_final_avg_acc"])
        best_forgetting = min(group, key=lambda r: r["_final_target_acc"])
        fu_row = next((r for r in group if r.get("method") == "fu"), None)

        lines.append(f"### {dataset_display} / {level}")
        lines.append(
            f"- 最佳可用性方法: `{best_utility.get('method')}`，`final_avg_acc={best_utility['_final_avg_acc']:.4f}`。"
        )
        lines.append(
            f"- 最强遗忘方法: `{best_forgetting.get('method')}`，`final_target_acc={best_forgetting['_final_target_acc']:.4f}`。"
        )
        if fu_row is not None:
            fu_avg = fu_row["_final_avg_acc"]
            fu_target = fu_row["_final_target_acc"]
            lines.append(
                f"- FU 指标: `final_avg_acc={fu_avg:.4f}`，`final_target_acc={fu_target:.4f}`。"
            )
        else:
            lines.append("- FU 结果缺失，无法在该组做 FU 对比。")
        lines.append("")

    lines.extend(["## 方法总体趋势", ""])

    by_method = {}
    for row in complete_rows:
        method = row.get("method")
        if method not in args.methods:
            continue
        by_method.setdefault(method, {"avg": [], "target": []})
        by_method[method]["avg"].append(row["_final_avg_acc"])
        by_method[method]["target"].append(row["_final_target_acc"])

    method_rows = []
    for method, metrics in by_method.items():
        if not metrics["avg"] or not metrics["target"]:
            continue
        method_rows.append(
            (
                method,
                sum(metrics["avg"]) / len(metrics["avg"]),
                sum(metrics["target"]) / len(metrics["target"]),
                len(metrics["avg"]),
            )
        )
    method_rows.sort(key=lambda x: (-x[1], x[2], x[0]))

    for method, avg_acc, target_acc, count in method_rows:
        lines.append(
            f"- `{method}`: 平均 `final_avg_acc={avg_acc:.4f}`，平均 `final_target_acc={target_acc:.4f}`（样本数={count}）。"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def warn_if_smoke_like_config(args, datasets):
    vision_datasets = {"CINIC10", "TinyImagenet", "STL10", "GTSRB"}
    if not any(dataset in vision_datasets for dataset in datasets):
        return

    warnings = []
    if str(args.model).upper() == "MLR":
        warnings.append("model=MLR")
    if int(args.global_rounds) <= 5:
        warnings.append(f"global_rounds={args.global_rounds}")
    if int(args.num_clients) <= 2:
        warnings.append(f"num_clients={args.num_clients}")
    if str(args.device).lower() == "cpu":
        warnings.append("device=cpu")
    if int(args.max_batches) <= 2:
        warnings.append(f"max_batches={args.max_batches}")

    if warnings:
        print(
            "[WARNING] Current cross-dataset configuration looks like a smoke test for vision benchmarks: "
            + ", ".join(warnings)
            + "."
        )
        print(
            "[WARNING] For publication-grade runs, prefer ResNet18/CUDA with >=10 clients and substantially more than 1-5 rounds."
        )


def generate_paper_assets(args, summary_path):
    asset_script = HETERO_ROOT / "generate_cross_dataset_paper_assets.py"
    if not asset_script.exists() or not summary_path.exists():
        return

    cmd = [
        args.python,
        "-u",
        str(asset_script),
        "--summary",
        str(summary_path),
        "--figure-png",
        str(HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_figure.png"),
        "--figure-pdf",
        str(HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_figure.pdf"),
        "--table",
        str(HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_table.tex"),
        "--analysis-md",
        str(HETERO_ROOT / f"{args.algorithm}_{args.experiment_name}_paper_analysis.md"),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    args = parse_args()
    datasets = normalize_dataset_names(args.datasets)
    warn_if_smoke_like_config(args, datasets)

    if args.target_client_id >= args.num_clients:
        raise ValueError(
            f"target_client_id={args.target_client_id} 必须小于 num_clients={args.num_clients}。"
        )
    if args.run and args.summary_only:
        raise ValueError("--run 和 --summary-only 不能同时使用。")

    commands = [(dataset_base, build_child_command(args, dataset_base)) for dataset_base in datasets]
    script_path = write_command_script(args, commands)

    print("Cross-dataset plan:")
    for dataset_base, cmd in commands:
        display_name = DATASET_DISPLAY_NAMES[dataset_base]
        print(f"  [{display_name}] num_classes={num_classes_for(args, dataset_base)}")
        print(f"    {shlex.join(wrap_command_for_shell(args, cmd))}")
    print(f"\nCommand script written to: {script_path}")

    if args.run:
        failures = []
        for dataset_base, cmd in commands:
            log_path = HETERO_ROOT / f"{dataset_base}_{args.algorithm}_{args.experiment_name}.log"
            print(f"\n===== Running [{dataset_base}] =====")
            print(f"Log file: {log_path}")
            try:
                run_logged_command(args, cmd, log_path)
            except subprocess.CalledProcessError as exc:
                failures.append((dataset_base, exc.returncode, cmd))
                print(f"[ERROR] [{dataset_base}] failed with return code {exc.returncode}.")
                if not args.continue_on_error:
                    raise

        if failures:
            print("\nFailed datasets:")
            for dataset_base, return_code, cmd in failures:
                print(f"  [{dataset_base}] return_code={return_code}: {shlex.join(wrap_command_for_shell(args, cmd))}")

    summary_path, rows = collect_combined_summary(args, datasets)
    analysis_md_path = generate_analysis(args, rows)
    if args.paper_assets and rows:
        generate_paper_assets(args, summary_path)
    print(f"\nCombined summary written to: {summary_path}")
    print(f"Analysis written to: {analysis_md_path}")


if __name__ == "__main__":
    main()
