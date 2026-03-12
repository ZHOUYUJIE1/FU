import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "exp_configs"
LOG_ROOT = ROOT / "system" / "heterogeneity_runs"
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

if str(ROOT / "system") not in sys.path:
    sys.path.insert(0, str(ROOT / "system"))

import run_heterogeneity_comparison as comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Targeted FU tuning for mild/moderate Dirichlet heterogeneity."
    )
    parser.add_argument("--python", type=str, default=ZYJ_PYTHON if os.path.exists(ZYJ_PYTHON) else sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--goal", type=str, default="test")
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-lambda", type=float, default=0.6)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=LOG_ROOT / "fu_dir_tuning_summary.csv")
    parser.add_argument("--best-csv", type=Path, default=LOG_ROOT / "fu_dir_tuning_best.csv")
    parser.add_argument("--mild-penalties", nargs="+", type=float, default=[0.6, 0.65, 0.7])
    parser.add_argument("--mild-lr-scales", nargs="+", type=float, default=[0.9, 1.0])
    parser.add_argument("--mild-recovery-rounds", nargs="+", type=int, default=[2])
    parser.add_argument("--moderate-penalties", nargs="+", type=float, default=[0.5, 0.55, 0.6])
    parser.add_argument("--moderate-lr-scales", nargs="+", type=float, default=[0.9, 1.0])
    parser.add_argument("--moderate-recovery-rounds", nargs="+", type=int, default=[3, 4])
    return parser.parse_args()


def slug_float(value):
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def result_tag(level, penalty, lr_scale, recovery_rounds):
    return f"fu_{level}_tp{slug_float(penalty)}_lr{slug_float(lr_scale)}_r{recovery_rounds}"


def build_main_command(args, dataset_name, result_tag_value, saved_model_path, penalty, lr_scale, recovery_rounds):
    desired_args = dict(comparison.MAIN_DEFAULTS)
    desired_args.update(
        {
            "goal": args.goal,
            "device": args.device,
            "device_id": args.device_id,
            "dataset": dataset_name,
            "num_classes": 10,
            "model": "ResNet18",
            "batch_size": 32,
            "local_learning_rate": 0.005,
            "global_rounds": 100,
            "local_epochs": 1,
            "algorithm": args.algorithm,
            "join_ratio": 1.0,
            "num_clients": 10,
            "times": args.times,
            "eval_gap": 1,
            "random_seed": args.seed,
            "target_client_id": args.target_client_id,
            "result_tag": result_tag_value,
            "forget_strategy": "gradient_reversal",
            "recovery_rounds": recovery_rounds,
            "max_batches": 8,
            "num_processes": 4,
            "fu_select_best_recovery": True,
            "fu_recovery_target_penalty": penalty,
            "fu_recovery_lr_scale": lr_scale,
            "load_saved_model": True,
            "saved_model_path": str(saved_model_path),
        }
    )

    cmd = [args.python, "-u", "system/main.py"]
    for arg_name, flag in comparison.MAIN_ARG_FLAGS.items():
        if arg_name not in desired_args:
            continue
        value = desired_args[arg_name]
        default_value = comparison.MAIN_DEFAULTS[arg_name]
        if value == default_value:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.extend([flag, str(value)])
    return cmd


def maybe_wrap_env(args, cmd):
    if args.keep_ld_library_path:
        return list(cmd)
    return ["env", "-u", "LD_LIBRARY_PATH", *cmd]


def run_command(args, cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not args.keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def eval_path(prefix, dataset_name, result_tag_value):
    return RESULT_ROOT / f"{prefix}_{dataset_name}_FedAvg_test_1_{result_tag_value}.json"


def forget_path(dataset_name, result_tag_value):
    return RESULT_ROOT / f"forget_effect_{dataset_name}_FedAvg_test_1_{result_tag_value}.json"


def collect_metrics(dataset_name, result_tag_value, target_client_id):
    pre = read_json(eval_path("eval_pre_forget", dataset_name, result_tag_value))
    post = read_json(eval_path("eval_post_forget", dataset_name, result_tag_value))
    recovery_eval_path = eval_path("eval_post_recovery", dataset_name, result_tag_value)
    final_eval = read_json(recovery_eval_path) if recovery_eval_path.exists() else post
    forget = read_json(forget_path(dataset_name, result_tag_value))

    final_target = final_eval["client_accuracies"][target_client_id]
    final_avg = final_eval["average_accuracy"]
    final_retain = float(np.mean([acc for idx, acc in enumerate(final_eval["client_accuracies"]) if idx != target_client_id]))
    post_target = post["client_accuracies"][target_client_id]

    return {
        "pre_avg_acc": pre["average_accuracy"],
        "post_avg_acc": post["average_accuracy"],
        "post_target_acc": post_target,
        "final_avg_acc": final_avg,
        "final_target_acc": final_target,
        "final_retain_avg_acc": final_retain,
        "recovery_stage": forget.get("recovery_stage"),
        "recovery_rounds_executed": forget.get("recovery_rounds"),
    }


def candidate_grid(args):
    specs = []
    for penalty in args.mild_penalties:
        for lr_scale in args.mild_lr_scales:
            for recovery_rounds in args.mild_recovery_rounds:
                specs.append(("mild", penalty, lr_scale, recovery_rounds))
    for penalty in args.moderate_penalties:
        for lr_scale in args.moderate_lr_scales:
            for recovery_rounds in args.moderate_recovery_rounds:
                specs.append(("moderate", penalty, lr_scale, recovery_rounds))
    return specs


def base_model_path(level):
    dataset_name = f"Cifar10_hetero_{level}"
    return RESULT_ROOT / f"global_model_{dataset_name}_FedAvg_test_1_fu_{level}.pt"


def summary_rows(args):
    rows = []
    for level, penalty, lr_scale, recovery_rounds in candidate_grid(args):
        dataset_name = f"Cifar10_hetero_{level}"
        tag = result_tag(level, penalty, lr_scale, recovery_rounds)
        row = {
            "level": level,
            "dataset": dataset_name,
            "result_tag": tag,
            "penalty": penalty,
            "lr_scale": lr_scale,
            "recovery_rounds": recovery_rounds,
        }
        try:
            metrics = collect_metrics(dataset_name, tag, args.target_client_id)
        except FileNotFoundError:
            row["status"] = "missing"
            rows.append(row)
            continue
        row["status"] = "complete"
        row.update(metrics)
        row["score"] = row["final_avg_acc"] - args.score_lambda * row["final_target_acc"]
        rows.append(row)
    return rows


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


def select_best_rows(rows):
    best_rows = []
    for level in ("mild", "moderate"):
        candidates = [row for row in rows if row["level"] == level and row.get("status") == "complete"]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                row["score"],
                row["final_avg_acc"],
                -row["final_target_acc"],
                row["final_retain_avg_acc"],
            ),
            reverse=True,
        )
        best_rows.append(candidates[0])
    return best_rows


def main():
    args = parse_args()
    specs = candidate_grid(args)
    print("FU tuning plan:")
    for level, penalty, lr_scale, recovery_rounds in specs:
        tag = result_tag(level, penalty, lr_scale, recovery_rounds)
        print(
            f"[{level}] tag={tag} penalty={penalty} lr_scale={lr_scale} "
            f"recovery_rounds={recovery_rounds}"
        )

    if args.run:
        for level, penalty, lr_scale, recovery_rounds in specs:
            dataset_name = f"Cifar10_hetero_{level}"
            tag = result_tag(level, penalty, lr_scale, recovery_rounds)
            result_file = forget_path(dataset_name, tag)
            if args.skip_existing and result_file.exists():
                print(f"Skipping {tag} because {result_file.name} already exists.")
                continue

            checkpoint = base_model_path(level)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing base model for {level}: {checkpoint}")

            cmd = build_main_command(args, dataset_name, tag, checkpoint, penalty, lr_scale, recovery_rounds)
            log_path = LOG_ROOT / dataset_name / f"{tag}.log"
            print(f"\n===== Running {tag} =====")
            print("Command:", " ".join(maybe_wrap_env(args, cmd)))
            print("Log file:", log_path)
            run_command(args, cmd, log_path)

    rows = summary_rows(args)
    rows.sort(key=lambda row: (row["level"], -(row.get("score") or -1e9)))
    write_csv(args.summary_csv, rows)
    best_rows = select_best_rows(rows)
    write_csv(args.best_csv, best_rows)

    print(f"\nSummary written to: {args.summary_csv}")
    print(f"Best rows written to: {args.best_csv}")
    for row in best_rows:
        print(
            f"BEST [{row['level']}] {row['result_tag']} "
            f"avg={row['final_avg_acc']:.4f} target={row['final_target_acc']:.4f} "
            f"retain={row['final_retain_avg_acc']:.4f} score={row['score']:.4f}"
        )


if __name__ == "__main__":
    main()
