import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = PROJECT_ROOT / "system"
HETERO_ROOT = SYSTEM_ROOT / "heterogeneity_runs"
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"


DATASET_CONFIGS = {
    "GTSRB": {
        "num_classes": 43,
        "sgd_momentum": 0.9,
        "weight_decay": 5e-4,
    },
    "TinyImagenet": {
        "num_classes": 200,
        "sgd_momentum": 0.9,
        "weight_decay": 5e-4,
    },
    "STL10": {
        "num_classes": 10,
        "sgd_momentum": 0.9,
        "weight_decay": 5e-4,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the current severe benchmark to finish, then launch severe "
            "FU/FedAU/FedCSA/FedOSD/Retrain comparisons on GTSRB, TinyImagenet, and STL10."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=["GTSRB", "TinyImagenet", "STL10"],
    )
    parser.add_argument(
        "--wait-pattern",
        type=str,
        default="Cifar100_FedAvg_claim_c100_severe_commands.sh",
        help="Wait until no running process matches this pattern before starting the queue.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=ZYJ_PYTHON if os.path.exists(ZYJ_PYTHON) else sys.executable,
    )
    parser.add_argument("--goal", type=str, default="test")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dataset-prefix", type=str, default="hetero_clean_s42")
    parser.add_argument("--experiment-name", type=str, default="heterogeneity_clean_seed42_severe")
    parser.add_argument("--partition", type=str, default="dir", choices=["dir"])
    parser.add_argument("--alpha-severe", type=float, default=0.1)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-learning-rate", type=float, default=0.02)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--eval-gap", type=int, default=1)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-rounds", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_logged(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def matching_processes(pattern):
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if str(Path(__file__).name) in line:
            continue
        lines.append(line)
    return lines


def wait_for_clear(pattern, poll_interval, queue_log):
    while True:
        matches = matching_processes(pattern)
        if not matches:
            message = f"[Queue] wait pattern cleared: {pattern}"
            print(message)
            with open(queue_log, "a", encoding="utf-8") as f:
                f.write(message + "\n")
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"[Queue] {timestamp} waiting for current experiment to finish. "
            f"pattern={pattern} matches={len(matches)}"
        )
        print(message)
        with open(queue_log, "a", encoding="utf-8") as f:
            f.write(message + "\n")
            for line in matches:
                f.write(f"  {line}\n")
        time.sleep(max(5, int(poll_interval)))


def dataset_name(dataset_base, dataset_prefix):
    return f"{dataset_base}_{dataset_prefix}_severe"


def command_script_path(args, dataset_base):
    return HETERO_ROOT / f"{dataset_base}_{args.algorithm}_{args.experiment_name}_commands.sh"


def summary_path(args, dataset_base):
    return HETERO_ROOT / f"{dataset_base}_{args.algorithm}_{args.experiment_name}_summary.csv"


def dataset_log_dir(args, dataset_base):
    return HETERO_ROOT / dataset_name(dataset_base, args.dataset_prefix)


def build_dataset_command(args, dataset_base):
    config = DATASET_CONFIGS[dataset_base]
    return [
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
        "--levels", "severe",
        "--methods", "fu", "fedau", "fedcsa", "fedosd", "retrain",
        "--partition", args.partition,
        "--alpha-severe", str(args.alpha_severe),
        "--num-clients", str(args.num_clients),
        "--num-classes", str(config["num_classes"]),
        "--global-rounds", str(args.global_rounds),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--local-learning-rate", str(args.local_learning_rate),
        "--sgd-momentum", str(config["sgd_momentum"]),
        "--weight-decay", str(config["weight_decay"]),
        "--join-ratio", str(args.join_ratio),
        "--times", str(args.times),
        "--eval-gap", str(args.eval_gap),
        "--target-client-id", str(args.target_client_id),
        "--seed", str(args.seed),
        "--recovery-rounds", str(args.recovery_rounds),
        "--max-batches", str(args.max_batches),
        "--num-processes", str(args.num_processes),
        "--enable-backdoor-attack", "false",
        "--skip-existing",
        "--resume-incomplete",
        "--continue-on-error",
        "--run",
    ]


def stage_dataset_artifacts(args, dataset_base, cmd):
    log_dir = dataset_log_dir(args, dataset_base)
    log_dir.mkdir(parents=True, exist_ok=True)

    src_summary = summary_path(args, dataset_base)
    if src_summary.exists():
        shutil.copy2(src_summary, log_dir / "summary.csv")

    src_commands = command_script_path(args, dataset_base)
    if src_commands.exists():
        shutil.copy2(src_commands, log_dir / "commands.sh")

    metadata = {
        "dataset_base": dataset_base,
        "dataset_name": dataset_name(dataset_base, args.dataset_prefix),
        "command": cmd,
        "summary_path": str(src_summary),
        "command_script_path": str(src_commands),
        "runner_log_path": str(log_dir / "runner.log"),
        "method_logs": {
            method: str(log_dir / f"{method}.log")
            for method in ("fu", "fedau", "fedcsa", "fedosd", "retrain")
        },
    }
    with open(log_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    HETERO_ROOT.mkdir(parents=True, exist_ok=True)
    queue_log = HETERO_ROOT / "severe_three_datasets_queue.log"

    queue_header = (
        f"[Queue] starting with datasets={args.datasets}, wait_pattern={args.wait_pattern}, "
        f"dry_run={args.dry_run}"
    )
    print(queue_header)
    with open(queue_log, "a", encoding="utf-8") as f:
        f.write(queue_header + "\n")

    if args.wait_pattern.strip():
        wait_for_clear(args.wait_pattern.strip(), args.poll_interval, queue_log)

    for dataset_base in args.datasets:
        cmd = build_dataset_command(args, dataset_base)
        runner_log = dataset_log_dir(args, dataset_base) / "runner.log"
        header = f"[Queue] launching {dataset_base}: {shlex.join(cmd)}"
        print(header)
        with open(queue_log, "a", encoding="utf-8") as f:
            f.write(header + "\n")

        if args.dry_run:
            stage_dataset_artifacts(args, dataset_base, cmd)
            continue

        return_code = run_logged(cmd, runner_log, PROJECT_ROOT)
        if return_code != 0:
            failure = f"[Queue] dataset {dataset_base} failed with return code {return_code}"
            print(failure)
            with open(queue_log, "a", encoding="utf-8") as f:
                f.write(failure + "\n")
            sys.exit(return_code)

        stage_dataset_artifacts(args, dataset_base, cmd)
        success = f"[Queue] dataset {dataset_base} completed successfully"
        print(success)
        with open(queue_log, "a", encoding="utf-8") as f:
            f.write(success + "\n")

    done = "[Queue] all requested datasets completed"
    print(done)
    with open(queue_log, "a", encoding="utf-8") as f:
        f.write(done + "\n")


if __name__ == "__main__":
    main()
