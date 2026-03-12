import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = PROJECT_ROOT / "system"
LOG_ROOT = SYSTEM_ROOT / "heterogeneity_runs"
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

import run_heterogeneity_comparison as comparison


DEFAULT_LEVEL_ALPHA = {
    "mild": 1.0,
    "moderate": 0.3,
    "severe": 0.1,
}

DEFAULT_LEVEL_CLASS_PER_CLIENT = {
    "mild": 5,
    "moderate": 3,
    "severe": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run FU/FedAU/FedOSD/FedCSA/Retrain comparisons under one heterogeneity level "
            "while sweeping the number of clients. Results are isolated from the existing "
            "heterogeneity experiments by using per-client-count dataset prefixes."
        )
    )
    parser.add_argument("--client-counts", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--level", choices=comparison.LEVEL_ORDER, default="severe")
    parser.add_argument("--methods", nargs="+", choices=comparison.METHOD_ORDER, default=comparison.METHOD_ORDER)
    parser.add_argument("--dataset-base", type=str, default="Cifar10")
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="client_count_severe",
        help="Base output stem. The script appends _c<num_clients> for each sweep point.",
    )
    parser.add_argument(
        "--dataset-prefix-base",
        type=str,
        default="clientscale",
        help="Base dataset infix. The script appends _c<num_clients> for each sweep point.",
    )
    parser.add_argument("--partition", choices=["dir", "pat", "exdir"], default="dir")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override heterogeneity strength for the selected level. Default follows the paper setting.",
    )
    parser.add_argument(
        "--class-per-client",
        type=int,
        default=None,
        help="Only used when partition is pat/exdir.",
    )
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--keep-ld-library-path", action="store_true")

    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--balance", dest="balance", action="store_true")
    balance_group.add_argument("--unbalance", dest="balance", action="store_false")
    parser.set_defaults(balance=True)

    return parser.parse_known_args()


def normalize_client_counts(client_counts):
    normalized = []
    seen = set()
    for count in client_counts:
        if count <= 0:
            raise ValueError(f"Client count must be positive, got {count}.")
        if count not in seen:
            normalized.append(int(count))
            seen.add(count)
    return normalized


def selected_alpha(args):
    if args.alpha is not None:
        return float(args.alpha)
    return DEFAULT_LEVEL_ALPHA[args.level]


def selected_class_per_client(args):
    if args.class_per_client is not None:
        return int(args.class_per_client)
    return DEFAULT_LEVEL_CLASS_PER_CLIENT[args.level]


def experiment_name_for_count(args, client_count):
    return f"{args.experiment_name}_c{client_count}"


def dataset_prefix_for_count(args, client_count):
    return f"{args.dataset_prefix_base}_c{client_count}"


def dataset_name_for_count(args, client_count):
    return f"{args.dataset_base}_{dataset_prefix_for_count(args, client_count)}_{args.level}"


def per_count_summary_path(args, client_count):
    return LOG_ROOT / f"{args.dataset_base}_{args.algorithm}_{experiment_name_for_count(args, client_count)}_summary.csv"


def combined_summary_path(args):
    return LOG_ROOT / f"{args.dataset_base}_{args.algorithm}_{args.experiment_name}_summary.csv"


def combined_command_script_path(args):
    return LOG_ROOT / f"{args.dataset_base}_{args.algorithm}_{args.experiment_name}_commands.sh"


def count_log_path(args, client_count):
    return LOG_ROOT / dataset_name_for_count(args, client_count) / "client_count_comparison.log"


def validate_args(args, client_counts):
    if args.run and args.summary_only:
        raise ValueError("--run and --summary-only cannot be used together.")
    for client_count in client_counts:
        if args.target_client_id >= client_count:
            raise ValueError(
                f"target_client_id={args.target_client_id} is invalid for num_clients={client_count}. "
                "Choose a larger client count or lower target_client_id."
            )


def build_child_command(args, client_count, passthrough_args):
    cmd = [
        args.python,
        "-u",
        "system/run_heterogeneity_comparison.py",
        "--dataset-base",
        args.dataset_base,
        "--dataset-prefix",
        dataset_prefix_for_count(args, client_count),
        "--experiment-name",
        experiment_name_for_count(args, client_count),
        "--goal",
        args.goal,
        "--algorithm",
        args.algorithm,
        "--model",
        args.model,
        "--device",
        args.device,
        "--device-id",
        args.device_id,
        "--python",
        args.python,
        "--levels",
        args.level,
        "--methods",
        *args.methods,
        "--num-clients",
        str(client_count),
        "--target-client-id",
        str(args.target_client_id),
        "--partition",
        args.partition,
        f"--alpha-{args.level}",
        str(selected_alpha(args)),
    ]

    if args.partition in {"pat", "exdir"}:
        cmd.extend([f"--class-per-client-{args.level}", str(selected_class_per_client(args))])

    cmd.append("--balance" if args.balance else "--unbalance")

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

    cmd.extend(passthrough_args)
    return cmd


def wrap_command_for_shell(args, cmd):
    if args.keep_ld_library_path:
        return list(cmd)
    return ["env", "-u", "LD_LIBRARY_PATH", *cmd]


def write_master_command_script(args, commands):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    script_path = combined_command_script_path(args)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"cd {shlex.quote(str(PROJECT_ROOT))}\n\n")
        for cmd in commands:
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


def method_sort_key(method_name):
    if method_name in comparison.METHOD_ORDER:
        return comparison.METHOD_ORDER.index(method_name)
    return len(comparison.METHOD_ORDER)


def merge_summaries(args, client_counts):
    rows = []
    for client_count in client_counts:
        summary_path = per_count_summary_path(args, client_count)
        for row in read_summary_rows(summary_path):
            merged_row = dict(row)
            merged_row["client_count"] = client_count
            rows.append(merged_row)

    output_path = combined_summary_path(args)
    if not rows:
        if output_path.exists():
            output_path.unlink()
        return output_path

    rows.sort(
        key=lambda row: (
            int(row.get("client_count") or row.get("num_clients") or 0),
            comparison.LEVEL_ORDER.index(row["level"]) if row.get("level") in comparison.LEVEL_ORDER else len(comparison.LEVEL_ORDER),
            method_sort_key(row.get("method")),
        )
    )

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_plan(args, client_counts, commands):
    print("Client-count comparison plan:")
    print(
        f"  level={args.level}, partition={args.partition}, alpha={selected_alpha(args)}, "
        f"target_client_id={args.target_client_id}"
    )
    print(f"  methods={', '.join(comparison.ordered_methods(args.methods))}")
    if args.partition in {"pat", "exdir"}:
        print(f"  class_per_client={selected_class_per_client(args)}")
    print("  isolated datasets/results:")
    for client_count in client_counts:
        print(
            f"    num_clients={client_count}: "
            f"dataset={dataset_name_for_count(args, client_count)}, "
            f"summary={per_count_summary_path(args, client_count).name}"
        )

    print("\nCommands:")
    for client_count, cmd in zip(client_counts, commands):
        print(f"[num_clients={client_count}] {shlex.join(wrap_command_for_shell(args, cmd))}")


def main():
    args, passthrough_args = parse_args()
    client_counts = normalize_client_counts(args.client_counts)
    validate_args(args, client_counts)

    commands = [build_child_command(args, client_count, passthrough_args) for client_count in client_counts]
    script_path = write_master_command_script(args, commands)
    print_plan(args, client_counts, commands)
    print(f"\nMaster command script written to: {script_path}")

    failures = []
    if args.run or args.summary_only:
        for client_count, cmd in zip(client_counts, commands):
            log_path = count_log_path(args, client_count)
            print(f"\n===== Processing num_clients={client_count} =====")
            print(f"Log file: {log_path}")
            try:
                run_logged_command(args, cmd, log_path)
            except subprocess.CalledProcessError as exc:
                failures.append((client_count, exc.returncode, cmd))
                print(f"\n[ERROR] [num_clients={client_count}] failed with return code {exc.returncode}.")
                if not args.continue_on_error:
                    raise

    summary_path = merge_summaries(args, client_counts)
    print(f"\nCombined summary written to: {summary_path}")

    if failures:
        print("\nFailed runs:")
        for client_count, return_code, cmd in failures:
            print(f"  [num_clients={client_count}] return_code={return_code}: {shlex.join(wrap_command_for_shell(args, cmd))}")


if __name__ == "__main__":
    main()
