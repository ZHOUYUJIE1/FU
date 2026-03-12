import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

DEFAULT_CLIENT_COUNTS = [10, 20, 50, 100]
DEFAULT_METHODS = ["fu", "fedau", "fedcsa", "fedosd", "retrain"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Watch client-count comparison progress. It periodically refreshes summary files and, "
            "once all runs are complete, generates publication assets automatically."
        )
    )
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--dataset-base", type=str, default="Cifar10")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--experiment-name", type=str, default="client_count_severe")
    parser.add_argument("--client-counts", nargs="+", type=int, default=DEFAULT_CLIENT_COUNTS)
    parser.add_argument("--methods", nargs="+", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--timeout-hours", type=float, default=0.0,
                        help="0 disables timeout; positive value sets max watch time.")
    return parser.parse_args()


def summary_path(args):
    return (
        PROJECT_ROOT
        / "system"
        / "heterogeneity_runs"
        / f"{args.dataset_base}_{args.algorithm}_{args.experiment_name}_summary.csv"
    )


def run_cmd(cmd):
    process = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def refresh_summary(args):
    cmd = [
        args.python,
        "-u",
        "system/run_client_count_comparison.py",
        "--client-counts",
        *(str(c) for c in args.client_counts),
        "--experiment-name",
        args.experiment_name,
        "--dataset-base",
        args.dataset_base,
        "--algorithm",
        args.algorithm,
        "--methods",
        *args.methods,
        "--summary-only",
    ]
    return run_cmd(cmd)


def compute_progress(args, df):
    if df.empty:
        return 0, len(args.client_counts) * len(args.methods)
    total = len(args.client_counts) * len(args.methods)
    done = int(df[df["status"] == "complete"].shape[0])
    return done, total


def build_assets(args):
    summary = summary_path(args)
    cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_client_count_paper_assets.py",
        "--summary",
        str(summary),
        "--figure-png",
        str(summary.parent / "client_count_paper_figure.png"),
        "--figure-pdf",
        str(summary.parent / "client_count_paper_figure.pdf"),
        "--table",
        str(summary.parent / "client_count_paper_table.tex"),
        "--analysis-md",
        str(summary.parent / "client_count_paper_analysis.md"),
        "--title",
        "Comparison Under Severe Heterogeneity with Varying Client Counts",
        "--caption",
        (
            "Results under severe Dirichlet heterogeneity on CIFAR-10 with different numbers of clients. "
            "Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, "
            "while higher average and retain-client accuracy indicate better utility retention."
        ),
    ]
    return run_cmd(cmd)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    start = time.time()

    while True:
        code, out = refresh_summary(args)
        if code != 0:
            print(f"[{now_str()}] summary refresh failed (code={code})")
            print(out)
            return 1

        summary = summary_path(args)
        if not summary.exists():
            print(f"[{now_str()}] summary not found yet: {summary}")
        else:
            df = pd.read_csv(summary)
            done, total = compute_progress(args, df)
            print(f"[{now_str()}] progress: {done}/{total} complete")
            if done >= total:
                asset_code, asset_out = build_assets(args)
                if asset_code != 0:
                    print(f"[{now_str()}] asset generation failed (code={asset_code})")
                    print(asset_out)
                    return 2
                print(asset_out.strip())
                print(f"[{now_str()}] all runs complete; assets generated.")
                return 0

        if args.timeout_hours > 0:
            elapsed_h = (time.time() - start) / 3600.0
            if elapsed_h >= args.timeout_hours:
                print(f"[{now_str()}] timeout reached ({args.timeout_hours}h), stop watching.")
                return 3

        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
