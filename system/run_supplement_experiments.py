import argparse
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


SUITE_ORDER = ["dir", "pat", "count", "assets"]

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

DEFAULT_NUM_CLASSES = {
    "Cifar10": 10,
    "Cifar100": 100,
    "Digit5": 10,
    "EMNIST": 10,
    "FEMNIST": 62,
    "DomainNet": 345,
    "HAR": 6,
    "PAMAP2": 12,
}

DATASET_DISPLAY_NAMES = {
    "Cifar10": "CIFAR-10",
    "Cifar100": "CIFAR-100",
    "Digit5": "Digits-Five",
    "EMNIST": "EMNIST",
    "FEMNIST": "FEMNIST",
    "DomainNet": "DomainNet",
    "HAR": "HAR",
    "PAMAP2": "PAMAP2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run supplementary federated unlearning experiments. The script reuses the "
            "existing heterogeneity and client-count pipelines while isolating outputs "
            "with a configurable experiment tag."
        )
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITE_ORDER + ["all"],
        default=["all"],
        help="Supplementary suites to execute sequentially.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the planned commands. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Collect existing results into summaries/assets without launching training.",
    )
    parser.add_argument("--dataset-base", type=str, default="Cifar10")
    parser.add_argument("--dataset-display-name", type=str, default="")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--goal", type=str, default="test")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument(
        "--python",
        type=str,
        default=ZYJ_PYTHON if os.path.exists(ZYJ_PYTHON) else sys.executable,
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default="supp",
        help=(
            "Namespace used for experiment-name and dataset-prefix generation. "
            "Use 'paper' to point at the existing paper naming convention."
        ),
    )
    parser.add_argument(
        "--artifact-tag",
        type=str,
        default="",
        help="Optional namespace for generated figures/tables. Defaults to experiment-tag.",
    )
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=None)
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
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=hetero.METHOD_ORDER,
        default=list(hetero.METHOD_ORDER),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=hetero.LEVEL_ORDER,
        default=list(hetero.LEVEL_ORDER),
        help="Levels used for the Dirichlet and pathological supplementary suites.",
    )
    parser.add_argument("--client-counts", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--count-level", choices=hetero.LEVEL_ORDER, default="severe")
    parser.add_argument("--count-alpha", type=float, default=None)
    parser.add_argument("--count-class-per-client", type=int, default=None)

    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--balance", dest="balance", action="store_true")
    balance_group.add_argument("--unbalance", dest="balance", action="store_false")
    parser.set_defaults(balance=True)

    return parser.parse_args()


def normalize_suites(suites):
    if "all" in suites:
        return list(SUITE_ORDER)
    return [suite for suite in SUITE_ORDER if suite in suites]


def normalize_tag(tag):
    return str(tag).strip().replace(os.sep, "_").replace(" ", "_")


def experiment_tag(args):
    return normalize_tag(args.experiment_tag) or "supp"


def artifact_tag(args):
    tag = normalize_tag(args.artifact_tag)
    if tag:
        return tag
    return experiment_tag(args)


def is_paper_tag(args):
    return experiment_tag(args).lower() == "paper"


def dataset_display_name(args):
    if args.dataset_display_name.strip():
        return args.dataset_display_name.strip()
    return DATASET_DISPLAY_NAMES.get(args.dataset_base, args.dataset_base)


def resolve_num_classes(args):
    if args.num_classes is not None:
        return int(args.num_classes)
    if args.dataset_base in DEFAULT_NUM_CLASSES:
        return DEFAULT_NUM_CLASSES[args.dataset_base]
    raise ValueError(
        f"Unknown default num_classes for dataset-base={args.dataset_base}. "
        "Please pass --num-classes explicitly."
    )


def dir_experiment_name(args):
    if is_paper_tag(args):
        return "heterogeneity"
    return f"{experiment_tag(args)}_heterogeneity"


def dir_dataset_prefix(args):
    if is_paper_tag(args):
        return "hetero"
    return f"{experiment_tag(args)}_hetero"


def pat_experiment_name(args):
    if is_paper_tag(args):
        return "pathological_label_skew"
    return f"{experiment_tag(args)}_pathological_label_skew"


def pat_dataset_prefix(args):
    if is_paper_tag(args):
        return "pat"
    return f"{experiment_tag(args)}_pat"


def count_experiment_name(args):
    if is_paper_tag(args) and args.count_level == "severe":
        return "client_count_severe"
    return f"{experiment_tag(args)}_client_count_{args.count_level}"


def count_dataset_prefix_base(args):
    if is_paper_tag(args) and args.count_level == "severe":
        return "clientscale"
    return f"{experiment_tag(args)}_clientscale"


def asset_prefix(args, suite_name):
    return f"{args.dataset_base}_{args.algorithm}_{artifact_tag(args)}_{suite_name}"


def run_mode_flags(args):
    if args.summary_only:
        return ["--summary-only"]
    if args.run:
        return ["--run"]
    return []


def common_flags(args):
    return [
        "--dataset-base", args.dataset_base,
        "--goal", args.goal,
        "--algorithm", args.algorithm,
        "--model", args.model,
        "--device", args.device,
        "--device-id", args.device_id,
        "--python", args.python,
        "--num-clients", str(args.num_clients),
        "--num-classes", str(resolve_num_classes(args)),
        "--global-rounds", str(args.global_rounds),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--local-learning-rate", str(args.local_learning_rate),
        "--join-ratio", str(args.join_ratio),
        "--times", str(args.times),
        "--eval-gap", str(args.eval_gap),
        "--target-client-id", str(args.target_client_id),
        "--seed", str(args.seed),
        "--recovery-rounds", str(args.recovery_rounds),
        "--max-batches", str(args.max_batches),
        "--num-processes", str(args.num_processes),
        "--methods", *args.methods,
        *run_mode_flags(args),
    ]


def dir_command(args):
    return [
        args.python,
        "-u",
        "system/run_heterogeneity_comparison.py",
        *common_flags(args),
        "--dataset-prefix", dir_dataset_prefix(args),
        "--experiment-name", dir_experiment_name(args),
        "--partition", "dir",
        "--levels", *args.levels,
        "--alpha-mild", str(DEFAULT_LEVEL_ALPHA["mild"]),
        "--alpha-moderate", str(DEFAULT_LEVEL_ALPHA["moderate"]),
        "--alpha-severe", str(DEFAULT_LEVEL_ALPHA["severe"]),
        "--balance" if args.balance else "--unbalance",
        *(["--keep-ld-library-path"] if args.keep_ld_library_path else []),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--resume-incomplete"] if args.resume_incomplete else []),
        *(["--continue-on-error"] if args.continue_on_error else []),
        *(["--force-regenerate"] if args.force_regenerate else []),
    ]


def pat_command(args):
    return [
        args.python,
        "-u",
        "system/run_heterogeneity_comparison.py",
        *common_flags(args),
        "--dataset-prefix", pat_dataset_prefix(args),
        "--experiment-name", pat_experiment_name(args),
        "--partition", "pat",
        "--levels", *args.levels,
        "--class-per-client-mild", str(DEFAULT_LEVEL_CLASS_PER_CLIENT["mild"]),
        "--class-per-client-moderate", str(DEFAULT_LEVEL_CLASS_PER_CLIENT["moderate"]),
        "--class-per-client-severe", str(DEFAULT_LEVEL_CLASS_PER_CLIENT["severe"]),
        "--alpha-mild", "1.0",
        "--alpha-moderate", "1.0",
        "--alpha-severe", "1.0",
        "--balance" if args.balance else "--unbalance",
        *(["--keep-ld-library-path"] if args.keep_ld_library_path else []),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--resume-incomplete"] if args.resume_incomplete else []),
        *(["--continue-on-error"] if args.continue_on_error else []),
        *(["--force-regenerate"] if args.force_regenerate else []),
    ]


def selected_count_alpha(args):
    if args.count_alpha is not None:
        return float(args.count_alpha)
    return DEFAULT_LEVEL_ALPHA[args.count_level]


def count_command(args):
    cmd = [
        args.python,
        "-u",
        "system/run_client_count_comparison.py",
        "--client-counts", *(str(count) for count in args.client_counts),
        "--level", args.count_level,
        "--methods", *args.methods,
        "--dataset-base", args.dataset_base,
        "--goal", args.goal,
        "--algorithm", args.algorithm,
        "--model", args.model,
        "--device", args.device,
        "--device-id", args.device_id,
        "--python", args.python,
        "--experiment-name", count_experiment_name(args),
        "--dataset-prefix-base", count_dataset_prefix_base(args),
        "--partition", "dir",
        "--alpha", str(selected_count_alpha(args)),
        "--target-client-id", str(args.target_client_id),
        "--balance" if args.balance else "--unbalance",
        *run_mode_flags(args),
        *(["--keep-ld-library-path"] if args.keep_ld_library_path else []),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--resume-incomplete"] if args.resume_incomplete else []),
        *(["--continue-on-error"] if args.continue_on_error else []),
        *(["--force-regenerate"] if args.force_regenerate else []),
    ]
    if args.count_class_per_client is not None:
        cmd.extend(["--class-per-client", str(args.count_class_per_client)])
    return cmd


def dir_summary_path(args):
    return HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_{dir_experiment_name(args)}_summary.csv"


def pat_summary_path(args):
    return HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_{pat_experiment_name(args)}_summary.csv"


def count_summary_path(args):
    return HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_{count_experiment_name(args)}_summary.csv"


def dir_assets_command(args):
    prefix = asset_prefix(args, "dir")
    display_name = dataset_display_name(args)
    return [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_heterogeneity_paper_assets.py",
        "--summary", str(dir_summary_path(args)),
        "--figure-png", str(HETERO_ROOT / f"{prefix}_figure.png"),
        "--figure-pdf", str(HETERO_ROOT / f"{prefix}_figure.pdf"),
        "--table", str(HETERO_ROOT / f"{prefix}_table.tex"),
        "--title", f"Supplementary Dirichlet Heterogeneity on {display_name}",
        "--caption",
        (
            f"Supplementary results under mild, moderate, and severe Dirichlet heterogeneity on {display_name}. "
            "Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, "
            "while higher average and retain-client accuracy indicate better utility retention."
        ),
    ]


def pat_assets_command(args):
    prefix = asset_prefix(args, "pat")
    display_name = dataset_display_name(args)
    return [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_heterogeneity_paper_assets.py",
        "--summary", str(pat_summary_path(args)),
        "--figure-png", str(HETERO_ROOT / f"{prefix}_figure.png"),
        "--figure-pdf", str(HETERO_ROOT / f"{prefix}_figure.pdf"),
        "--table", str(HETERO_ROOT / f"{prefix}_table.tex"),
        "--title", f"Supplementary Pathological Label Skew on {display_name}",
        "--caption",
        (
            f"Supplementary results under mild, moderate, and severe pathological label skew on {display_name}. "
            "Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, "
            "while higher average and retain-client accuracy indicate better utility retention."
        ),
    ]


def count_assets_command(args):
    prefix = asset_prefix(args, "client_count")
    display_name = dataset_display_name(args)
    return [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_client_count_paper_assets.py",
        "--summary", str(count_summary_path(args)),
        "--figure-png", str(HETERO_ROOT / f"{prefix}_figure.png"),
        "--figure-pdf", str(HETERO_ROOT / f"{prefix}_figure.pdf"),
        "--table", str(HETERO_ROOT / f"{prefix}_table.tex"),
        "--analysis-md", str(HETERO_ROOT / f"{prefix}_analysis.md"),
        "--title", f"Supplementary {args.count_level.capitalize()} Heterogeneity with Varying Client Counts on {display_name}",
        "--caption",
        (
            f"Supplementary results under {args.count_level} Dirichlet heterogeneity on {display_name} with varying "
            "numbers of clients. Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger "
            "forgetting/privacy protection, while higher average and retain-client accuracy indicate better utility retention."
        ),
    ]


def wrap_env(args, cmd):
    if args.keep_ld_library_path:
        return list(cmd)
    return ["env", "-u", "LD_LIBRARY_PATH", *cmd]


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


def log_root(args):
    return HETERO_ROOT / "supplement_logs" / artifact_tag(args)


def planned_steps(args, suites):
    steps = []
    if "dir" in suites:
        steps.append(("dir", dir_command(args), None))
    if "pat" in suites:
        steps.append(("pat", pat_command(args), None))
    if "count" in suites:
        steps.append(("count", count_command(args), None))
    if "assets" in suites:
        steps.append(("assets-dir", dir_assets_command(args), dir_summary_path(args)))
        steps.append(("assets-pat", pat_assets_command(args), pat_summary_path(args)))
        steps.append(("assets-count", count_assets_command(args), count_summary_path(args)))
    return steps


def print_plan(args, steps):
    print("Supplement experiment pipeline:")
    print(
        f"  dataset={args.dataset_base} ({dataset_display_name(args)}), "
        f"algorithm={args.algorithm}, tag={experiment_tag(args)}, artifact_tag={artifact_tag(args)}"
    )
    print(
        f"  modes: run={args.run}, summary_only={args.summary_only}, "
        f"num_clients={args.num_clients}, num_classes={resolve_num_classes(args)}"
    )
    if "count" in normalize_suites(args.suites):
        print(
            f"  client-count sweep: counts={args.client_counts}, level={args.count_level}, "
            f"alpha={selected_count_alpha(args)}"
        )
    print("\nCommands:")
    for name, cmd, dependency in steps:
        if dependency is not None:
            print(f"[{name}] requires {dependency}")
        print(f"[{name}] {shlex.join(wrap_env(args, cmd))}")


def main():
    args = parse_args()
    if args.run and args.summary_only:
        raise ValueError("--run and --summary-only cannot be used together.")

    suites = normalize_suites(args.suites)
    steps = planned_steps(args, suites)
    print_plan(args, steps)

    if not args.run and not args.summary_only:
        return

    failures = []
    for name, cmd, dependency in steps:
        if dependency is not None and not Path(dependency).exists():
            message = f"Missing dependency for {name}: {dependency}"
            if args.continue_on_error:
                print(f"[WARN] {message}")
                failures.append((name, message))
                continue
            raise FileNotFoundError(message)

        log_path = log_root(args) / f"{name}.log"
        print(f"\n===== Running {name} =====")
        print(f"Log file: {log_path}")
        try:
            run_logged_command(args, cmd, log_path)
        except subprocess.CalledProcessError as exc:
            failures.append((name, f"return_code={exc.returncode}"))
            print(f"[ERROR] {name} failed with return code {exc.returncode}")
            if not args.continue_on_error:
                raise

    if failures:
        print("\nPipeline failures:")
        for name, message in failures:
            print(f"  {name}: {message}")


if __name__ == "__main__":
    main()
