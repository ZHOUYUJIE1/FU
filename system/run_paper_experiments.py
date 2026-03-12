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

import run_heterogeneity_comparison as comparison


DIR_FU_FINAL_TAGS = {
    "mild": {
        "result_tag": "fu_mild_tp06_lr09_r2",
        "fu_recovery_target_penalty": 0.6,
        "fu_recovery_lr_scale": 0.9,
        "recovery_rounds": 2,
    },
    "moderate": {
        "result_tag": "fu_moderate_tp055_lr1_r4",
        "fu_recovery_target_penalty": 0.55,
        "fu_recovery_lr_scale": 1.0,
        "recovery_rounds": 4,
    },
    "severe": {
        "result_tag": "fu_severe_final",
        "fu_recovery_target_penalty": 0.5,
        "fu_recovery_lr_scale": 1.0,
    },
}

SUITE_ORDER = [
    "dir-main",
    "dir-fu-final",
    "dir-assets",
    "pat-main",
    "pat-assets",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the paper experiment pipeline for heterogeneous federated unlearning."
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITE_ORDER + ["all"],
        default=["all"],
        help="Pipeline stages to execute sequentially.",
    )
    parser.add_argument("--run", action="store_true",
                        help="Execute the pipeline. Without this flag, only print the planned commands.")
    parser.add_argument("--dataset-base", type=str, default="Cifar10")
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
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-learning-rate", type=float, default=0.005)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--eval-gap", type=int, default=1)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-rounds", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=4)
    return parser.parse_args()


def normalized_suites(suites):
    if "all" in suites:
        return list(SUITE_ORDER)
    return [suite for suite in SUITE_ORDER if suite in suites]


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


def append_common_comparison_args(args):
    cmd = [
        "--dataset-base", args.dataset_base,
        "--goal", args.goal,
        "--algorithm", args.algorithm,
        "--model", args.model,
        "--device", args.device,
        "--device-id", args.device_id,
        "--python", args.python,
        "--num-clients", str(args.num_clients),
        "--num-classes", str(args.num_classes),
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
    ]
    if args.keep_ld_library_path:
        cmd.append("--keep-ld-library-path")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.resume_incomplete:
        cmd.append("--resume-incomplete")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.force_regenerate:
        cmd.append("--force-regenerate")
    if args.run:
        cmd.append("--run")
    return cmd


def comparison_command(args, extra_args):
    return [
        args.python,
        "-u",
        "system/run_heterogeneity_comparison.py",
        *append_common_comparison_args(args),
        *extra_args,
    ]


def build_main_command(args, overrides):
    desired_args = dict(comparison.MAIN_DEFAULTS)
    desired_args.update(
        {
            "goal": args.goal,
            "device": args.device,
            "device_id": args.device_id,
            "num_classes": args.num_classes,
            "model": args.model,
            "batch_size": args.batch_size,
            "local_learning_rate": args.local_learning_rate,
            "global_rounds": args.global_rounds,
            "local_epochs": args.local_epochs,
            "algorithm": args.algorithm,
            "join_ratio": args.join_ratio,
            "num_clients": args.num_clients,
            "times": args.times,
            "eval_gap": args.eval_gap,
            "random_seed": args.seed,
            "target_client_id": args.target_client_id,
            "recovery_rounds": args.recovery_rounds,
            "max_batches": args.max_batches,
            "num_processes": args.num_processes,
            "fu_select_best_recovery": True,
        }
    )
    desired_args.update(overrides)

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


def dir_main_suite(args):
    return comparison_command(
        args,
        [
            "--experiment-name", "heterogeneity",
            "--dataset-prefix", "hetero",
            "--partition", "dir",
            "--levels", "mild", "moderate", "severe",
            "--methods", "fu", "fedau", "fedcsa", "fedosd", "retrain",
            "--alpha-mild", "1.0",
            "--alpha-moderate", "0.3",
            "--alpha-severe", "0.1",
        ],
    )


def dir_fu_final_commands(args):
    commands = []
    for level, config in DIR_FU_FINAL_TAGS.items():
        dataset_name = f"{args.dataset_base}_hetero_{level}"
        base_model_path = comparison.build_result_path(
            "global_model",
            dataset_name,
            args.algorithm,
            args.goal,
            args.times,
            f"fu_{level}",
            "pt",
        )
        final_result_path = comparison.build_result_path(
            "forget_effect",
            dataset_name,
            args.algorithm,
            args.goal,
            args.times,
            config["result_tag"],
            "json",
        )
        overrides = {
            "dataset": dataset_name,
            "result_tag": config["result_tag"],
            "forget_strategy": "gradient_reversal",
            "load_saved_model": True,
            "saved_model_path": str(base_model_path),
            "fu_recovery_target_penalty": config["fu_recovery_target_penalty"],
            "fu_recovery_lr_scale": config.get("fu_recovery_lr_scale", 1.0),
        }
        if "recovery_rounds" in config:
            overrides["recovery_rounds"] = config["recovery_rounds"]
        commands.append((level, build_main_command(args, overrides), base_model_path, final_result_path))
    return commands


def dir_asset_commands(args):
    base_summary = HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_heterogeneity_summary.csv"
    final_summary = HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_heterogeneity_summary_final.csv"
    delta_csv = HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_fu_final_delta.csv"
    figure_png = HETERO_ROOT / "heterogeneity_paper_figure_final.png"
    figure_pdf = HETERO_ROOT / "heterogeneity_paper_figure_final.pdf"
    table_tex = HETERO_ROOT / "heterogeneity_paper_table_final.tex"

    build_summary_cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/build_fu_adaptive_summary.py",
        "--base-summary", str(base_summary),
        "--output-summary", str(final_summary),
        "--delta-output", str(delta_csv),
        "--algorithm", args.algorithm,
        "--goal", args.goal,
        "--times", str(args.times),
        "--target-client-id", str(args.target_client_id),
        "--fu-tag", f"mild:{DIR_FU_FINAL_TAGS['mild']['result_tag']}",
        "--fu-tag", f"moderate:{DIR_FU_FINAL_TAGS['moderate']['result_tag']}",
        "--fu-tag", f"severe:{DIR_FU_FINAL_TAGS['severe']['result_tag']}",
    ]
    asset_cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_heterogeneity_paper_assets.py",
        "--summary", str(final_summary),
        "--figure-png", str(figure_png),
        "--figure-pdf", str(figure_pdf),
        "--table", str(table_tex),
        "--title", "Comparison Under Mild, Moderate, and Severe Dirichlet Heterogeneity",
        "--caption",
        "Results under different Dirichlet heterogeneity levels on CIFAR-10. Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, while higher average accuracy indicates better utility retention.",
    ]
    return [
        ("dir-final-summary", build_summary_cmd),
        ("dir-final-assets", asset_cmd),
    ]


def pat_main_suite(args):
    return comparison_command(
        args,
        [
            "--experiment-name", "pathological_label_skew",
            "--dataset-prefix", "pat",
            "--partition", "pat",
            "--levels", "mild", "moderate", "severe",
            "--methods", "fu", "fedau", "fedcsa", "fedosd", "retrain",
            "--class-per-client-mild", "5",
            "--class-per-client-moderate", "3",
            "--class-per-client-severe", "2",
            "--alpha-mild", "1.0",
            "--alpha-moderate", "1.0",
            "--alpha-severe", "1.0",
        ],
    )


def pat_asset_commands(args):
    pat_summary = HETERO_ROOT / f"{args.dataset_base}_{args.algorithm}_pathological_label_skew_summary.csv"
    figure_png = HETERO_ROOT / "pat_paper_figure.png"
    figure_pdf = HETERO_ROOT / "pat_paper_figure.pdf"
    table_tex = HETERO_ROOT / "pat_paper_table.tex"
    asset_cmd = [
        args.python,
        "-u",
        "system/heterogeneity_runs/generate_heterogeneity_paper_assets.py",
        "--summary", str(pat_summary),
        "--figure-png", str(figure_png),
        "--figure-pdf", str(figure_pdf),
        "--table", str(table_tex),
        "--title", "Comparison Under Mild, Moderate, and Severe Pathological Label Skew",
        "--caption",
        "Results under pathological label-skew on CIFAR-10, where mild/moderate/severe correspond to 5/3/2 classes per client. Lower target accuracy, MIA AUC, and backdoor accuracy indicate stronger forgetting/privacy protection, while higher average accuracy indicates better utility retention.",
    ]
    return [("pat-assets", asset_cmd)]


def planned_steps(args, suites):
    steps = []
    for suite in suites:
        if suite == "dir-main":
            steps.append(("dir-main", dir_main_suite(args)))
        elif suite == "dir-fu-final":
            for level, cmd, base_model_path, result_path in dir_fu_final_commands(args):
                steps.append((f"dir-fu-final-{level}", cmd, base_model_path, result_path))
        elif suite == "dir-assets":
            steps.extend(dir_asset_commands(args))
        elif suite == "pat-main":
            steps.append(("pat-main", pat_main_suite(args)))
        elif suite == "pat-assets":
            steps.extend(pat_asset_commands(args))
    return steps


def print_plan(args, steps):
    print("Paper experiment pipeline:")
    for entry in steps:
        if len(entry) >= 3:
            name, cmd, dependency = entry[:3]
            print(f"[{name}] requires {dependency}")
        else:
            name, cmd = entry
        print(f"[{name}] {shlex.join(wrap_env(args, cmd))}")


def main():
    args = parse_args()
    suites = normalized_suites(args.suites)
    steps = planned_steps(args, suites)
    print_plan(args, steps)

    if not args.run:
        return

    failures = []
    pipeline_log_root = HETERO_ROOT / "paper_pipeline_logs"
    pipeline_log_root.mkdir(parents=True, exist_ok=True)

    for entry in steps:
        dependency = None
        existing_result = None
        if len(entry) >= 3:
            name, cmd, dependency = entry[:3]
            if len(entry) >= 4:
                existing_result = entry[3]
        else:
            name, cmd = entry

        if dependency is not None and not Path(dependency).exists():
            message = f"Missing dependency for {name}: {dependency}"
            if args.continue_on_error:
                print(f"[WARN] {message}")
                failures.append((name, message))
                continue
            raise FileNotFoundError(message)

        if args.skip_existing and existing_result is not None and Path(existing_result).exists():
            print(f"\nSkipping {name} because {existing_result.name} already exists.")
            continue

        log_path = pipeline_log_root / f"{name}.log"
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
