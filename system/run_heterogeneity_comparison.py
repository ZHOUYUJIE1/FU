import argparse
import csv
import importlib
import inspect
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "exp_configs"
LOG_ROOT = PROJECT_ROOT / "system" / "heterogeneity_runs"
ZYJ_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from dataset.utils import dataset_utils


LEVEL_ORDER = ["mild", "moderate", "severe"]
METHOD_ORDER = ["fu", "fedau", "fedcsa", "fedosd", "retrain"]

METHOD_SPECS = {
    "fu": {
        "display_name": "FU",
        "forget_strategy": "gradient_reversal",
        "retrain_only": False,
        "load_saved_model": False,
        "requires_shared_base": False,
    },
    "fedau": {
        "display_name": "FedAU",
        "forget_strategy": "fedau",
        "retrain_only": False,
        "load_saved_model": False,
        "requires_shared_base": False,
    },
    "fedcsa": {
        "display_name": "FedCSA",
        "forget_strategy": "fedcsa",
        "retrain_only": False,
        "load_saved_model": True,
        "requires_shared_base": True,
    },
    "fedosd": {
        "display_name": "FedOSD",
        "forget_strategy": "fedosd",
        "retrain_only": False,
        "load_saved_model": True,
        "requires_shared_base": True,
    },
    "retrain": {
        "display_name": "Retrain",
        "forget_strategy": "gradient_reversal",
        "retrain_only": True,
        "load_saved_model": False,
        "requires_shared_base": False,
    },
}

MAIN_DEFAULTS = {
    "goal": "test",
    "device": "cuda",
    "device_id": "0",
    "dataset": "Cifar10",
    "num_classes": 10,
    "model": "ResNet18",
    "batch_size": 32,
    "local_learning_rate": 0.005,
    "global_rounds": 100,
    "local_epochs": 1,
    "algorithm": "FedAvg",
    "join_ratio": 1.0,
    "num_clients": 10,
    "times": 1,
    "eval_gap": 1,
    "random_seed": 42,
    "result_tag": "",
    "forget_strategy": "sifu",
    "recovery_rounds": 5,
    "max_batches": 8,
    "num_processes": 4,
    "fedau_alpha": 0.9,
    "fedau_client_gamma": 0.5,
    "fedau_skip_recovery": True,
    "fedosd_lr": 0.0004,
    "fedosd_unlearn_rounds": 20,
    "fedosd_recovery_rounds": 1,
    "fedosd_recovery_lr": 1e-6,
    "fedosd_force_target_online": True,
    "fedosd_max_online_clients": 0,
    "fedu_lr": 0.003,
    "fedu_eps": 0.005,
    "fedu_alpha": 0.02,
    "fedu_erased_ratio": 0.35,
    "fedu_erased_max_samples": 1200,
    "fedu_forget_rounds": 12,
    "fedu_local_steps": 6,
    "fedu_hutchinson_samples": 8,
    "fedu_utility_scale": 0.35,
    "fu_retain_calibration_rounds": 2,
    "fu_retain_calibration_lr": 4e-4,
    "fu_retain_calibration_batches": 3,
    "fu_mask_retain_scale": 0.12,
    "fu_similarity_boost": 1.2,
    "fu_select_best_recovery": True,
    "fu_recovery_target_penalty": 0.5,
    "fu_recovery_lr_scale": 1.0,
    "protection_level": "weak",
    "target_client_id": 5,
    "load_saved_model": False,
    "saved_model_path": "/home/siguangchen/zyj/FUcopy/system/results/exp_configs/global_model_Cifar10_FedAvg_test_1_fedau.pt",
    "retrain_only": False,
}

MAIN_ARG_FLAGS = {
    "goal": "--goal",
    "device": "--device",
    "device_id": "--device_id",
    "dataset": "--dataset",
    "num_classes": "--num_classes",
    "model": "--model",
    "batch_size": "--batch_size",
    "local_learning_rate": "--local_learning_rate",
    "global_rounds": "--global_rounds",
    "local_epochs": "--local_epochs",
    "algorithm": "--algorithm",
    "join_ratio": "--join_ratio",
    "num_clients": "--num_clients",
    "times": "--times",
    "eval_gap": "--eval_gap",
    "random_seed": "--random_seed",
    "result_tag": "--result_tag",
    "forget_strategy": "--forget_strategy",
    "recovery_rounds": "--recovery_rounds",
    "max_batches": "--max_batches",
    "num_processes": "--num_processes",
    "fedau_alpha": "--fedau_alpha",
    "fedau_client_gamma": "--fedau_client_gamma",
    "fedau_skip_recovery": "--fedau_skip_recovery",
    "fedosd_lr": "--fedosd_lr",
    "fedosd_unlearn_rounds": "--fedosd_unlearn_rounds",
    "fedosd_recovery_rounds": "--fedosd_recovery_rounds",
    "fedosd_recovery_lr": "--fedosd_recovery_lr",
    "fedosd_force_target_online": "--fedosd_force_target_online",
    "fedosd_max_online_clients": "--fedosd_max_online_clients",
    "fedu_lr": "--fedu_lr",
    "fedu_eps": "--fedu_eps",
    "fedu_alpha": "--fedu_alpha",
    "fedu_erased_ratio": "--fedu_erased_ratio",
    "fedu_erased_max_samples": "--fedu_erased_max_samples",
    "fedu_forget_rounds": "--fedu_forget_rounds",
    "fedu_local_steps": "--fedu_local_steps",
    "fedu_hutchinson_samples": "--fedu_hutchinson_samples",
    "fedu_utility_scale": "--fedu_utility_scale",
    "fu_retain_calibration_rounds": "--fu_retain_calibration_rounds",
    "fu_retain_calibration_lr": "--fu_retain_calibration_lr",
    "fu_retain_calibration_batches": "--fu_retain_calibration_batches",
    "fu_mask_retain_scale": "--fu_mask_retain_scale",
    "fu_similarity_boost": "--fu_similarity_boost",
    "fu_select_best_recovery": "--fu_select_best_recovery",
    "fu_recovery_target_penalty": "--fu_recovery_target_penalty",
    "fu_recovery_lr_scale": "--fu_recovery_lr_scale",
    "protection_level": "--protection_level",
    "target_client_id": "--target_client_id",
    "load_saved_model": "--load_saved_model",
    "saved_model_path": "--saved_model_path",
    "retrain_only": "--retrain_only",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and run FU/FedAU/FedCSA/FedOSD/Retrain comparisons under mild/moderate/severe heterogeneity."
    )
    parser.add_argument("--dataset-base", type=str, default="Cifar10")
    parser.add_argument("--dataset-prefix", type=str, default="hetero",
                        help="Dataset name infix used when generating per-level dataset variants, e.g. Cifar10_hetero_mild.")
    parser.add_argument("--experiment-name", type=str, default="heterogeneity",
                        help="Output stem used for command scripts and summary CSVs.")
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
    parser.add_argument("--keep-ld-library-path", action="store_true",
                        help="Keep the current LD_LIBRARY_PATH when launching experiment commands.")
    parser.add_argument("--levels", nargs="+", choices=LEVEL_ORDER, default=LEVEL_ORDER)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument("--run", action="store_true",
                        help="Actually execute the generated commands. Without this flag the script only prepares datasets, writes commands, and collects summaries from existing results.")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip dataset preparation and only generate/run commands against existing datasets.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a method-level run if its primary result file already exists.")
    parser.add_argument("--resume-incomplete", action="store_true",
                        help="If an FU shared checkpoint already exists but its forgetting result is missing, resume from that checkpoint instead of retraining.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep running remaining method-level experiments even if one command fails.")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Delete existing train/test/config files before regenerating each heterogeneity dataset.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only collect existing results into a CSV summary.")

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

    parser.add_argument("--partition", type=str, default="dir", choices=["dir", "pat", "exdir"])
    parser.add_argument("--balance", dest="balance", action="store_true")
    parser.add_argument("--unbalance", dest="balance", action="store_false")
    parser.set_defaults(balance=True)
    parser.add_argument("--alpha-mild", type=float, default=1.0)
    parser.add_argument("--alpha-moderate", type=float, default=0.3)
    parser.add_argument("--alpha-severe", type=float, default=0.1)
    parser.add_argument("--class-per-client-mild", type=int, default=5,
                        help="Used when partition=pat/exdir. Larger means milder label skew.")
    parser.add_argument("--class-per-client-moderate", type=int, default=3)
    parser.add_argument("--class-per-client-severe", type=int, default=2)

    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--recovery-rounds", type=int, default=5)

    parser.add_argument("--fedau-alpha", type=float, default=0.9)
    parser.add_argument("--fedau-client-gamma", type=float, default=0.5)
    parser.add_argument("--fedau-skip-recovery", type=str, default="true")

    parser.add_argument("--fedosd-lr", type=float, default=0.0004)
    parser.add_argument("--fedosd-unlearn-rounds", type=int, default=20)
    parser.add_argument("--fedosd-recovery-rounds", type=int, default=1)
    parser.add_argument("--fedosd-recovery-lr", type=float, default=1e-6)
    parser.add_argument("--fedosd-force-target-online", type=str, default="true")
    parser.add_argument("--fedosd-max-online-clients", type=int, default=0)

    parser.add_argument("--fedu-lr", type=float, default=0.003)
    parser.add_argument("--fedu-eps", type=float, default=0.005)
    parser.add_argument("--fedu-alpha", type=float, default=0.02)
    parser.add_argument("--fedu-erased-ratio", type=float, default=0.35)
    parser.add_argument("--fedu-erased-max-samples", type=int, default=1200)
    parser.add_argument("--fedu-forget-rounds", type=int, default=12)
    parser.add_argument("--fedu-local-steps", type=int, default=6)
    parser.add_argument("--fedu-hutchinson-samples", type=int, default=8)
    parser.add_argument("--fedu-utility-scale", type=float, default=0.35)

    parser.add_argument("--fu-retain-calibration-rounds", type=int, default=2)
    parser.add_argument("--fu-retain-calibration-lr", type=float, default=4e-4)
    parser.add_argument("--fu-retain-calibration-batches", type=int, default=3)
    parser.add_argument("--fu-mask-retain-scale", type=float, default=0.12)
    parser.add_argument("--fu-similarity-boost", type=float, default=1.2)
    parser.add_argument("--fu-select-best-recovery", type=str, default="true")
    parser.add_argument("--fu-recovery-target-penalty", type=float, default=0.5)
    parser.add_argument("--fu-recovery-lr-scale", type=float, default=1.0)
    parser.add_argument("--protection-level", type=str, default="weak", choices=["strong", "moderate", "weak"])

    return parser.parse_args()


def normalize_result_tag(result_tag):
    return result_tag.replace(os.sep, "_").replace(" ", "_").strip("_")


def build_result_path(prefix, dataset_name, algorithm, goal, times, result_tag, extension):
    safe_tag = normalize_result_tag(result_tag)
    suffix = f"_{safe_tag}" if safe_tag else ""
    filename = f"{prefix}_{dataset_name}_{algorithm}_{goal}_{times}{suffix}.{extension}"
    return RESULT_ROOT / filename


def dataset_name_for_level(args, level):
    return f"{args.dataset_base}_{args.dataset_prefix}_{level}"


def result_tag_for(level, method):
    return f"{method}_{level}"


def level_alpha_map(args):
    return {
        "mild": float(args.alpha_mild),
        "moderate": float(args.alpha_moderate),
        "severe": float(args.alpha_severe),
    }


def level_class_per_client_map(args):
    return {
        "mild": int(args.class_per_client_mild),
        "moderate": int(args.class_per_client_moderate),
        "severe": int(args.class_per_client_severe),
    }


def output_stem(args):
    return f"{args.dataset_base}_{args.algorithm}_{args.experiment_name}"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_generator(dataset_base):
    module = importlib.import_module(f"dataset.generate_{dataset_base}")
    generate_dataset = getattr(module, "generate_dataset", None)
    if generate_dataset is None:
        raise AttributeError(f"dataset.generate_{dataset_base} does not define generate_dataset")

    signature = inspect.signature(generate_dataset)
    if len(signature.parameters) < 5:
        raise TypeError(
            f"dataset.generate_{dataset_base}.generate_dataset does not support (dir_path, num_clients, niid, balance, partition)"
        )
    return generate_dataset


def apply_partition_config(alpha):
    dataset_utils.set_partition_config(alpha_value=alpha)
    try:
        legacy_dataset_utils = importlib.import_module("utils.dataset_utils")
    except ModuleNotFoundError:
        return

    if hasattr(legacy_dataset_utils, "set_partition_config"):
        legacy_dataset_utils.set_partition_config(alpha_value=alpha)
    else:
        legacy_dataset_utils.alpha = float(alpha)


def level_metadata(args, level, alpha_map, class_per_client_map):
    class_per_client = None
    alpha = None
    if args.partition == "dir":
        alpha = float(alpha_map[level])
    elif args.partition in {"pat", "exdir"}:
        class_per_client = int(class_per_client_map[level])
        alpha = float(alpha_map[level])

    return {
        "level": level,
        "dataset_name": dataset_name_for_level(args, level),
        "partition": args.partition,
        "alpha": alpha,
        "num_clients": int(args.num_clients),
        "class_per_client": class_per_client,
    }


def ensure_rawdata_link(dataset_base, dataset_dir):
    source_rawdata = DATASET_ROOT / dataset_base / "rawdata"
    target_rawdata = dataset_dir / "rawdata"
    if target_rawdata.exists() or not source_rawdata.exists():
        return

    try:
        target_rawdata.symlink_to(source_rawdata, target_is_directory=True)
    except OSError:
        shutil.copytree(source_rawdata, target_rawdata)


def clear_partition_outputs(dataset_dir):
    for name in ("train", "test"):
        path = dataset_dir / name
        if path.exists():
            shutil.rmtree(path)
    config_path = dataset_dir / "config.json"
    if config_path.exists():
        config_path.unlink()


def prepare_dataset_variant(args, variant_meta):
    dataset_name = variant_meta["dataset_name"]
    dataset_dir = DATASET_ROOT / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.force_regenerate:
        clear_partition_outputs(dataset_dir)

    ensure_rawdata_link(args.dataset_base, dataset_dir)
    generate_dataset = load_generator(args.dataset_base)
    apply_partition_config(variant_meta["alpha"])
    set_global_seed(args.seed)
    call_kwargs = {}
    if "class_per_client" in inspect.signature(generate_dataset).parameters:
        call_kwargs["class_per_client"] = variant_meta["class_per_client"]
    generate_dataset(
        str(dataset_dir) + os.sep,
        variant_meta["num_clients"],
        True,
        bool(args.balance),
        args.partition,
        **call_kwargs,
    )
    return dataset_name


def primary_result_path(args, dataset_name, level, method):
    result_tag = result_tag_for(level, method)
    if method == "retrain":
        return build_result_path(
            "eval_retrain_only",
            dataset_name,
            args.algorithm,
            args.goal,
            args.times,
            result_tag,
            "json",
        )
    return build_result_path(
        "forget_effect",
        dataset_name,
        args.algorithm,
        args.goal,
        args.times,
        result_tag,
        "json",
    )


def shared_base_model_path(args, dataset_name, level):
    return build_result_path(
        "global_model",
        dataset_name,
        args.algorithm,
        args.goal,
        args.times,
        result_tag_for(level, "fu"),
        "pt",
    )


def build_main_command(args, dataset_name, level, method):
    result_tag = result_tag_for(level, method)
    spec = METHOD_SPECS[method]
    base_model_path = shared_base_model_path(args, dataset_name, level)
    primary_path = primary_result_path(args, dataset_name, level, method)

    desired_args = {
        "goal": args.goal,
        "device": args.device,
        "device_id": args.device_id,
        "dataset": dataset_name,
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
        "result_tag": result_tag,
        "forget_strategy": spec["forget_strategy"],
        "recovery_rounds": args.recovery_rounds,
        "max_batches": args.max_batches,
        "num_processes": args.num_processes,
        "fedau_alpha": args.fedau_alpha,
        "fedau_client_gamma": args.fedau_client_gamma,
        "fedau_skip_recovery": args.fedau_skip_recovery.lower() == "true",
        "fedosd_lr": args.fedosd_lr,
        "fedosd_unlearn_rounds": args.fedosd_unlearn_rounds,
        "fedosd_recovery_rounds": args.fedosd_recovery_rounds,
        "fedosd_recovery_lr": args.fedosd_recovery_lr,
        "fedosd_force_target_online": args.fedosd_force_target_online.lower() == "true",
        "fedosd_max_online_clients": args.fedosd_max_online_clients,
        "fedu_lr": args.fedu_lr,
        "fedu_eps": args.fedu_eps,
        "fedu_alpha": args.fedu_alpha,
        "fedu_erased_ratio": args.fedu_erased_ratio,
        "fedu_erased_max_samples": args.fedu_erased_max_samples,
        "fedu_forget_rounds": args.fedu_forget_rounds,
        "fedu_local_steps": args.fedu_local_steps,
        "fedu_hutchinson_samples": args.fedu_hutchinson_samples,
        "fedu_utility_scale": args.fedu_utility_scale,
        "fu_retain_calibration_rounds": args.fu_retain_calibration_rounds,
        "fu_retain_calibration_lr": args.fu_retain_calibration_lr,
        "fu_retain_calibration_batches": args.fu_retain_calibration_batches,
        "fu_mask_retain_scale": args.fu_mask_retain_scale,
        "fu_similarity_boost": args.fu_similarity_boost,
        "fu_select_best_recovery": args.fu_select_best_recovery.lower() == "true",
        "fu_recovery_target_penalty": args.fu_recovery_target_penalty,
        "fu_recovery_lr_scale": args.fu_recovery_lr_scale,
        "protection_level": args.protection_level,
        "load_saved_model": False,
        "retrain_only": spec["retrain_only"],
    }

    if spec["load_saved_model"]:
        desired_args["load_saved_model"] = True
        desired_args["saved_model_path"] = str(base_model_path)

    if (
        args.resume_incomplete
        and method == "fu"
        and not primary_path.exists()
        and base_model_path.exists()
    ):
        desired_args["load_saved_model"] = True
        desired_args["saved_model_path"] = str(base_model_path)

    cmd = [args.python, "-u", "system/main.py"]

    for arg_name, flag in MAIN_ARG_FLAGS.items():
        if arg_name not in desired_args:
            continue

        value = desired_args[arg_name]
        default_value = MAIN_DEFAULTS[arg_name]
        if value == default_value:
            continue

        if isinstance(value, bool):
            value = str(value).lower()
        cmd.extend([flag, str(value)])

    return cmd


def ordered_levels(levels):
    return sorted(levels, key=LEVEL_ORDER.index)


def ordered_methods(methods):
    return sorted(methods, key=METHOD_ORDER.index)


def wrap_command_for_shell(args, cmd):
    if getattr(args, "keep_ld_library_path", False):
        return list(cmd)
    return ["env", "-u", "LD_LIBRARY_PATH", *cmd]


def write_command_script(args, commands):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    script_path = LOG_ROOT / f"{output_stem(args)}_commands.sh"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f"cd {shlex.quote(str(PROJECT_ROOT))}\n\n")
        for _, _, _, cmd in commands:
            f.write(shlex.join(wrap_command_for_shell(args, cmd)) + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def print_command_plan(args, commands, variant_meta_by_level):
    print("Heterogeneity plan:")
    for level in ordered_levels(args.levels):
        meta = variant_meta_by_level[level]
        parts = [f"partition={meta['partition']}", f"num_clients={meta['num_clients']}"]
        if meta["alpha"] is not None:
            parts.append(f"alpha={meta['alpha']}")
        if meta["class_per_client"] is not None:
            parts.append(f"class_per_client={meta['class_per_client']}")
        print(f"  {level}: " + ", ".join(parts))
        for method in ordered_methods(args.methods):
            dataset_name = meta["dataset_name"]
            result_tag = result_tag_for(level, method)
            print(f"    {METHOD_SPECS[method]['display_name']}: dataset={dataset_name}, result_tag={result_tag}")

    print("\nCommands:")
    for level, method, _, cmd in commands:
        print(f"[{level}][{method}] {shlex.join(wrap_command_for_shell(args, cmd))}")


def run_command(args, cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if not getattr(args, "keep_ld_library_path", False):
        run_env.pop("LD_LIBRARY_PATH", None)
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=run_env,
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_attack_file(path):
    if not path.exists():
        return {}

    parsed = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            try:
                parsed[normalized_key] = float(value.strip())
            except ValueError:
                parsed[normalized_key] = value.strip()
    return parsed


def safe_client_metric(eval_data, target_client_id, retain=False):
    if not eval_data:
        return None

    client_accs = eval_data.get("client_accuracies") or []
    if not client_accs:
        return None

    if retain:
        retained = [acc for idx, acc in enumerate(client_accs) if idx != target_client_id]
        if not retained:
            return None
        return float(np.mean(retained))

    if 0 <= target_client_id < len(client_accs):
        return float(client_accs[target_client_id])
    return None


def collect_summary(args, variant_meta_by_level):
    rows = []
    for level in ordered_levels(args.levels):
        meta = variant_meta_by_level[level]
        dataset_name = meta["dataset_name"]
        for method in ordered_methods(args.methods):
            result_tag = result_tag_for(level, method)
            row = {
                "level": level,
                "partition": meta["partition"],
                "alpha": meta["alpha"],
                "num_clients": meta["num_clients"],
                "class_per_client": meta["class_per_client"],
                "dataset": dataset_name,
                "method": method,
                "display_name": METHOD_SPECS[method]["display_name"],
                "result_tag": result_tag,
            }

            if method == "retrain":
                eval_path = build_result_path(
                    "eval_retrain_only",
                    dataset_name,
                    args.algorithm,
                    args.goal,
                    args.times,
                    result_tag,
                    "json",
                )
                attack_path = build_result_path(
                    "attack_retrain_only",
                    dataset_name,
                    args.algorithm,
                    args.goal,
                    args.times,
                    result_tag,
                    "txt",
                )
                eval_data = read_json(eval_path)
                attack_data = parse_attack_file(attack_path)

                row["status"] = "complete" if eval_data else "missing"
                if eval_data:
                    row["final_avg_acc"] = eval_data.get("average_accuracy")
                    row["final_avg_auc"] = eval_data.get("average_auc")
                    row["final_target_acc"] = safe_client_metric(eval_data, args.target_client_id)
                    row["final_retain_avg_acc"] = safe_client_metric(eval_data, args.target_client_id, retain=True)
                row["final_mia_auc"] = attack_data.get("mia_auc")
                row["final_backdoor_acc"] = attack_data.get("backdoor_acc")
                rows.append(row)
                continue

            pre_eval_path = build_result_path(
                "eval_pre_forget",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "json",
            )
            post_eval_path = build_result_path(
                "eval_post_forget",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "json",
            )
            recovery_eval_path = build_result_path(
                "eval_post_recovery",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "json",
            )
            forget_summary_path = build_result_path(
                "forget_effect",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "json",
            )
            attack_pre_path = build_result_path(
                "attack_pre_forget",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "txt",
            )
            attack_post_path = build_result_path(
                "attack_post_forget",
                dataset_name,
                args.algorithm,
                args.goal,
                args.times,
                result_tag,
                "txt",
            )

            pre_eval = read_json(pre_eval_path)
            post_eval = read_json(post_eval_path)
            recovery_eval = read_json(recovery_eval_path)
            forget_summary = read_json(forget_summary_path)
            attack_pre = parse_attack_file(attack_pre_path)
            attack_post = parse_attack_file(attack_post_path)

            row["status"] = "complete" if pre_eval and post_eval and forget_summary else "missing"
            row["pre_avg_acc"] = pre_eval.get("average_accuracy") if pre_eval else None
            row["post_avg_acc"] = post_eval.get("average_accuracy") if post_eval else None
            row["recovery_avg_acc"] = recovery_eval.get("average_accuracy") if recovery_eval else None
            row["final_avg_acc"] = (
                recovery_eval.get("average_accuracy")
                if recovery_eval
                else post_eval.get("average_accuracy")
                if post_eval
                else None
            )
            row["pre_target_acc"] = safe_client_metric(pre_eval, args.target_client_id)
            row["post_target_acc"] = safe_client_metric(post_eval, args.target_client_id)
            row["final_target_acc"] = (
                safe_client_metric(recovery_eval, args.target_client_id)
                if recovery_eval
                else safe_client_metric(post_eval, args.target_client_id)
            )
            row["pre_retain_avg_acc"] = safe_client_metric(pre_eval, args.target_client_id, retain=True)
            row["post_retain_avg_acc"] = safe_client_metric(post_eval, args.target_client_id, retain=True)
            row["final_retain_avg_acc"] = (
                safe_client_metric(recovery_eval, args.target_client_id, retain=True)
                if recovery_eval
                else safe_client_metric(post_eval, args.target_client_id, retain=True)
            )

            if forget_summary:
                row["target_acc_change"] = forget_summary.get("target_abs_change")
                row["target_acc_change_ratio"] = forget_summary.get("target_rel_change")
                row["others_acc_change_avg"] = forget_summary.get("others_abs_change_avg")
                row["others_acc_change_ratio"] = forget_summary.get("others_rel_change_avg")
                row["recovery_effect"] = forget_summary.get("recovery_effect")
                row["recovery_stage"] = forget_summary.get("recovery_stage")

            row["mia_pre_auc"] = attack_pre.get("mia_auc")
            row["mia_post_auc"] = attack_post.get("mia_auc")
            row["mia_auc_drop"] = (
                attack_pre.get("mia_auc") - attack_post.get("mia_auc")
                if attack_pre.get("mia_auc") is not None and attack_post.get("mia_auc") is not None
                else None
            )
            row["backdoor_pre_acc"] = attack_pre.get("backdoor_acc")
            row["backdoor_post_acc"] = attack_post.get("backdoor_acc")

            rows.append(row)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_ROOT / f"{output_stem(args)}_summary.csv"
    if rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def main():
    args = parse_args()
    alpha_map = level_alpha_map(args)
    class_per_client_map = level_class_per_client_map(args)
    levels = ordered_levels(args.levels)
    methods = ordered_methods(args.methods)
    variant_meta_by_level = {
        level: level_metadata(args, level, alpha_map, class_per_client_map)
        for level in levels
    }

    if args.summary_only:
        summary_path = collect_summary(args, variant_meta_by_level)
        print(f"Summary written to: {summary_path}")
        return

    if not args.skip_prepare:
        for level in levels:
            meta = variant_meta_by_level[level]
            dataset_name = prepare_dataset_variant(args, meta)
            descriptor = [f"partition={meta['partition']}"]
            if meta["alpha"] is not None:
                descriptor.append(f"alpha={meta['alpha']}")
            if meta["class_per_client"] is not None:
                descriptor.append(f"class_per_client={meta['class_per_client']}")
            descriptor.append(f"num_clients={meta['num_clients']}")
            print(f"Prepared dataset: {dataset_name} ({', '.join(descriptor)})")

    commands = []
    for level in levels:
        dataset_name = variant_meta_by_level[level]["dataset_name"]
        for method in methods:
            cmd = build_main_command(args, dataset_name, level, method)
            log_path = LOG_ROOT / dataset_name / f"{method}.log"
            commands.append((level, method, log_path, cmd))

    script_path = write_command_script(args, commands)
    print_command_plan(args, commands, variant_meta_by_level)
    print(f"\nCommand script written to: {script_path}")

    if not args.run:
        summary_path = collect_summary(args, variant_meta_by_level)
        print(f"Summary written to: {summary_path}")
        return

    failures = []
    for level, method, log_path, cmd in commands:
        dataset_name = variant_meta_by_level[level]["dataset_name"]
        primary_path = primary_result_path(args, dataset_name, level, method)
        if args.skip_existing and primary_path.exists():
            print(f"\nSkipping [{level}][{method}] because {primary_path.name} already exists.")
            continue

        if METHOD_SPECS[method]["requires_shared_base"]:
            base_model_path = shared_base_model_path(args, dataset_name, level)
            if not base_model_path.exists():
                raise FileNotFoundError(
                    f"Shared base checkpoint not found for [{level}][{method}]: {base_model_path}. "
                    f"Run FU first for this level, or keep the existing FU checkpoint."
                )

        print(f"\n===== Running [{level}] {METHOD_SPECS[method]['display_name']} =====")
        print(f"Log file: {log_path}")
        try:
            run_command(args, cmd, log_path)
        except subprocess.CalledProcessError as exc:
            failures.append((level, method, exc.returncode, cmd))
            print(f"\n[ERROR] [{level}][{method}] failed with return code {exc.returncode}.")
            if not args.continue_on_error:
                raise

    summary_path = collect_summary(args, variant_meta_by_level)
    print(f"\nSummary written to: {summary_path}")
    if failures:
        print("\nFailed runs:")
        for level, method, return_code, cmd in failures:
            print(f"  [{level}][{method}] return_code={return_code}: {shlex.join(wrap_command_for_shell(args, cmd))}")


if __name__ == "__main__":
    main()
