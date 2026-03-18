import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "exp_configs"
SUMMARY_ROOT = PROJECT_ROOT / "system" / "heterogeneity_runs"
DEFAULT_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"
PYTHON_BIN = DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON) else sys.executable

ALGORITHM = "FedAvg"
GOAL = "test"
TIMES = 1
TARGET_CLIENT_ID = 5

LEVELS = ("mild", "moderate", "severe")
METHODS = ("fu", "fedau", "fedcsa", "fedosd", "retrain")

METHOD_FORGET_STRATEGY = {
    "fu": "gradient_reversal",
    "fedau": "fedau",
    "fedcsa": "fedcsa",
    "fedosd": "fedosd",
    "retrain": "gradient_reversal",
}

MAIN_FLAG_MAP = {
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
    "mia_pre_model_source": "--mia_pre_model_source",
    "save_target_local_model_last": "--save_target_local_model_last",
    "random_seed": "--random_seed",
    "result_tag": "--result_tag",
    "forget_strategy": "--forget_strategy",
    "recovery_rounds": "--recovery_rounds",
    "max_batches": "--max_batches",
    "num_processes": "--num_processes",
    "fedau_alpha": "--fedau_alpha",
    "fedau_mode": "--fedau_mode",
    "fedau_client_gamma": "--fedau_client_gamma",
    "fedau_skip_recovery": "--fedau_skip_recovery",
    "fedosd_lr": "--fedosd_lr",
    "fedosd_unlearn_rounds": "--fedosd_unlearn_rounds",
    "fedosd_recovery_rounds": "--fedosd_recovery_rounds",
    "fedosd_recovery_lr": "--fedosd_recovery_lr",
    "fedosd_force_target_online": "--fedosd_force_target_online",
    "fedosd_max_online_clients": "--fedosd_max_online_clients",
    "fu_retain_calibration_rounds": "--fu_retain_calibration_rounds",
    "fu_retain_calibration_lr": "--fu_retain_calibration_lr",
    "fu_retain_calibration_batches": "--fu_retain_calibration_batches",
    "fu_mask_retain_scale": "--fu_mask_retain_scale",
    "fu_similarity_boost": "--fu_similarity_boost",
    "fu_select_best_recovery": "--fu_select_best_recovery",
    "fu_recovery_target_penalty": "--fu_recovery_target_penalty",
    "fu_recovery_lr_scale": "--fu_recovery_lr_scale",
    "fu_bn_recalibration": "--fu_bn_recalibration",
    "fu_bn_recalibration_batches": "--fu_bn_recalibration_batches",
    "protection_level": "--protection_level",
    "target_client_id": "--target_client_id",
    "load_saved_model": "--load_saved_model",
    "saved_model_path": "--saved_model_path",
    "retrain_only": "--retrain_only",
    "mia_mode": "--mia_mode",
    "mia_client_oriented": "--mia_client_oriented",
    "mia_reference_mode": "--mia_reference_mode",
    "enable_backdoor_attack": "--enable_backdoor_attack",
    "backdoor_client_oriented": "--backdoor_client_oriented",
}


def dataset_name(level):
    return f"Cifar10_hetero_clean_s42_{level}"


def original_result_tag(method, level):
    return f"{method}_{level}"


def eval_result_tag(method, level, suffix):
    return f"{method}_{level}_{suffix}"


def artifact_path(prefix, dataset, result_tag, extension):
    suffix = f"_{result_tag}" if result_tag else ""
    return RESULT_ROOT / f"{prefix}_{dataset}_{ALGORITHM}_{GOAL}_{TIMES}{suffix}.{extension}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_model_path(level):
    return artifact_path("global_model", dataset_name(level), original_result_tag("fu", level), "pt")


def build_fedau_global_model_path(level):
    return artifact_path("global_model", dataset_name(level), original_result_tag("fedau", level), "pt")


def build_forget_model_path(method, level):
    return artifact_path("forget_model", dataset_name(level), original_result_tag(method, level), "pt")


def build_retrain_model_path(level):
    return artifact_path("retrain_model", dataset_name(level), original_result_tag("retrain", level), "pt")


def build_config_path(method, level):
    return artifact_path("exp_config", dataset_name(level), original_result_tag(method, level), "json")


def _stringify(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_main_command(config_path, result_tag, overrides):
    cfg = load_json(config_path)
    merged = dict(cfg)
    merged.setdefault("goal", GOAL)
    merged.setdefault("dataset", cfg["dataset"])
    merged.setdefault("algorithm", ALGORITHM)
    merged.setdefault("times", TIMES)
    merged.setdefault("target_client_id", TARGET_CLIENT_ID)
    merged.setdefault("random_seed", int(cfg.get("seed", 42)))
    merged.setdefault("mia_mode", "shadow_lr")
    merged.setdefault("mia_client_oriented", True)
    merged.setdefault("mia_reference_mode", "target_train_vs_other_train")
    merged.setdefault("enable_backdoor_attack", False)
    merged.setdefault("backdoor_client_oriented", False)
    merged.update(overrides)
    merged["result_tag"] = result_tag

    cmd = [PYTHON_BIN, "-u", "system/main.py"]
    for key, flag in MAIN_FLAG_MAP.items():
        if key not in merged:
            continue
        cmd.extend([flag, _stringify(merged[key])])
    return cmd


def build_saved_attack_eval_command(
    config_path,
    model_path,
    result_tag,
    stage_name,
    display_name,
    forget_strategy=None,
    mia_client_oriented=False,
):
    cmd = [
        PYTHON_BIN,
        "-u",
        "system/scripts/run_saved_attack_eval.py",
        "--config-path",
        str(config_path),
        "--model-path",
        str(model_path),
        "--result-tag",
        result_tag,
        "--stage-name",
        stage_name,
        "--display-name",
        display_name,
        "--target-client-id",
        str(TARGET_CLIENT_ID),
        "--mia-client-oriented",
        _stringify(mia_client_oriented),
    ]
    if forget_strategy is not None:
        cmd.extend(["--forget-strategy", str(forget_strategy)])
    return cmd


def run_command(cmd, keep_ld_library_path=False):
    env = os.environ.copy()
    if not keep_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)
    print("=" * 80, flush=True)
    print("Running:", " ".join(cmd), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def parse_attack_file(path):
    if not path.exists():
        return None
    payload = load_json(path)
    mia = payload.get("mia", {})
    shadow_training = mia.get("shadow_training", {}) or {}
    return {
        "path": str(path),
        "stage": payload.get("stage"),
        "auc": mia.get("auc"),
        "best_acc": mia.get("best_acc"),
        "score_name": mia.get("score_name"),
        "mia_mode": mia.get("mia_mode"),
        "model_source": mia.get("model_source"),
        "reference_mode": (mia.get("reference") or {}).get("reference_mode"),
        "shadow_enabled": shadow_training.get("enabled"),
        "shadow_num_clients": shadow_training.get("num_shadow_clients"),
        "shadow_num_samples": shadow_training.get("num_shadow_samples"),
    }


def summarise_runs(rows, summary_prefix):
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = SUMMARY_ROOT / f"{summary_prefix}.csv"
    json_path = SUMMARY_ROOT / f"{summary_prefix}.json"

    headers = [
        "level",
        "dataset",
        "method",
        "result_tag",
        "run_mode",
        "source_model_path",
        "post_model_path",
        "pre_stage",
        "pre_attack_path",
        "pre_auc",
        "pre_best_acc",
        "pre_mia_mode",
        "pre_model_source",
        "pre_shadow_clients",
        "pre_shadow_samples",
        "post_stage",
        "post_attack_path",
        "post_auc",
        "post_best_acc",
        "post_mia_mode",
        "post_model_source",
        "post_shadow_clients",
        "post_shadow_samples",
        "auc_delta_post_minus_pre",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"MIA summary saved to: {csv_path}")
    print(f"MIA summary saved to: {json_path}")


def measure_fu_or_fedcsa(level, method, suffix, skip_existing):
    dataset = dataset_name(level)
    result_tag = eval_result_tag(method, level, suffix)
    config_path = build_config_path(method, level)
    base_model_path = build_base_model_path(level)
    pre_attack_path = artifact_path("attack_pre_forget", dataset, result_tag, "json")
    post_attack_path = artifact_path("attack_post_forget", dataset, result_tag, "json")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base model: {base_model_path}")

    if not (skip_existing and pre_attack_path.exists() and post_attack_path.exists()):
        cmd = build_main_command(
            config_path=config_path,
            result_tag=result_tag,
            overrides={
                "forget_strategy": METHOD_FORGET_STRATEGY[method],
                "load_saved_model": True,
                "saved_model_path": str(base_model_path),
                "retrain_only": False,
                "mia_mode": "shadow_lr",
                "mia_client_oriented": False,
                "enable_backdoor_attack": False,
                "backdoor_client_oriented": False,
            },
        )
        run_command(cmd)

    pre = parse_attack_file(pre_attack_path)
    post = parse_attack_file(post_attack_path)
    return {
        "level": level,
        "dataset": dataset,
        "method": method,
        "result_tag": result_tag,
        "run_mode": "rerun_forgetting_from_base_model",
        "source_model_path": str(base_model_path),
        "post_model_path": str(artifact_path("forget_model", dataset, result_tag, "pt")),
        "pre_stage": None if pre is None else pre["stage"],
        "pre_attack_path": None if pre is None else pre["path"],
        "pre_auc": None if pre is None else pre["auc"],
        "pre_best_acc": None if pre is None else pre["best_acc"],
        "pre_mia_mode": None if pre is None else pre["mia_mode"],
        "pre_model_source": None if pre is None else pre["model_source"],
        "pre_shadow_clients": None if pre is None else pre["shadow_num_clients"],
        "pre_shadow_samples": None if pre is None else pre["shadow_num_samples"],
        "post_stage": None if post is None else post["stage"],
        "post_attack_path": None if post is None else post["path"],
        "post_auc": None if post is None else post["auc"],
        "post_best_acc": None if post is None else post["best_acc"],
        "post_mia_mode": None if post is None else post["mia_mode"],
        "post_model_source": None if post is None else post["model_source"],
        "post_shadow_clients": None if post is None else post["shadow_num_clients"],
        "post_shadow_samples": None if post is None else post["shadow_num_samples"],
        "auc_delta_post_minus_pre": (
            None
            if pre is None or post is None or pre["auc"] is None or post["auc"] is None
            else float(post["auc"] - pre["auc"])
        ),
    }


def measure_fedau_or_fedosd(level, method, suffix, skip_existing):
    dataset = dataset_name(level)
    result_tag = eval_result_tag(method, level, suffix)
    config_path = build_config_path(method, level)
    if method == "fedau":
        global_model_path = build_fedau_global_model_path(level)
    else:
        global_model_path = build_base_model_path(level)
    forget_model_path = build_forget_model_path(method, level)
    pre_attack_path = artifact_path("attack_pre_forget", dataset, result_tag, "json")
    post_attack_path = artifact_path("attack_post_forget", dataset, result_tag, "json")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not global_model_path.exists():
        raise FileNotFoundError(f"Missing global model: {global_model_path}")
    if not forget_model_path.exists():
        raise FileNotFoundError(f"Missing forget model: {forget_model_path}")

    if not (skip_existing and pre_attack_path.exists()):
        run_command(
            build_saved_attack_eval_command(
                config_path=config_path,
                model_path=global_model_path,
                result_tag=result_tag,
                stage_name="pre_forget",
                display_name=f"{method}_{level}_pre_mia",
                forget_strategy=METHOD_FORGET_STRATEGY[method],
                mia_client_oriented=False,
            )
        )
    if not (skip_existing and post_attack_path.exists()):
        run_command(
            build_saved_attack_eval_command(
                config_path=config_path,
                model_path=forget_model_path,
                result_tag=result_tag,
                stage_name="post_forget",
                display_name=f"{method}_{level}_post_mia",
                forget_strategy=METHOD_FORGET_STRATEGY[method],
                mia_client_oriented=False,
            )
        )

    pre = parse_attack_file(pre_attack_path)
    post = parse_attack_file(post_attack_path)
    return {
        "level": level,
        "dataset": dataset,
        "method": method,
        "result_tag": result_tag,
        "run_mode": "direct_global_and_forget_model_eval",
        "source_model_path": str(global_model_path),
        "post_model_path": str(forget_model_path),
        "pre_stage": None if pre is None else pre["stage"],
        "pre_attack_path": None if pre is None else pre["path"],
        "pre_auc": None if pre is None else pre["auc"],
        "pre_best_acc": None if pre is None else pre["best_acc"],
        "pre_mia_mode": None if pre is None else pre["mia_mode"],
        "pre_model_source": None if pre is None else pre["model_source"],
        "pre_shadow_clients": None if pre is None else pre["shadow_num_clients"],
        "pre_shadow_samples": None if pre is None else pre["shadow_num_samples"],
        "post_stage": None if post is None else post["stage"],
        "post_attack_path": None if post is None else post["path"],
        "post_auc": None if post is None else post["auc"],
        "post_best_acc": None if post is None else post["best_acc"],
        "post_mia_mode": None if post is None else post["mia_mode"],
        "post_model_source": None if post is None else post["model_source"],
        "post_shadow_clients": None if post is None else post["shadow_num_clients"],
        "post_shadow_samples": None if post is None else post["shadow_num_samples"],
        "auc_delta_post_minus_pre": (
            None
            if pre is None or post is None or pre["auc"] is None or post["auc"] is None
            else float(post["auc"] - pre["auc"])
        ),
    }


def measure_retrain(level, suffix, skip_existing):
    dataset = dataset_name(level)
    method = "retrain"
    result_tag = eval_result_tag(method, level, suffix)
    config_path = build_config_path(method, level)
    retrain_model_path = build_retrain_model_path(level)
    attack_path = artifact_path("attack_retrain_only", dataset, result_tag, "json")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not retrain_model_path.exists():
        raise FileNotFoundError(f"Missing retrain model: {retrain_model_path}")

    if not (skip_existing and attack_path.exists()):
        run_command(
            build_saved_attack_eval_command(
                config_path=config_path,
                model_path=retrain_model_path,
                result_tag=result_tag,
                stage_name="retrain_only",
                display_name=f"{method}_{level}_mia",
                forget_strategy=METHOD_FORGET_STRATEGY[method],
                mia_client_oriented=False,
            )
        )

    post = parse_attack_file(attack_path)
    return {
        "level": level,
        "dataset": dataset,
        "method": method,
        "result_tag": result_tag,
        "run_mode": "direct_retrain_model_eval",
        "source_model_path": "",
        "post_model_path": str(retrain_model_path),
        "pre_stage": "",
        "pre_attack_path": "",
        "pre_auc": "",
        "pre_best_acc": "",
        "pre_mia_mode": "",
        "pre_model_source": "",
        "pre_shadow_clients": "",
        "pre_shadow_samples": "",
        "post_stage": None if post is None else post["stage"],
        "post_attack_path": None if post is None else post["path"],
        "post_auc": None if post is None else post["auc"],
        "post_best_acc": None if post is None else post["best_acc"],
        "post_mia_mode": None if post is None else post["mia_mode"],
        "post_model_source": None if post is None else post["model_source"],
        "post_shadow_clients": None if post is None else post["shadow_num_clients"],
        "post_shadow_samples": None if post is None else post["shadow_num_samples"],
        "auc_delta_post_minus_pre": "",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure shadow-LR MIA for CIFAR10 clean seed42 heterogeneity experiments."
    )
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tag-suffix", default="shadow_mia")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for level in args.levels:
        for method in args.methods:
            if method in {"fu", "fedcsa"}:
                row = measure_fu_or_fedcsa(level, method, args.tag_suffix, args.skip_existing)
            elif method in {"fedau", "fedosd"}:
                row = measure_fedau_or_fedosd(level, method, args.tag_suffix, args.skip_existing)
            else:
                row = measure_retrain(level, args.tag_suffix, args.skip_existing)
            rows.append(row)

    summary_prefix = f"cifar10_clean_s42_shadow_mia_{args.tag_suffix}"
    summarise_runs(rows, summary_prefix)


if __name__ == "__main__":
    main()
