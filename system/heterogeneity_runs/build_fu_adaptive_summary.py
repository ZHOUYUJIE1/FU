import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT.parents[1] / "results" / "exp_configs"
DEFAULT_BASE_SUMMARY = ROOT / "Cifar10_FedAvg_heterogeneity_summary.csv"
DEFAULT_OUTPUT_SUMMARY = ROOT / "Cifar10_FedAvg_heterogeneity_summary_fu_adaptive.csv"
DEFAULT_DELTA_OUTPUT = ROOT / "Cifar10_FedAvg_fu_adaptive_delta.csv"
LEVEL_ORDER = ["mild", "moderate", "severe"]


def parse_args():
    parser = argparse.ArgumentParser(description="Replace FU rows in the heterogeneity summary with specified FU result tags.")
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--delta-output", type=Path, default=DEFAULT_DELTA_OUTPUT)
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--goal", type=str, default="test")
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument(
        "--fu-tag",
        action="append",
        default=[],
        help="Level-specific FU result tag in the form level:result_tag. Defaults to fu_{level}_adaptive for unspecified levels.",
    )
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_attack_file(path):
    parsed = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            try:
                parsed[key] = float(value.strip())
            except ValueError:
                parsed[key] = value.strip()
    return parsed


def safe_client_metric(eval_data, target_client_id, retain=False):
    client_accs = eval_data.get("client_accuracies") or []
    if not client_accs:
        return None

    if retain:
        retained = [acc for idx, acc in enumerate(client_accs) if idx != target_client_id]
        return float(np.mean(retained)) if retained else None

    if 0 <= target_client_id < len(client_accs):
        return float(client_accs[target_client_id])
    return None


def build_result_path(prefix, dataset_name, algorithm, goal, times, result_tag, extension):
    filename = f"{prefix}_{dataset_name}_{algorithm}_{goal}_{times}_{result_tag}.{extension}"
    return RESULT_ROOT / filename


def parse_fu_tags(tag_overrides):
    parsed = {level: f"fu_{level}_adaptive" for level in LEVEL_ORDER}
    for item in tag_overrides:
        if ":" not in item:
            raise ValueError(f"Invalid --fu-tag value '{item}'. Expected level:result_tag")
        level, result_tag = item.split(":", 1)
        level = level.strip().lower()
        result_tag = result_tag.strip()
        if level not in LEVEL_ORDER:
            raise ValueError(f"Unsupported level '{level}' in --fu-tag")
        if not result_tag:
            raise ValueError(f"Empty result_tag in --fu-tag '{item}'")
        parsed[level] = result_tag
    return parsed


def build_fu_row(level, original_row, args, fu_tags):
    dataset_name = f"Cifar10_hetero_{level}"
    result_tag = fu_tags[level]

    pre_eval = read_json(build_result_path("eval_pre_forget", dataset_name, args.algorithm, args.goal, args.times, result_tag, "json"))
    post_eval = read_json(build_result_path("eval_post_forget", dataset_name, args.algorithm, args.goal, args.times, result_tag, "json"))
    recovery_eval_path = build_result_path("eval_post_recovery", dataset_name, args.algorithm, args.goal, args.times, result_tag, "json")
    recovery_eval = read_json(recovery_eval_path) if recovery_eval_path.exists() else None
    forget_summary = read_json(build_result_path("forget_effect", dataset_name, args.algorithm, args.goal, args.times, result_tag, "json"))
    attack_pre = parse_attack_file(build_result_path("attack_pre_forget", dataset_name, args.algorithm, args.goal, args.times, result_tag, "txt"))
    attack_post = parse_attack_file(build_result_path("attack_post_forget", dataset_name, args.algorithm, args.goal, args.times, result_tag, "txt"))
    final_eval = recovery_eval or post_eval

    row = dict(original_row)
    row.update({
        "dataset": dataset_name,
        "display_name": "FU",
        "result_tag": result_tag,
        "status": "complete",
        "pre_avg_acc": pre_eval.get("average_accuracy"),
        "post_avg_acc": post_eval.get("average_accuracy"),
        "recovery_avg_acc": recovery_eval.get("average_accuracy") if recovery_eval else None,
        "final_avg_acc": final_eval.get("average_accuracy"),
        "pre_target_acc": safe_client_metric(pre_eval, args.target_client_id),
        "post_target_acc": safe_client_metric(post_eval, args.target_client_id),
        "final_target_acc": safe_client_metric(final_eval, args.target_client_id),
        "pre_retain_avg_acc": safe_client_metric(pre_eval, args.target_client_id, retain=True),
        "post_retain_avg_acc": safe_client_metric(post_eval, args.target_client_id, retain=True),
        "final_retain_avg_acc": safe_client_metric(final_eval, args.target_client_id, retain=True),
        "target_acc_change": forget_summary.get("target_abs_change"),
        "target_acc_change_ratio": forget_summary.get("target_rel_change"),
        "others_acc_change_avg": forget_summary.get("others_abs_change_avg"),
        "others_acc_change_ratio": forget_summary.get("others_rel_change_avg"),
        "recovery_effect": forget_summary.get("recovery_effect"),
        "recovery_stage": forget_summary.get("recovery_stage"),
        "mia_pre_auc": attack_pre.get("mia_auc"),
        "mia_post_auc": attack_post.get("mia_auc"),
        "mia_auc_drop": (
            attack_pre.get("mia_auc") - attack_post.get("mia_auc")
            if attack_pre.get("mia_auc") is not None and attack_post.get("mia_auc") is not None
            else None
        ),
        "backdoor_pre_acc": attack_pre.get("backdoor_acc"),
        "backdoor_post_acc": attack_post.get("backdoor_acc"),
        "final_avg_auc": "",
        "final_mia_auc": "",
        "final_backdoor_acc": "",
    })
    return row


def main():
    args = parse_args()
    fu_tags = parse_fu_tags(args.fu_tag)

    with args.base_summary.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    updated_rows = []
    delta_rows = []

    for row in rows:
        if row["method"] != "fu":
            updated_rows.append(row)
            continue

        level = row["level"]
        new_row = build_fu_row(level, row, args, fu_tags)
        updated_rows.append(new_row)

        delta_rows.append({
            "level": level,
            "old_result_tag": row["result_tag"],
            "new_result_tag": new_row["result_tag"],
            "final_avg_acc_old": float(row["final_avg_acc"]),
            "final_avg_acc_new": float(new_row["final_avg_acc"]),
            "final_avg_acc_delta": float(new_row["final_avg_acc"]) - float(row["final_avg_acc"]),
            "final_target_acc_old": float(row["final_target_acc"]),
            "final_target_acc_new": float(new_row["final_target_acc"]),
            "final_target_acc_delta": float(new_row["final_target_acc"]) - float(row["final_target_acc"]),
            "final_retain_avg_acc_old": float(row["final_retain_avg_acc"]),
            "final_retain_avg_acc_new": float(new_row["final_retain_avg_acc"]),
            "final_retain_avg_acc_delta": float(new_row["final_retain_avg_acc"]) - float(row["final_retain_avg_acc"]),
            "post_target_acc_new": float(new_row["post_target_acc"]),
            "recovery_stage_new": new_row["recovery_stage"],
        })

    updated_df = pd.DataFrame(updated_rows)
    updated_df["level"] = pd.Categorical(updated_df["level"], categories=LEVEL_ORDER, ordered=True)
    method_order = {"fu": 0, "fedau": 1, "fedcsa": 2, "fedosd": 3, "retrain": 4}
    updated_df["method_order"] = updated_df["method"].map(method_order)
    updated_df = updated_df.sort_values(["level", "method_order"]).drop(columns=["method_order"])
    updated_df.to_csv(args.output_summary, index=False)

    delta_df = pd.DataFrame(delta_rows)
    delta_df["level"] = pd.Categorical(delta_df["level"], categories=LEVEL_ORDER, ordered=True)
    delta_df = delta_df.sort_values("level")
    delta_df.to_csv(args.delta_output, index=False)

    print(f"FU summary written to: {args.output_summary}")
    print(f"FU delta summary written to: {args.delta_output}")


if __name__ == "__main__":
    main()
