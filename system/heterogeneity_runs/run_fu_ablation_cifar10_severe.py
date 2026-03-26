import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "exp_configs"
OUTPUT_ROOT = PROJECT_ROOT / "system" / "heterogeneity_runs" / "fu_ablation_cifar10_severe"
DEFAULT_PYTHON = "/home/siguangchen/anaconda3/envs/zyj/bin/python"

RUN_CONFIGS = {
    42: {
        "dataset": "Cifar10_hetero_clean_s42_severe",
        "saved_model_path": str(
            RESULT_ROOT / "global_model_Cifar10_hetero_clean_s42_severe_FedAvg_test_1_fu_severe.pt"
        ),
    },
    321: {
        "dataset": "Cifar10_hetero_clean_s321_severe",
        "saved_model_path": str(
            RESULT_ROOT / "global_model_Cifar10_hetero_clean_s321_severe_FedAvg_test_1_fu_severe.pt"
        ),
    },
}

VARIANT_LABELS = {
    "full": "Full",
    "wo_h": "w/o H",
    "wo_a": "w/o A",
    "wo_u_mask": "w/o U-mask",
    "wo_retain": "w/o retain",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 severe FU/FedHAU ablations from saved checkpoints and summarize results."
    )
    parser.add_argument("--run", action="store_true", help="Execute the experiments. Otherwise only print commands.")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON) else sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--eval-gap", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 321])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_LABELS.keys()),
        default=list(VARIANT_LABELS.keys()),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose forget_effect JSON already exists.")
    parser.add_argument("--keep-ld-library-path", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=OUTPUT_ROOT / "summary.csv")
    parser.add_argument("--summary-md", type=Path, default=OUTPUT_ROOT / "summary.md")
    return parser.parse_args()


def result_tag_for_variant(variant):
    return f"fu_ablation_{variant}"


def artifact_path(prefix, dataset, result_tag, ext):
    return RESULT_ROOT / f"{prefix}_{dataset}_FedAvg_test_1_{result_tag}.{ext}"


def build_command(args, seed, variant):
    cfg = RUN_CONFIGS[seed]
    result_tag = result_tag_for_variant(variant)
    return [
        args.python,
        "-u",
        "system/main.py",
        "--dataset", cfg["dataset"],
        "--goal", "test",
        "--algorithm", "FedAvg",
        "--model", "ResNet18",
        "--device", args.device,
        "--device_id", args.device_id,
        "--global_rounds", str(args.global_rounds),
        "--eval_gap", str(args.eval_gap),
        "--random_seed", str(seed),
        "--target_client_id", str(args.target_client_id),
        "--forget_strategy", "gradient_reversal",
        "--load_saved_model", "true",
        "--saved_model_path", cfg["saved_model_path"],
        "--result_tag", result_tag,
        "--fu_ablation_mode", variant,
        "--use_optimized_forgetting", "true",
        "--enable_backdoor_attack", "true",
    ]


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


def read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_row(args, seed, variant):
    cfg = RUN_CONFIGS[seed]
    dataset = cfg["dataset"]
    result_tag = result_tag_for_variant(variant)
    eval_post = read_json(artifact_path("eval_post_forget", dataset, result_tag, "json"))
    forget_effect = read_json(artifact_path("forget_effect", dataset, result_tag, "json"))
    log_path = args.output_dir / dataset / f"{variant}.log"

    row = {
        "seed": seed,
        "dataset": dataset,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "result_tag": result_tag,
        "log_path": str(log_path),
        "status": "missing",
    }

    if not eval_post or not forget_effect:
        return row

    attack_post = forget_effect.get("attack_post_forget", {})
    row.update(
        {
            "status": "complete",
            "final_avg_acc": float(eval_post.get("average_accuracy", float("nan"))),
            "final_target_acc": float(eval_post.get("target_client_accuracy", float("nan"))),
            "final_retain_avg_acc": float(eval_post.get("retained_average_accuracy", float("nan"))),
            "global_buffer_avg_acc": float(eval_post.get("global_buffer_average_accuracy", float("nan"))),
            "global_buffer_target_acc": float(eval_post.get("global_buffer_target_client_accuracy", float("nan"))),
            "global_buffer_retain_avg_acc": float(eval_post.get("global_buffer_retained_average_accuracy", float("nan"))),
            "mia_post_auc": float(attack_post.get("mia_auc", float("nan"))),
            "backdoor_post_acc": float(attack_post.get("backdoor_acc", float("nan"))),
            "target_abs_change": float(forget_effect.get("target_abs_change", float("nan"))),
            "others_abs_change_avg": float(forget_effect.get("others_abs_change_avg", float("nan"))),
            "others_rel_change_avg": float(forget_effect.get("others_rel_change_avg", float("nan"))),
            "recovery_stage": str(forget_effect.get("recovery_stage", "")),
        }
    )
    return row


def add_delta_columns(df):
    full_df = (
        df[df["variant"] == "full"][["seed", "final_avg_acc", "final_target_acc", "final_retain_avg_acc", "mia_post_auc"]]
        .rename(
            columns={
                "final_avg_acc": "full_final_avg_acc",
                "final_target_acc": "full_final_target_acc",
                "final_retain_avg_acc": "full_final_retain_avg_acc",
                "mia_post_auc": "full_mia_post_auc",
            }
        )
    )
    merged = df.merge(full_df, on="seed", how="left")
    for col, full_col in [
        ("final_avg_acc", "full_final_avg_acc"),
        ("final_target_acc", "full_final_target_acc"),
        ("final_retain_avg_acc", "full_final_retain_avg_acc"),
        ("mia_post_auc", "full_mia_post_auc"),
    ]:
        merged[f"delta_vs_full_{col}"] = merged[col] - merged[full_col]
    return merged


def write_markdown_summary(df, path):
    lines = [
        "# CIFAR-10 Severe FU Ablation Summary",
        "",
        "Metrics follow the current FU evaluation pipeline.",
        "- `final_*`: local-buffer metrics from `eval_post_forget`.",
        "- `global_buffer_*`: global-buffer utility metrics from `eval_post_forget`.",
        "",
    ]

    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed].copy()
        sub = sub.sort_values("variant", key=lambda s: s.map({k: i for i, k in enumerate(VARIANT_LABELS)}))
        lines.append(f"## Seed {seed}")
        lines.append("")
        lines.append("| Variant | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in sub.iterrows():
            if row["status"] != "complete":
                lines.append(f"| {row['variant_label']} | missing | missing | missing | missing | missing | missing |")
                continue
            lines.append(
                f"| {row['variant_label']} | {row['final_avg_acc']:.4f} | {row['final_target_acc']:.4f} | "
                f"{row['final_retain_avg_acc']:.4f} | {row['global_buffer_retain_avg_acc']:.4f} | "
                f"{row['mia_post_auc']:.4f} | {row['backdoor_post_acc']:.4f} |"
            )
        lines.append("")

    agg = (
        df[df["status"] == "complete"]
        .groupby(["variant", "variant_label"], as_index=False)[
            [
                "final_avg_acc",
                "final_target_acc",
                "final_retain_avg_acc",
                "global_buffer_retain_avg_acc",
                "mia_post_auc",
                "backdoor_post_acc",
            ]
        ]
        .mean()
    )
    agg = agg.sort_values("variant", key=lambda s: s.map({k: i for i, k in enumerate(VARIANT_LABELS)}))
    lines.append("## Mean Across Seeds")
    lines.append("")
    lines.append("| Variant | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['variant_label']} | {row['final_avg_acc']:.4f} | {row['final_target_acc']:.4f} | "
            f"{row['final_retain_avg_acc']:.4f} | {row['global_buffer_retain_avg_acc']:.4f} | "
            f"{row['mia_post_auc']:.4f} | {row['backdoor_post_acc']:.4f} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    planned_rows = []
    for seed in args.seeds:
        if seed not in RUN_CONFIGS:
            raise ValueError(f"Unsupported seed: {seed}. Available seeds: {sorted(RUN_CONFIGS)}")
        dataset = RUN_CONFIGS[seed]["dataset"]
        for variant in args.variants:
            result_tag = result_tag_for_variant(variant)
            cmd = build_command(args, seed, variant)
            log_path = args.output_dir / dataset / f"{variant}.log"
            forget_path = artifact_path("forget_effect", dataset, result_tag, "json")
            if args.skip_existing and forget_path.exists():
                print(f"[skip] seed={seed} variant={variant} -> {forget_path}")
            else:
                print(" ".join(cmd))
                if args.run:
                    run_logged_command(args, cmd, log_path)
            planned_rows.append(collect_row(args, seed, variant))

    df = pd.DataFrame(planned_rows)
    if not df.empty:
        df = add_delta_columns(df)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.summary_csv, index=False)
        write_markdown_summary(df, args.summary_md)
        print(f"Summary written to: {args.summary_csv}")
        print(f"Summary written to: {args.summary_md}")


if __name__ == "__main__":
    main()
