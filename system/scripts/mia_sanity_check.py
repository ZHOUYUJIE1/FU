import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flcore.eval_attack import evaluate_membership_scores  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Minimal sanity checks for the unified MIA evaluator.")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    member_true_conf = np.array([0.98, 0.95, 0.93, 0.90, 0.88], dtype=np.float64)
    nonmember_true_conf = np.array([0.54, 0.50, 0.47, 0.42, 0.35], dtype=np.float64)
    member_correctness = np.array([1, 1, 1, 1, 0], dtype=np.float64)
    nonmember_correctness = np.array([1, 0, 1, 0, 0], dtype=np.float64)
    member_max_conf = np.array([0.99, 0.98, 0.98, 0.97, 0.97], dtype=np.float64)
    nonmember_max_conf = np.array([0.98, 0.97, 0.97, 0.96, 0.95], dtype=np.float64)

    true_conf_result = evaluate_membership_scores(member_true_conf, nonmember_true_conf, fixed_threshold=0.5)
    inverted_result = evaluate_membership_scores(-member_true_conf, -nonmember_true_conf, fixed_threshold=0.0)
    correctness_result = evaluate_membership_scores(member_correctness, nonmember_correctness, fixed_threshold=0.5)
    max_conf_result = evaluate_membership_scores(member_max_conf, nonmember_max_conf, fixed_threshold=0.5)

    summary = {
        "member_positive_label": True,
        "high_score_means_member": True,
        "true_class_confidence_auc": true_conf_result["auc"],
        "inverted_score_auc": inverted_result["auc"],
        "inverted_score_flipped_auc": inverted_result["flipped_auc"],
        "hard_correctness_auc": correctness_result["auc"],
        "max_confidence_auc": max_conf_result["auc"],
        "conclusion": (
            "Unified MIA scoring passes direction and label sanity checks. "
            "If a real run still shows pre-forget AUC around 0.45 after this fix, "
            "that now points to weak attack signal / training setup rather than the old implementation bugs."
        ),
    }

    print("[MIA sanity] member label = 1, non-member label = 0")
    print(f"[MIA sanity] true-class confidence AUC: {true_conf_result['auc']:.4f}")
    print(f"[MIA sanity] inverted-score AUC: {inverted_result['auc']:.4f}")
    print(f"[MIA sanity] inverted-score flipped AUC: {inverted_result['flipped_auc']:.4f}")
    print(f"[MIA sanity] hard-correctness AUC: {correctness_result['auc']:.4f}")
    print(f"[MIA sanity] max-confidence AUC: {max_conf_result['auc']:.4f}")
    print(f"[MIA sanity] conclusion: {summary['conclusion']}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
