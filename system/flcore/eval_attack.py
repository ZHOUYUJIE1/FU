import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data


def _normalize_mia_reference_mode(mode):
    mode_key = str(mode).strip().lower()
    if mode_key in {
        "target_train_vs_other_train",
        "other_train",
        "other_clients_train",
        "client_level",
    }:
        return "other_train"
    return "target_test"


def _normalize_mia_mode(mode):
    mode_key = str(mode).strip().lower()
    if mode_key in {"shadow_lr", "shadow", "shadow_logistic_regression"}:
        return "shadow_lr"
    if mode_key in {"target_lr", "target_ref_lr", "direct_lr", "global_lr"}:
        return "target_lr"
    return "threshold_tconf"


def _extract_scalar_label(sample):
    label = sample[1]
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


def _clone_nested(item):
    if torch.is_tensor(item):
        return item.detach().cpu().clone()
    if isinstance(item, tuple):
        return tuple(_clone_nested(value) for value in item)
    if isinstance(item, list):
        return [_clone_nested(value) for value in item]
    return copy.deepcopy(item)


def _move_to_device(item, device):
    if torch.is_tensor(item):
        return item.to(device)
    if isinstance(item, tuple):
        return tuple(_move_to_device(value, device) for value in item)
    if isinstance(item, list):
        return [_move_to_device(value, device) for value in item]
    return item


def _load_backdoor_eval_data(client, data_source="test"):
    is_train = str(data_source).strip().lower() == "train"
    return list(
        read_client_data(
            client.dataset,
            client.id,
            is_train=is_train,
            few_shot=client.few_shot,
            apply_train_transform=False,
        )
    )


def build_membership_reference(
    client,
    max_samples=None,
    nonmember_source_mode="target_train_vs_other_train",
    all_clients=None,
):
    """Build a deterministic, clean member/non-member reference set for one target client."""
    member_data = list(
        read_client_data(
            client.dataset,
            client.id,
            is_train=True,
            few_shot=client.few_shot,
            apply_train_transform=False,
        )
    )
    ref_mode = _normalize_mia_reference_mode(nonmember_source_mode)

    if ref_mode == "other_train":
        nonmember_pool = []
        for other_client in (all_clients or []):
            if int(other_client.id) == int(client.id):
                continue
            nonmember_pool.extend(
                read_client_data(
                    other_client.dataset,
                    other_client.id,
                    is_train=True,
                    few_shot=other_client.few_shot,
                    apply_train_transform=False,
                )
            )

        if nonmember_pool:
            sample_count = len(member_data)
            if max_samples is not None and max_samples > 0:
                sample_count = min(sample_count, int(max_samples))
            member_subset = member_data[:sample_count]

            label_buckets = {}
            for sample in nonmember_pool:
                label = _extract_scalar_label(sample)
                label_buckets.setdefault(label, []).append(sample)
            label_cursors = {label: 0 for label in label_buckets}
            fallback_cursor = 0

            nonmember_data = []
            for member_sample in member_subset:
                label = _extract_scalar_label(member_sample)
                candidates = label_buckets.get(label, [])
                if candidates:
                    cursor = label_cursors[label]
                    chosen = candidates[cursor % len(candidates)]
                    label_cursors[label] = cursor + 1
                else:
                    chosen = nonmember_pool[fallback_cursor % len(nonmember_pool)]
                    fallback_cursor += 1
                nonmember_data.append(chosen)

            member_data = [_clone_nested(sample) for sample in member_subset]
            nonmember_data = [_clone_nested(sample) for sample in nonmember_data]
            return {
                "client_id": int(client.id),
                "dataset": client.dataset,
                "member_data": member_data,
                "nonmember_data": nonmember_data,
                "member_source": "target_client_train_clean",
                "nonmember_source": "other_clients_train_clean_label_matched",
                "sample_count_per_split": int(sample_count),
                "uses_fixed_reference": True,
                "uses_clean_train_split": True,
                "target_client_only": True,
                "reference_mode": "target_train_vs_other_train",
            }

    nonmember_data = list(
        read_client_data(
            client.dataset,
            client.id,
            is_train=False,
            few_shot=client.few_shot,
            apply_train_transform=False,
        )
    )
    sample_count = min(len(member_data), len(nonmember_data))
    if max_samples is not None and max_samples > 0:
        sample_count = min(sample_count, int(max_samples))

    member_data = [_clone_nested(sample) for sample in member_data[:sample_count]]
    nonmember_data = [_clone_nested(sample) for sample in nonmember_data[:sample_count]]

    return {
        "client_id": int(client.id),
        "dataset": client.dataset,
        "member_data": member_data,
        "nonmember_data": nonmember_data,
        "member_source": "target_client_train_clean",
        "nonmember_source": "target_client_test_clean",
        "sample_count_per_split": int(sample_count),
        "uses_fixed_reference": True,
        "uses_clean_train_split": True,
        "target_client_only": True,
        "reference_mode": "target_train_vs_target_test",
    }


def _collect_prediction_stats(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    true_confidence = []
    neg_loss = []
    max_confidence = []
    correctness = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = _move_to_device(x, device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            true_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")

            true_confidence.extend(true_prob.detach().cpu().numpy().tolist())
            neg_loss.extend((-per_sample_loss).detach().cpu().numpy().tolist())
            max_confidence.extend(torch.max(probs, dim=1).values.detach().cpu().numpy().tolist())
            correctness.extend(pred.eq(y).float().detach().cpu().numpy().tolist())

    return {
        "true_class_confidence": np.asarray(true_confidence, dtype=np.float64),
        "negative_loss": np.asarray(neg_loss, dtype=np.float64),
        "max_confidence": np.asarray(max_confidence, dtype=np.float64),
        "correctness": np.asarray(correctness, dtype=np.float64),
    }


def _collect_prediction_features(model, dataset, device, batch_size=256):
    """
    Per-sample features used by learned MIA attacks.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    feature_blocks = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = _move_to_device(x, device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            true_prob = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")
            max_confidence = torch.max(probs, dim=1).values
            correctness = pred.eq(y).float()

            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            if probs.shape[1] >= 2:
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
            else:
                margin = max_confidence

            features = torch.stack((
                true_prob,
                -per_sample_loss,
                max_confidence,
                correctness,
                -entropy,
                margin,
            ), dim=1)
            feature_blocks.append(features.detach().cpu().numpy())

    if not feature_blocks:
        return np.zeros((0, 6), dtype=np.float64)
    return np.concatenate(feature_blocks, axis=0).astype(np.float64)


def _split_attack_features(features, train_ratio=0.5, seed=0):
    feature_array = np.asarray(features, dtype=np.float64)
    sample_count = int(feature_array.shape[0])
    if sample_count < 2:
        return None, None

    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(sample_count)
    train_count = int(round(sample_count * float(train_ratio)))
    train_count = max(1, min(sample_count - 1, train_count))

    train_idx = indices[:train_count]
    eval_idx = indices[train_count:]
    return feature_array[train_idx], feature_array[eval_idx]


def evaluate_membership_scores(member_scores, nonmember_scores, fixed_threshold=0.5):
    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(nonmember_scores), dtype=np.int64),
    ])
    y_score = np.concatenate([member_scores, nonmember_scores]).astype(np.float64)

    if len(np.unique(y_true)) < 2 or len(y_score) == 0:
        return {
            "auc": 0.5,
            "best_acc": 0.0,
            "best_thr": float(fixed_threshold),
            "fixed_thr": float(fixed_threshold),
            "fixed_acc": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
            "best_f1": 0.0,
            "fixed_precision": 0.0,
            "fixed_recall": 0.0,
            "fixed_f1": 0.0,
            "flipped_auc": 0.5,
        }

    auc = float(roc_auc_score(y_true, y_score))
    flipped_auc = float(roc_auc_score(y_true, -y_score))

    unique_scores = np.unique(y_score)
    thresholds = np.concatenate((
        [np.inf],
        np.sort(unique_scores)[::-1],
        [-np.inf],
    ))

    best_acc = -1.0
    best_thr = float(fixed_threshold)
    for thr in thresholds:
        acc = float(accuracy_score(y_true, (y_score >= thr).astype(np.int64)))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)

    best_pred = (y_score >= best_thr).astype(np.int64)
    fixed_pred = (y_score >= fixed_threshold).astype(np.int64)

    fixed_acc = float(accuracy_score(y_true, fixed_pred))
    best_precision, best_recall, best_f1, _ = precision_recall_fscore_support(
        y_true,
        best_pred,
        average="binary",
        zero_division=0,
    )
    fixed_precision, fixed_recall, fixed_f1, _ = precision_recall_fscore_support(
        y_true,
        fixed_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "auc": auc,
        "best_acc": best_acc,
        "best_thr": best_thr,
        "fixed_thr": float(fixed_threshold),
        "fixed_acc": fixed_acc,
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
        "best_f1": float(best_f1),
        "fixed_precision": float(fixed_precision),
        "fixed_recall": float(fixed_recall),
        "fixed_f1": float(fixed_f1),
        "flipped_auc": flipped_auc,
    }


def _run_single_membership_attack(model, client, device, reference_data=None, max_samples=None, batch_size=256):
    """
    Unified membership inference evaluation on a fixed target-client reference set.

    Positive label is always `member=1`, negative label is `non-member=0`.
    Primary attack score is the true-class confidence: higher score => more likely member.
    """
    reference = reference_data or build_membership_reference(client, max_samples=max_samples)
    member_stats = _collect_prediction_stats(
        model,
        reference["member_data"],
        device=device,
        batch_size=batch_size,
    )
    nonmember_stats = _collect_prediction_stats(
        model,
        reference["nonmember_data"],
        device=device,
        batch_size=batch_size,
    )

    metric_results = {}
    for metric_name, fixed_thr in (
        ("true_class_confidence", 0.5),
        ("negative_loss", 0.0),
        ("max_confidence", 0.5),
        ("correctness", 0.5),
    ):
        metric_results[metric_name] = evaluate_membership_scores(
            member_stats[metric_name],
            nonmember_stats[metric_name],
            fixed_threshold=fixed_thr,
        )

    primary_metric = "true_class_confidence"
    primary = metric_results[primary_metric]

    return {
        "auc": primary["auc"],
        "best_acc": primary["best_acc"],
        "best_thr": primary["best_thr"],
        "mia_sr": primary["best_acc"],
        "fixed_thr": primary["fixed_thr"],
        "fixed_acc": primary["fixed_acc"],
        "score_name": primary_metric,
        "metric_details": metric_results,
        "reference": {
            "client_id": reference["client_id"],
            "dataset": reference["dataset"],
            "member_source": reference["member_source"],
            "nonmember_source": reference["nonmember_source"],
            "sample_count_per_split": reference["sample_count_per_split"],
            "uses_fixed_reference": reference["uses_fixed_reference"],
            "uses_clean_train_split": reference["uses_clean_train_split"],
            "target_client_only": reference["target_client_only"],
            "reference_mode": reference.get("reference_mode", "target_train_vs_target_test"),
        },
        "sanity_checks": {
            "member_label_is_positive": True,
            "higher_score_means_more_likely_member": True,
            "primary_score_uses_true_label": True,
            "hard_label_correctness_is_auxiliary_only": True,
            "flipped_primary_auc": primary["flipped_auc"],
        },
    }, member_stats[primary_metric], nonmember_stats[primary_metric]


def _run_shadow_lr_membership_attack(
    model,
    client,
    device,
    all_clients=None,
    reference_data=None,
    max_samples=None,
    batch_size=256,
    mia_reference_mode="target_train_vs_other_train",
):
    """
    Train a shadow logistic-regression attack on retained clients and
    evaluate on the target client.
    """
    eval_clients = list(all_clients) if all_clients is not None else []
    target_reference = reference_data
    if target_reference is None or target_reference.get("reference_mode") != "target_train_vs_target_test":
        target_reference = build_membership_reference(
            client,
            max_samples=max_samples,
            nonmember_source_mode="target_train_vs_target_test",
        )

    baseline_result, baseline_member_scores, baseline_nonmember_scores = _run_single_membership_attack(
        model,
        client,
        device,
        reference_data=target_reference,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    shadow_clients = [c for c in eval_clients if int(c.id) != int(client.id)]
    if not shadow_clients:
        baseline_result["mia_mode"] = "threshold_tconf_fallback"
        baseline_result["shadow_training"] = {
            "enabled": False,
            "reason": "no_shadow_clients",
            "num_shadow_clients": 0,
            "num_shadow_samples": 0,
        }
        return baseline_result, baseline_member_scores, baseline_nonmember_scores

    target_member_x = _collect_prediction_features(
        model, target_reference["member_data"], device=device, batch_size=batch_size
    )
    target_nonmember_x = _collect_prediction_features(
        model, target_reference["nonmember_data"], device=device, batch_size=batch_size
    )

    shadow_x_parts = []
    shadow_y_parts = []
    used_shadow_clients = 0
    total_shadow_samples = 0
    for shadow_client in shadow_clients:
        shadow_reference = build_membership_reference(
            shadow_client,
            max_samples=max_samples,
            nonmember_source_mode="target_train_vs_target_test",
        )
        shadow_member_x = _collect_prediction_features(
            model, shadow_reference["member_data"], device=device, batch_size=batch_size
        )
        shadow_nonmember_x = _collect_prediction_features(
            model, shadow_reference["nonmember_data"], device=device, batch_size=batch_size
        )
        if len(shadow_member_x) == 0 or len(shadow_nonmember_x) == 0:
            continue

        shadow_x_parts.append(shadow_member_x)
        shadow_y_parts.append(np.ones(len(shadow_member_x), dtype=np.int64))
        shadow_x_parts.append(shadow_nonmember_x)
        shadow_y_parts.append(np.zeros(len(shadow_nonmember_x), dtype=np.int64))
        used_shadow_clients += 1
        total_shadow_samples += int(len(shadow_member_x) + len(shadow_nonmember_x))

    if not shadow_x_parts:
        baseline_result["mia_mode"] = "threshold_tconf_fallback"
        baseline_result["shadow_training"] = {
            "enabled": False,
            "reason": "empty_shadow_dataset",
            "num_shadow_clients": 0,
            "num_shadow_samples": 0,
        }
        return baseline_result, baseline_member_scores, baseline_nonmember_scores

    shadow_x = np.concatenate(shadow_x_parts, axis=0)
    shadow_y = np.concatenate(shadow_y_parts, axis=0)
    if len(np.unique(shadow_y)) < 2:
        baseline_result["mia_mode"] = "threshold_tconf_fallback"
        baseline_result["shadow_training"] = {
            "enabled": False,
            "reason": "single_class_shadow_labels",
            "num_shadow_clients": int(used_shadow_clients),
            "num_shadow_samples": int(total_shadow_samples),
        }
        return baseline_result, baseline_member_scores, baseline_nonmember_scores

    attack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        ),
    )
    attack_model.fit(shadow_x, shadow_y)

    member_scores = attack_model.predict_proba(target_member_x)[:, 1].astype(np.float64)
    nonmember_scores = attack_model.predict_proba(target_nonmember_x)[:, 1].astype(np.float64)
    primary = evaluate_membership_scores(member_scores, nonmember_scores, fixed_threshold=0.5)

    metric_details = dict(baseline_result.get("metric_details", {}))
    metric_details["shadow_lr_probability"] = primary

    return {
        "auc": primary["auc"],
        "best_acc": primary["best_acc"],
        "best_thr": primary["best_thr"],
        "mia_sr": primary["best_acc"],
        "fixed_thr": primary["fixed_thr"],
        "fixed_acc": primary["fixed_acc"],
        "score_name": "shadow_lr_probability",
        "metric_details": metric_details,
        "reference": baseline_result.get("reference", {}),
        "sanity_checks": {
            **baseline_result.get("sanity_checks", {}),
            "shadow_model_enabled": True,
            "flipped_primary_auc": primary["flipped_auc"],
        },
        "mia_mode": "shadow_lr",
        "shadow_training": {
            "enabled": True,
            "num_shadow_clients": int(used_shadow_clients),
            "num_shadow_samples": int(total_shadow_samples),
            "feature_dim": int(shadow_x.shape[1]),
            "member_source": "shadow_client_train_clean",
            "nonmember_source": "shadow_client_test_clean",
            "reference_mode": "shadow_client_train_vs_shadow_client_test",
            "target_eval_reference_mode": "target_train_vs_target_test",
        },
    }, member_scores, nonmember_scores


def _run_target_lr_membership_attack(
    model,
    client,
    device,
    reference_data=None,
    max_samples=None,
    batch_size=256,
    attack_train_ratio=0.5,
):
    """
    Train a logistic-regression attack directly on a holdout split of the
    target client's own train/test reference set.
    """
    target_reference = reference_data
    if target_reference is None or target_reference.get("reference_mode") != "target_train_vs_target_test":
        target_reference = build_membership_reference(
            client,
            max_samples=max_samples,
            nonmember_source_mode="target_train_vs_target_test",
        )

    baseline_result, baseline_member_scores, baseline_nonmember_scores = _run_single_membership_attack(
        model,
        client,
        device,
        reference_data=target_reference,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    target_member_x = _collect_prediction_features(
        model,
        target_reference["member_data"],
        device=device,
        batch_size=batch_size,
    )
    target_nonmember_x = _collect_prediction_features(
        model,
        target_reference["nonmember_data"],
        device=device,
        batch_size=batch_size,
    )

    member_train_x, member_eval_x = _split_attack_features(
        target_member_x,
        train_ratio=attack_train_ratio,
        seed=0,
    )
    nonmember_train_x, nonmember_eval_x = _split_attack_features(
        target_nonmember_x,
        train_ratio=attack_train_ratio,
        seed=1,
    )
    if (
        member_train_x is None or member_eval_x is None
        or nonmember_train_x is None or nonmember_eval_x is None
    ):
        baseline_result["mia_mode"] = "threshold_tconf_fallback"
        baseline_result["attack_training"] = {
            "enabled": False,
            "reason": "not_enough_target_samples_for_target_lr",
            "member_source": target_reference["member_source"],
            "nonmember_source": target_reference["nonmember_source"],
            "reference_mode": target_reference["reference_mode"],
        }
        return baseline_result, baseline_member_scores, baseline_nonmember_scores

    attack_train_x = np.concatenate((member_train_x, nonmember_train_x), axis=0)
    attack_train_y = np.concatenate((
        np.ones(len(member_train_x), dtype=np.int64),
        np.zeros(len(nonmember_train_x), dtype=np.int64),
    ))
    if len(np.unique(attack_train_y)) < 2:
        baseline_result["mia_mode"] = "threshold_tconf_fallback"
        baseline_result["attack_training"] = {
            "enabled": False,
            "reason": "single_class_target_lr_labels",
            "member_source": target_reference["member_source"],
            "nonmember_source": target_reference["nonmember_source"],
            "reference_mode": target_reference["reference_mode"],
        }
        return baseline_result, baseline_member_scores, baseline_nonmember_scores

    attack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        ),
    )
    attack_model.fit(attack_train_x, attack_train_y)

    member_scores = attack_model.predict_proba(member_eval_x)[:, 1].astype(np.float64)
    nonmember_scores = attack_model.predict_proba(nonmember_eval_x)[:, 1].astype(np.float64)
    primary = evaluate_membership_scores(member_scores, nonmember_scores, fixed_threshold=0.5)

    metric_details = dict(baseline_result.get("metric_details", {}))
    metric_details["target_lr_probability"] = primary
    training_details = {
        "enabled": True,
        "scheme": "target_reference_holdout_lr",
        "train_ratio": float(attack_train_ratio),
        "member_source": target_reference["member_source"],
        "nonmember_source": target_reference["nonmember_source"],
        "reference_mode": target_reference["reference_mode"],
        "feature_dim": int(attack_train_x.shape[1]),
        "attack_train_member_samples": int(len(member_train_x)),
        "attack_train_nonmember_samples": int(len(nonmember_train_x)),
        "attack_eval_member_samples": int(len(member_eval_x)),
        "attack_eval_nonmember_samples": int(len(nonmember_eval_x)),
        "uses_other_clients": False,
    }

    return {
        "auc": primary["auc"],
        "best_acc": primary["best_acc"],
        "best_thr": primary["best_thr"],
        "mia_sr": primary["best_acc"],
        "fixed_thr": primary["fixed_thr"],
        "fixed_acc": primary["fixed_acc"],
        "score_name": "target_lr_probability",
        "metric_details": metric_details,
        "reference": baseline_result.get("reference", {}),
        "sanity_checks": {
            **baseline_result.get("sanity_checks", {}),
            "shadow_model_enabled": False,
            "target_reference_lr_enabled": True,
            "flipped_primary_auc": primary["flipped_auc"],
        },
        "mia_mode": "target_lr",
        "attack_training": training_details,
        "shadow_training": {
            "enabled": False,
            "reason": "target_lr_does_not_use_shadow_clients",
        },
    }, member_scores, nonmember_scores


def _summarize_client_metrics(client_metrics):
    summary = {
        "num_clients": int(len(client_metrics)),
        "client_ids": [int(item["client_id"]) for item in client_metrics],
    }
    if not client_metrics:
        for metric in ("auc", "best_acc", "fixed_acc", "best_precision", "best_recall", "best_f1"):
            summary[f"{metric}_mean"] = 0.0
            summary[f"{metric}_std"] = 0.0
        return summary

    for metric in ("auc", "best_acc", "fixed_acc", "best_precision", "best_recall", "best_f1"):
        values = np.asarray([item[metric] for item in client_metrics], dtype=np.float64)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    return summary


def membership_inference_attack(
    model,
    client,
    device,
    reference_data=None,
    max_samples=None,
    batch_size=256,
    mia_mode="target_lr",
    mia_reference_mode="target_train_vs_other_train",
    client_oriented=False,
    all_clients=None,
    forgotten_client_ids=None,
    retained_client_ids=None,
):
    mode = _normalize_mia_mode(mia_mode)
    if mode in {"shadow_lr", "target_lr"}:
        primary_reference = reference_data
        if primary_reference is None or primary_reference.get("reference_mode") != "target_train_vs_target_test":
            primary_reference = build_membership_reference(
                client,
                max_samples=max_samples,
                nonmember_source_mode="target_train_vs_target_test",
            )
    else:
        primary_reference = reference_data or build_membership_reference(
            client,
            max_samples=max_samples,
            nonmember_source_mode=mia_reference_mode,
            all_clients=all_clients,
        )
    if mode == "shadow_lr":
        result, _, _ = _run_shadow_lr_membership_attack(
            model,
            client,
            device,
            all_clients=all_clients,
            reference_data=primary_reference,
            max_samples=max_samples,
            batch_size=batch_size,
            mia_reference_mode=mia_reference_mode,
        )
    elif mode == "target_lr":
        result, _, _ = _run_target_lr_membership_attack(
            model,
            client,
            device,
            reference_data=primary_reference,
            max_samples=max_samples,
            batch_size=batch_size,
        )
    else:
        result, _, _ = _run_single_membership_attack(
            model,
            client,
            device,
            reference_data=primary_reference,
            max_samples=max_samples,
            batch_size=batch_size,
        )

    if "mia_mode" not in result:
        result["mia_mode"] = "threshold_tconf"
    if not client_oriented:
        return result

    eval_clients = list(all_clients) if all_clients is not None else [client]
    if len(eval_clients) == 0:
        result["client_oriented"] = {
            "enabled": True,
            "per_client": [],
            "overall_client_macro": _summarize_client_metrics([]),
            "overall_sample_level": evaluate_membership_scores(
                np.asarray([], dtype=np.float64),
                np.asarray([], dtype=np.float64),
                fixed_threshold=0.5,
            ),
            "forgotten_clients": _summarize_client_metrics([]),
            "retained_clients": _summarize_client_metrics([]),
            "forgotten_minus_retained_gap": {},
        }
        return result

    forgotten_set = set(int(cid) for cid in (forgotten_client_ids or [client.id]))
    if retained_client_ids is None:
        retained_set = set(int(c.id) for c in eval_clients if int(c.id) not in forgotten_set)
    else:
        retained_set = set(int(cid) for cid in retained_client_ids)

    per_client_metrics = []
    all_member_scores = []
    all_nonmember_scores = []

    # Reuse the single-client MIA flow so old and new paths stay consistent.
    for eval_client in eval_clients:
        if mode == "shadow_lr":
            single_reference = build_membership_reference(
                eval_client,
                max_samples=max_samples,
                nonmember_source_mode="target_train_vs_target_test",
            )
            single_result, member_scores, nonmember_scores = _run_shadow_lr_membership_attack(
                model,
                eval_client,
                device,
                all_clients=eval_clients,
                reference_data=single_reference,
                max_samples=max_samples,
                batch_size=batch_size,
                mia_reference_mode=mia_reference_mode,
            )
        elif mode == "target_lr":
            single_reference = build_membership_reference(
                eval_client,
                max_samples=max_samples,
                nonmember_source_mode="target_train_vs_target_test",
            )
            single_result, member_scores, nonmember_scores = _run_target_lr_membership_attack(
                model,
                eval_client,
                device,
                reference_data=single_reference,
                max_samples=max_samples,
                batch_size=batch_size,
            )
        else:
            single_reference = build_membership_reference(
                eval_client,
                max_samples=max_samples,
                nonmember_source_mode=mia_reference_mode,
                all_clients=eval_clients,
            )
            single_result, member_scores, nonmember_scores = _run_single_membership_attack(
                model,
                eval_client,
                device,
                reference_data=single_reference,
                max_samples=max_samples,
                batch_size=batch_size,
            )
        score_name = single_result.get("score_name", "true_class_confidence")
        primary = single_result["metric_details"].get(
            score_name,
            single_result["metric_details"]["true_class_confidence"],
        )
        per_client_metrics.append({
            "client_id": int(eval_client.id),
            "auc": float(primary["auc"]),
            "best_acc": float(primary["best_acc"]),
            "fixed_acc": float(primary["fixed_acc"]),
            "best_precision": float(primary.get("best_precision", 0.0)),
            "best_recall": float(primary.get("best_recall", 0.0)),
            "best_f1": float(primary.get("best_f1", 0.0)),
            "sample_count_per_split": int(single_result["reference"]["sample_count_per_split"]),
        })
        all_member_scores.append(member_scores)
        all_nonmember_scores.append(nonmember_scores)

    forgotten_metrics = [item for item in per_client_metrics if item["client_id"] in forgotten_set]
    retained_metrics = [item for item in per_client_metrics if item["client_id"] in retained_set]

    if all_member_scores and all_nonmember_scores:
        overall_member = np.concatenate(all_member_scores)
        overall_nonmember = np.concatenate(all_nonmember_scores)
    else:
        overall_member = np.asarray([], dtype=np.float64)
        overall_nonmember = np.asarray([], dtype=np.float64)

    forgotten_summary = _summarize_client_metrics(forgotten_metrics)
    retained_summary = _summarize_client_metrics(retained_metrics)
    gap = {}
    for metric in ("auc", "best_acc", "fixed_acc", "best_precision", "best_recall", "best_f1"):
        key = f"{metric}_mean_gap"
        if forgotten_summary["num_clients"] > 0 and retained_summary["num_clients"] > 0:
            gap[key] = float(forgotten_summary[f"{metric}_mean"] - retained_summary[f"{metric}_mean"])
        else:
            gap[key] = None

    result["client_oriented"] = {
        "enabled": True,
        "per_client": per_client_metrics,
        "overall_client_macro": _summarize_client_metrics(per_client_metrics),
        "overall_sample_level": evaluate_membership_scores(
            overall_member,
            overall_nonmember,
            fixed_threshold=0.5,
        ),
        "forgotten_clients": forgotten_summary,
        "retained_clients": retained_summary,
        "forgotten_minus_retained_gap": gap,
    }
    return result


def _default_backdoor_trigger(
    x,
    y,
    target_label=0,
    patch_size=3,
    patch_value=1.0,
    attack_portion=1.0,
    label_mode="fixed_target",
    num_classes=None,
):
    x = _clone_nested(x)
    y = _clone_nested(y)
    tensor = None
    trigger_applied = False
    if torch.is_tensor(x):
        tensor = x
    elif isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
        tensor = x[0]

    if tensor is not None and tensor.dim() >= 4:
        attack_count = int(round(float(attack_portion) * int(tensor.shape[0])))
        attack_count = max(0, min(int(tensor.shape[0]), attack_count))
        patch = max(1, min(int(patch_size), int(tensor.shape[-2]), int(tensor.shape[-1])))
        if attack_count > 0 and patch > 0:
            tensor[:attack_count, ..., -patch:, -patch:] = patch_value
            trigger_applied = True

    if trigger_applied and torch.is_tensor(y):
        poisoned_y = y.detach().clone()
        if str(label_mode).strip().lower() == "class_shift":
            class_count = int(num_classes) if num_classes is not None else 10
            shift = max(1, class_count // 2)
            poisoned_y[:attack_count] = (poisoned_y[:attack_count] + shift) % class_count
        else:
            poisoned_y[:attack_count] = int(target_label)
    elif trigger_applied:
        poisoned_y = copy.deepcopy(y)
    else:
        poisoned_y = y
    return x, poisoned_y


def _evaluate_client_backdoor_split(model, client, device, trigger_func=None, max_batches=5, data_source="test"):
    model.eval()
    total = 0
    correct = 0

    if str(data_source).strip().lower() == "train":
        eval_data = _load_backdoor_eval_data(client, data_source="train")
        testloader = DataLoader(eval_data, batch_size=client.batch_size, shuffle=False, drop_last=False)
    else:
        testloader = client.load_test_data()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            if trigger_func is not None:
                x, y = trigger_func(x, y)
            x = _move_to_device(x, device)
            y = y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if max_batches is not None and batch_idx >= max_batches:
                break

    acc = correct / max(total, 1)
    return float(acc), int(total)


def _summarize_backdoor_metrics(client_metrics):
    summary = {
        "num_clients": int(len(client_metrics)),
        "client_ids": [int(item["client_id"]) for item in client_metrics],
    }
    if not client_metrics:
        for metric in ("asr", "clean_acc"):
            summary[f"{metric}_mean"] = 0.0
            summary[f"{metric}_std"] = 0.0
            summary[f"{metric}_sample_weighted"] = 0.0
        return summary

    for metric, weight_key in (("asr", "num_attack_samples"), ("clean_acc", "num_clean_samples")):
        values = np.asarray([item[metric] for item in client_metrics], dtype=np.float64)
        weights = np.asarray([item[weight_key] for item in client_metrics], dtype=np.float64)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_sample_weighted"] = (
            float(np.average(values, weights=weights)) if np.sum(weights) > 0 else 0.0
        )
    return summary


def backdoor_attack(
    model,
    client,
    device,
    trigger_func=None,
    max_batches=5,
    client_oriented=False,
    all_clients=None,
    forgotten_client_ids=None,
    retained_client_ids=None,
    return_details=False,
    auto_trigger=False,
    trigger_target_label=0,
    trigger_patch_size=3,
    trigger_patch_value=1.0,
    attack_portion=1.0,
    data_source="test",
    label_mode="fixed_target",
    num_classes=None,
):
    """
    后门攻击评估：默认保持旧接口返回 float（target client 上的 ASR）。
    开启 client_oriented 时，返回包含按 client/分组聚合的详细信息。
    """
    effective_trigger = trigger_func
    if effective_trigger is None and auto_trigger:
        # Keep legacy behavior unless explicitly enabled via config.
        effective_trigger = lambda x, y: _default_backdoor_trigger(
            x,
            y,
            target_label=trigger_target_label,
            patch_size=trigger_patch_size,
            patch_value=trigger_patch_value,
            attack_portion=attack_portion,
            label_mode=label_mode,
            num_classes=num_classes,
        )

    target_asr, target_attack_n = _evaluate_client_backdoor_split(
        model,
        client,
        device,
        trigger_func=effective_trigger,
        max_batches=max_batches,
        data_source=data_source,
    )
    target_clean_acc, target_clean_n = _evaluate_client_backdoor_split(
        model,
        client,
        device,
        trigger_func=None,
        max_batches=max_batches,
        data_source=data_source,
    )

    if not client_oriented and not return_details:
        return target_asr

    details = {
        "target_client_id": int(client.id),
        "asr": float(target_asr),
        "clean_acc": float(target_clean_acc),
        "num_attack_samples": int(target_attack_n),
        "num_clean_samples": int(target_clean_n),
        "trigger_applied": bool(effective_trigger is not None),
        "data_source": str(data_source).strip().lower(),
        "attack_portion": float(attack_portion),
    }

    if not client_oriented:
        return details

    eval_clients = list(all_clients) if all_clients is not None else [client]
    forgotten_set = set(int(cid) for cid in (forgotten_client_ids or [client.id]))
    if retained_client_ids is None:
        retained_set = set(int(c.id) for c in eval_clients if int(c.id) not in forgotten_set)
    else:
        retained_set = set(int(cid) for cid in retained_client_ids)

    per_client = []
    for eval_client in eval_clients:
        asr, attack_n = _evaluate_client_backdoor_split(
            model,
            eval_client,
            device,
            trigger_func=effective_trigger,
            max_batches=max_batches,
            data_source=data_source,
        )
        clean_acc, clean_n = _evaluate_client_backdoor_split(
            model,
            eval_client,
            device,
            trigger_func=None,
            max_batches=max_batches,
            data_source=data_source,
        )
        per_client.append({
            "client_id": int(eval_client.id),
            "asr": float(asr),
            "clean_acc": float(clean_acc),
            "num_attack_samples": int(attack_n),
            "num_clean_samples": int(clean_n),
        })

    forgotten_metrics = [item for item in per_client if item["client_id"] in forgotten_set]
    retained_metrics = [item for item in per_client if item["client_id"] in retained_set]
    forgotten_summary = _summarize_backdoor_metrics(forgotten_metrics)
    retained_summary = _summarize_backdoor_metrics(retained_metrics)

    gap = {}
    for metric in ("asr", "clean_acc"):
        key = f"{metric}_mean_gap"
        if forgotten_summary["num_clients"] > 0 and retained_summary["num_clients"] > 0:
            gap[key] = float(forgotten_summary[f"{metric}_mean"] - retained_summary[f"{metric}_mean"])
        else:
            gap[key] = None

    details["client_oriented"] = {
        "enabled": True,
        "per_client": per_client,
        "global": _summarize_backdoor_metrics(per_client),
        "forgotten_clients": forgotten_summary,
        "retained_clients": retained_summary,
        "forgotten_minus_retained_gap": gap,
    }
    return details
