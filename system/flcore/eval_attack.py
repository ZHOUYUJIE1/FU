import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data


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


def build_membership_reference(client, max_samples=None):
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

    fixed_acc = float(accuracy_score(y_true, (y_score >= fixed_threshold).astype(np.int64)))
    return {
        "auc": auc,
        "best_acc": best_acc,
        "best_thr": best_thr,
        "fixed_thr": float(fixed_threshold),
        "fixed_acc": fixed_acc,
        "flipped_auc": flipped_auc,
    }


def membership_inference_attack(model, client, device, reference_data=None, max_samples=None, batch_size=256):
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
        },
        "sanity_checks": {
            "member_label_is_positive": True,
            "higher_score_means_more_likely_member": True,
            "primary_score_uses_true_label": True,
            "hard_label_correctness_is_auxiliary_only": True,
            "flipped_primary_auc": primary["flipped_auc"],
        },
    }


def backdoor_attack(model, client, device, trigger_func=None, max_batches=5):
    """
    后门攻击评估：测试模型对后门样本的识别能力
    trigger_func: 用于生成带触发器的样本的函数
    返回后门样本准确率
    """
    model.eval()
    total = 0
    correct = 0

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
            if batch_idx >= max_batches:
                break
    acc = correct / max(total, 1)
    return acc
