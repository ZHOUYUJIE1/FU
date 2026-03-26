import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def bytes_to_mib(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


def summarize_tensor_bytes(obj):
    if torch.is_tensor(obj):
        return int(obj.numel()) * int(obj.element_size())

    if isinstance(obj, dict):
        total = 0
        for value in obj.values():
            total += summarize_tensor_bytes(value)
        return total

    if isinstance(obj, (list, tuple)):
        total = 0
        for value in obj:
            total += summarize_tensor_bytes(value)
        return total

    return 0


def infer_model_payload_bytes(model_artifact):
    loaded = torch.load(model_artifact, map_location="cpu")
    return int(summarize_tensor_bytes(loaded))


def load_client_train_sizes(dataset_dir, num_clients):
    train_dir = Path(dataset_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"dataset train dir not found: {train_dir}")

    sizes = []
    for client_id in range(num_clients):
        client_path = train_dir / f"{client_id}.npz"
        if not client_path.exists():
            raise FileNotFoundError(f"client file not found: {client_path}")
        payload = np.load(client_path, allow_pickle=True)["data"].tolist()
        sizes.append(int(len(payload["x"])))
    return sizes


def train_batches_per_client(train_sizes, batch_size, drop_last=True):
    if drop_last:
        return [size // batch_size for size in train_sizes]
    return [math.ceil(size / batch_size) for size in train_sizes]


def aggregate_training_batches(client_batch_counts, client_ids, local_epochs):
    return int(sum(client_batch_counts[cid] for cid in client_ids) * local_epochs)


def build_payload_summary(download_payloads, upload_payloads, model_payload_bytes, extra_upload_bytes=0):
    downlink_bytes = int(download_payloads * model_payload_bytes)
    uplink_bytes = int(upload_payloads * model_payload_bytes + extra_upload_bytes)
    total_bytes = int(downlink_bytes + uplink_bytes)
    return {
        "downloads_equiv": int(download_payloads),
        "uploads_equiv": int(upload_payloads),
        "downlink_bytes": downlink_bytes,
        "uplink_bytes": uplink_bytes,
        "total_bytes": total_bytes,
        "downlink_mib": bytes_to_mib(downlink_bytes),
        "uplink_mib": bytes_to_mib(uplink_bytes),
        "total_mib": bytes_to_mib(total_bytes),
        "extra_upload_bytes": int(extra_upload_bytes),
        "extra_upload_mib": bytes_to_mib(extra_upload_bytes),
    }


def estimate_fu(
    num_clients,
    target_client_id,
    train_batch_counts,
    model_payload_bytes,
    *,
    epochs,
    feature_probe_batches,
    mask_batches,
    precompute_batches,
    train_gradient_batches,
    retain_calibration_rounds,
    retain_calibration_batches,
    strict_precompute,
    feature_upload_mode,
    feature_dim,
    feature_batch_size,
):
    retained_ids = [cid for cid in range(num_clients) if cid != target_client_id]
    retained_count = len(retained_ids)

    # Stage 1: client feature probe for clustering.
    feature_forward_batches = int(num_clients * feature_probe_batches)
    initial_downloads = int(num_clients)
    if feature_upload_mode == "raw_feature_matrix":
        feature_upload_bytes = int(num_clients * feature_probe_batches * feature_batch_size * feature_dim * 4)
    elif feature_upload_mode == "feature_norms":
        feature_upload_bytes = int(num_clients * feature_probe_batches * feature_batch_size * 4)
    else:
        feature_upload_bytes = int(num_clients * 10 * 4)

    # Stage 2: target mask generation.
    mask_backward_batches = int(mask_batches)
    mask_gradient_uploads = 1

    # Stage 3: precompute adaptive strengths.
    if strict_precompute:
        precompute_backward_batches = int(epochs * retained_count * 2 * precompute_batches)
        precompute_gradient_uploads = int(epochs * retained_count * 2)
    else:
        precompute_backward_batches = int((1 + retained_count) * precompute_batches)
        precompute_gradient_uploads = int(1 + retained_count)

    # Stage 4: main forgetting epochs.
    main_train_backward_batches = int(num_clients * epochs * train_gradient_batches)
    main_train_downloads = int(num_clients * epochs)
    main_train_uploads = int(num_clients * epochs)

    # Stage 5: retain calibration.
    retain_calibration_backward_batches = int(retain_calibration_rounds * retained_count * retain_calibration_batches)
    retain_calibration_downloads = int(retain_calibration_rounds * retained_count)
    retain_calibration_uploads = int(retain_calibration_rounds * retained_count)

    total_backward_batches = (
        mask_backward_batches
        + precompute_backward_batches
        + main_train_backward_batches
        + retain_calibration_backward_batches
    )

    total_downloads = (
        initial_downloads
        + main_train_downloads
        + retain_calibration_downloads
    )
    total_uploads = (
        mask_gradient_uploads
        + precompute_gradient_uploads
        + main_train_uploads
        + retain_calibration_uploads
    )

    return {
        "assumption": "Strict RPC emulation" if strict_precompute else "Client-side dedup for precompute similarity gradients",
        "compute": {
            "forward_only_batches": int(feature_forward_batches),
            "backward_batches": int(total_backward_batches),
            "round_equivalent_all_clients": float(total_backward_batches / max(1, sum(train_batch_counts))),
            "round_equivalent_retained_clients": float(
                total_backward_batches / max(1, sum(train_batch_counts[cid] for cid in retained_ids))
            ),
            "details": {
                "feature_probe_forward_batches": int(feature_forward_batches),
                "mask_backward_batches": int(mask_backward_batches),
                "precompute_backward_batches": int(precompute_backward_batches),
                "main_train_backward_batches": int(main_train_backward_batches),
                "retain_calibration_backward_batches": int(retain_calibration_backward_batches),
            },
        },
        "communication": build_payload_summary(
            total_downloads,
            total_uploads,
            model_payload_bytes,
            extra_upload_bytes=feature_upload_bytes,
        ),
    }


def estimate_fedcsa(
    num_clients,
    target_client_id,
    train_batch_counts,
    model_payload_bytes,
    *,
    feature_probe_batches,
    forget_epochs,
    local_epochs_per_round,
):
    retained_ids = [cid for cid in range(num_clients) if cid != target_client_id]
    retained_count = len(retained_ids)

    feature_forward_batches = int(num_clients * feature_probe_batches)
    initial_downloads = int(num_clients)
    feature_upload_bytes = int(num_clients * 10 * 4)

    backward_batches = int(sum(train_batch_counts[cid] for cid in retained_ids) * forget_epochs * local_epochs_per_round)
    round_downloads = int(retained_count * forget_epochs)
    round_uploads = int(retained_count * forget_epochs)

    return {
        "assumption": "Feature grouping requires one initial global-model sync; retained clients then run subset-local training once per forget epoch.",
        "compute": {
            "forward_only_batches": int(feature_forward_batches),
            "backward_batches": int(backward_batches),
            "round_equivalent_retained_clients": float(
                backward_batches / max(1, sum(train_batch_counts[cid] for cid in retained_ids))
            ),
            "details": {
                "feature_probe_forward_batches": int(feature_forward_batches),
                "subset_training_backward_batches": int(backward_batches),
            },
        },
        "communication": build_payload_summary(
            initial_downloads + round_downloads,
            round_uploads,
            model_payload_bytes,
            extra_upload_bytes=feature_upload_bytes,
        ),
    }


def estimate_fedosd(
    num_clients,
    target_client_id,
    train_batch_counts,
    model_payload_bytes,
    *,
    unlearn_rounds,
    recovery_rounds,
    local_epochs,
):
    retained_ids = [cid for cid in range(num_clients) if cid != target_client_id]
    retained_count = len(retained_ids)
    all_ids = list(range(num_clients))

    unlearn_backward_batches = int(
        aggregate_training_batches(train_batch_counts, all_ids, local_epochs) * unlearn_rounds
    )
    recovery_backward_batches = int(
        aggregate_training_batches(train_batch_counts, retained_ids, local_epochs) * recovery_rounds
    )

    total_backward_batches = int(unlearn_backward_batches + recovery_backward_batches)
    total_downloads = int(unlearn_rounds * num_clients + recovery_rounds * retained_count)
    total_uploads = int(unlearn_rounds * num_clients + recovery_rounds * retained_count)

    return {
        "assumption": "Official-style FedOSD: each online client downloads the current model, trains locally, and uploads one local model/update per round.",
        "compute": {
            "forward_only_batches": 0,
            "backward_batches": int(total_backward_batches),
            "round_equivalent_all_clients": float(total_backward_batches / max(1, sum(train_batch_counts))),
            "details": {
                "unlearning_backward_batches": int(unlearn_backward_batches),
                "post_training_backward_batches": int(recovery_backward_batches),
            },
        },
        "communication": build_payload_summary(total_downloads, total_uploads, model_payload_bytes),
    }


def estimate_retrain(
    num_clients,
    target_client_id,
    train_batch_counts,
    model_payload_bytes,
    *,
    retrain_rounds,
    local_epochs,
):
    retained_ids = [cid for cid in range(num_clients) if cid != target_client_id]
    retained_count = len(retained_ids)
    backward_batches = int(
        aggregate_training_batches(train_batch_counts, retained_ids, local_epochs) * retrain_rounds
    )
    total_downloads = int(retained_count * retrain_rounds)
    total_uploads = int(retained_count * retrain_rounds)

    return {
        "assumption": "Retrain baseline restarts federated training from scratch and excludes the target client in every round.",
        "compute": {
            "forward_only_batches": 0,
            "backward_batches": int(backward_batches),
            "round_equivalent_retained_clients": float(
                backward_batches / max(1, sum(train_batch_counts[cid] for cid in retained_ids))
            ),
        },
        "communication": build_payload_summary(total_downloads, total_uploads, model_payload_bytes),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate realistic federated forgetting overhead for FU / FedCSA / FedOSD / retrain."
    )
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory, e.g. dataset/Cifar10_hetero_clean_s321_severe")
    parser.add_argument("--model-artifact", required=True, help="Checkpoint artifact used to estimate model payload bytes")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--target-client-id", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--fu-epochs", type=int, default=10)
    parser.add_argument("--fu-feature-probe-batches", type=int, default=11)
    parser.add_argument("--fu-mask-batches", type=int, default=4)
    parser.add_argument("--fu-precompute-batches", type=int, default=6)
    parser.add_argument("--fu-train-gradient-batches", type=int, default=8)
    parser.add_argument("--fu-retain-calibration-rounds", type=int, default=2)
    parser.add_argument("--fu-retain-calibration-batches", type=int, default=3)
    parser.add_argument(
        "--fu-feature-upload-mode",
        choices=["raw_feature_matrix", "feature_norms", "tiny_summary"],
        default="raw_feature_matrix",
    )
    parser.add_argument("--fu-feature-dim", type=int, default=512)
    parser.add_argument("--fu-feature-batch-size", type=int, default=32)

    parser.add_argument("--fedcsa-feature-probe-batches", type=int, default=6)
    parser.add_argument("--fedcsa-forget-epochs", type=int, default=5)
    parser.add_argument("--fedcsa-local-epochs-per-round", type=int, default=2)

    parser.add_argument("--fedosd-unlearn-rounds", type=int, default=20)
    parser.add_argument("--fedosd-recovery-rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)

    parser.add_argument("--retrain-rounds", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact text report")
    return parser.parse_args()


def main():
    args = parse_args()

    model_payload_bytes = infer_model_payload_bytes(args.model_artifact)
    train_sizes = load_client_train_sizes(args.dataset_dir, args.num_clients)
    train_batch_counts = train_batches_per_client(train_sizes, args.batch_size, drop_last=True)

    results = {
        "dataset_dir": args.dataset_dir,
        "model_artifact": args.model_artifact,
        "model_payload_bytes": int(model_payload_bytes),
        "model_payload_mib": bytes_to_mib(model_payload_bytes),
        "train_sizes": train_sizes,
        "train_batch_counts_drop_last": train_batch_counts,
        "methods": {
            "fu_strict_rpc": estimate_fu(
                args.num_clients,
                args.target_client_id,
                train_batch_counts,
                model_payload_bytes,
                epochs=args.fu_epochs,
                feature_probe_batches=args.fu_feature_probe_batches,
                mask_batches=args.fu_mask_batches,
                precompute_batches=args.fu_precompute_batches,
                train_gradient_batches=args.fu_train_gradient_batches,
                retain_calibration_rounds=args.fu_retain_calibration_rounds,
                retain_calibration_batches=args.fu_retain_calibration_batches,
                strict_precompute=True,
                feature_upload_mode=args.fu_feature_upload_mode,
                feature_dim=args.fu_feature_dim,
                feature_batch_size=args.fu_feature_batch_size,
            ),
            "fu_client_dedup": estimate_fu(
                args.num_clients,
                args.target_client_id,
                train_batch_counts,
                model_payload_bytes,
                epochs=args.fu_epochs,
                feature_probe_batches=args.fu_feature_probe_batches,
                mask_batches=args.fu_mask_batches,
                precompute_batches=args.fu_precompute_batches,
                train_gradient_batches=args.fu_train_gradient_batches,
                retain_calibration_rounds=args.fu_retain_calibration_rounds,
                retain_calibration_batches=args.fu_retain_calibration_batches,
                strict_precompute=False,
                feature_upload_mode=args.fu_feature_upload_mode,
                feature_dim=args.fu_feature_dim,
                feature_batch_size=args.fu_feature_batch_size,
            ),
            "fedcsa": estimate_fedcsa(
                args.num_clients,
                args.target_client_id,
                train_batch_counts,
                model_payload_bytes,
                feature_probe_batches=args.fedcsa_feature_probe_batches,
                forget_epochs=args.fedcsa_forget_epochs,
                local_epochs_per_round=args.fedcsa_local_epochs_per_round,
            ),
            "fedosd": estimate_fedosd(
                args.num_clients,
                args.target_client_id,
                train_batch_counts,
                model_payload_bytes,
                unlearn_rounds=args.fedosd_unlearn_rounds,
                recovery_rounds=args.fedosd_recovery_rounds,
                local_epochs=args.local_epochs,
            ),
            "retrain": estimate_retrain(
                args.num_clients,
                args.target_client_id,
                train_batch_counts,
                model_payload_bytes,
                retrain_rounds=args.retrain_rounds,
                local_epochs=args.local_epochs,
            ),
        },
    }

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"Dataset: {args.dataset_dir}")
    print(f"Model payload: {model_payload_bytes} bytes ({bytes_to_mib(model_payload_bytes):.2f} MiB)")
    print(f"Train sizes per client: {train_sizes}")
    print(f"Train batches per client (drop_last=True): {train_batch_counts}")

    for method_name, result in results["methods"].items():
        compute = result["compute"]
        comm = result["communication"]
        print("")
        print(f"[{method_name}]")
        print(f"  backward_batches={compute['backward_batches']}, forward_only_batches={compute['forward_only_batches']}")
        if "round_equivalent_all_clients" in compute:
            print(f"  round_equivalent_all_clients={compute['round_equivalent_all_clients']:.4f}")
        if "round_equivalent_retained_clients" in compute:
            print(f"  round_equivalent_retained_clients={compute['round_equivalent_retained_clients']:.4f}")
        print(
            "  communication="
            f"down {comm['downlink_mib']:.2f} MiB + "
            f"up {comm['uplink_mib']:.2f} MiB = "
            f"{comm['total_mib']:.2f} MiB"
        )


if __name__ == "__main__":
    main()
