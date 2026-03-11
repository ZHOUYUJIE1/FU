import copy

import numpy as np
import torch


def _cpu_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _local_only_prefixes(model):
    return tuple(getattr(model, "local_only_param_prefixes", ()))


def _flatten_model(model):
    prefixes = _local_only_prefixes(model)
    flat_params = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        flat_params.append(param.detach().float().cpu().reshape(-1))
    if not flat_params:
        return torch.zeros(1)
    return torch.cat(flat_params, dim=0)


def _psi_star(sigma, epsilon, num_clients):
    delta = 1.0 / max(1, num_clients)
    c = np.sqrt(2.0 * np.log(1.25 / delta))
    return float(epsilon) * float(sigma) / float(c)


def _clone_client_buffer_states(server):
    return {
        client.id: server._extract_buffer_state(client.model)
        for client in server.clients
    }


def _restore_client_buffer_states(server, buffer_states, device):
    if not buffer_states:
        return 0

    restored = 0
    for client in server.clients:
        state = buffer_states.get(client.id)
        if not state:
            continue

        model_state = client.model.state_dict()
        for key, tensor in state.items():
            if key in model_state:
                model_state[key] = tensor.to(device)
        client.model.load_state_dict(model_state, strict=False)
        restored += 1

    return restored


def _uploaded_probability_map(server):
    uploaded_ids = list(getattr(server, "uploaded_ids", []) or [])
    if not uploaded_ids:
        return {}

    uploaded_weights = list(getattr(server, "uploaded_weights", []) or [])
    if len(uploaded_weights) == len(uploaded_ids):
        return {
            int(client_id): float(weight)
            for client_id, weight in zip(uploaded_ids, uploaded_weights)
        }

    uniform_weight = 1.0 / max(1, len(uploaded_ids))
    return {int(client_id): uniform_weight for client_id in uploaded_ids}


def _renormalize_weights(weight_map, uploaded_ids):
    weights = [float(weight_map.get(int(client_id), 0.0)) for client_id in uploaded_ids]
    total = sum(weights)
    if total <= 0:
        uniform = 1.0 / max(1, len(uploaded_ids))
        return [uniform for _ in uploaded_ids]
    return [weight / total for weight in weights]


def _round_history_record(server, round_idx):
    return {
        "round": int(round_idx),
        "uploaded_ids": [int(client_id) for client_id in getattr(server, "uploaded_ids", [])],
        "uploaded_weights": [float(weight) for weight in getattr(server, "uploaded_weights", [])],
    }


def _remaining_round_records(tracker, snapshot_round):
    return [
        record
        for record in getattr(tracker, "round_history", [])
        if int(record.get("round", -1)) > int(snapshot_round)
    ]


def _resolve_recovery_rounds(args, tracker, snapshot_round):
    configured_rounds = int(getattr(args, "sifu_recovery_rounds", -1))
    round_history = getattr(tracker, "round_history", []) or []
    if round_history:
        remaining_rounds = len(_remaining_round_records(tracker, snapshot_round))
    else:
        remaining_rounds = max(0, int(getattr(tracker, "round_idx", -1)) - int(snapshot_round))
    if configured_rounds < 0:
        return remaining_rounds, configured_rounds
    return min(configured_rounds, remaining_rounds), configured_rounds


def _evaluate_global_model(server):
    accs = []
    aucs = []

    for client in server.clients:
        client.set_parameters(server.global_model)
        test_acc, test_num, auc = client.test_metrics()
        if test_num <= 0:
            accs.append(0.0)
            aucs.append(0.0)
            continue
        accs.append(float(test_acc) / float(test_num))
        aucs.append(float(auc))

    avg_acc = float(np.mean(accs)) if accs else 0.0
    avg_auc = float(np.mean(aucs)) if aucs else 0.0
    return avg_acc, avg_auc


def _run_recovery_rounds(server, round_records, target_client_id):
    if server is None:
        return [], None, 0

    clients_by_id = {int(client.id): client for client in server.clients}
    recovery_results = []
    executed_rounds = 0

    for local_round_idx, record in enumerate(round_records, start=1):
        original_ids = [int(client_id) for client_id in record.get("uploaded_ids", [])]
        scheduled_ids = [
            client_id
            for client_id in original_ids
            if client_id != int(target_client_id) and client_id in clients_by_id
        ]
        if not scheduled_ids:
            continue

        server.send_models()
        selected_clients = [clients_by_id[client_id] for client_id in scheduled_ids]
        for client in selected_clients:
            client.train()

        weight_map = {
            int(client_id): float(weight)
            for client_id, weight in zip(
                original_ids,
                record.get("uploaded_weights", []),
            )
        }

        server.selected_clients = selected_clients
        server.uploaded_ids = [int(client.id) for client in selected_clients]
        server.uploaded_models = [client.model for client in selected_clients]
        server.uploaded_weights = _renormalize_weights(weight_map, server.uploaded_ids)
        server.aggregate_parameters()

        avg_acc, avg_auc = _evaluate_global_model(server)
        recovery_results.append({
            "round": int(local_round_idx),
            "scheduled_round": int(record.get("round", -1)),
            "test_acc": avg_acc,
            "test_auc": avg_auc,
        })
        executed_rounds += 1

    post_recovery_client_buffer_states = _clone_client_buffer_states(server)
    return recovery_results, post_recovery_client_buffer_states, executed_rounds


class SIFUTracker:
    def __init__(self, args):
        self.args = args
        self.num_clients = int(args.num_clients)
        self.lr_g = float(
            getattr(
                args,
                "sifu_lr_g",
                getattr(args, "server_learning_rate", 1.0),
            )
        )
        self.lr_l = float(getattr(args, "local_learning_rate", 0.01))
        self.local_steps = max(1, int(getattr(args, "local_epochs", 1)))
        self.lambd = float(getattr(args, "sifu_lambd", 0.0))
        self.epsilon = float(getattr(args, "sifu_epsilon", 10.0))
        self.sigma = float(getattr(args, "sifu_sigma", 0.05))
        self.psi_star = _psi_star(self.sigma, self.epsilon, self.num_clients)
        self.metric = np.zeros(self.num_clients, dtype=np.float64)
        self.best_snapshots = {}
        self.round_idx = -1
        self.round_history = []

    def update(self, server):
        if not getattr(server, "uploaded_ids", None):
            return

        self.round_idx += 1
        coef_regu = max(0.0, 1.0 - self.lr_l * self.lambd) ** self.local_steps
        self.metric *= coef_regu

        vec_g = _flatten_model(server.global_model)
        p_kept = _uploaded_probability_map(server)

        for client_id, local_model in zip(server.uploaded_ids, server.uploaded_models):
            p_i = float(p_kept.get(client_id, 0.0))
            if p_i >= 1.0:
                continue
            vec_l = _flatten_model(local_model)
            coef_client_removal = self.lr_g * p_i / max(1e-12, 1.0 - p_i)
            self.metric[client_id] += coef_client_removal * torch.norm(vec_g - vec_l, p=2).item()

        snapshot = {
            "round": int(self.round_idx),
            "model_state_dict": _cpu_state_dict(server.global_model.state_dict()),
            "client_buffer_states": _clone_client_buffer_states(server),
        }
        self.round_history.append(_round_history_record(server, self.round_idx))

        for client_id in range(self.num_clients):
            if self.metric[client_id] <= self.psi_star:
                self.best_snapshots[client_id] = {
                    "metric": float(self.metric[client_id]),
                    **snapshot,
                }

    def state_dict(self):
        return {
            "num_clients": self.num_clients,
            "lr_g": self.lr_g,
            "lr_l": self.lr_l,
            "local_steps": self.local_steps,
            "lambd": self.lambd,
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "psi_star": self.psi_star,
            "metric": self.metric.tolist(),
            "round_idx": self.round_idx,
            "best_snapshots": self.best_snapshots,
            "round_history": self.round_history,
        }

    @classmethod
    def from_state_dict(cls, args, state):
        tracker = cls(args)
        tracker.num_clients = int(state.get("num_clients", tracker.num_clients))
        tracker.lr_g = float(state.get("lr_g", tracker.lr_g))
        tracker.lr_l = float(state.get("lr_l", tracker.lr_l))
        tracker.local_steps = int(state.get("local_steps", tracker.local_steps))
        tracker.lambd = float(state.get("lambd", tracker.lambd))
        tracker.epsilon = float(state.get("epsilon", tracker.epsilon))
        tracker.sigma = float(state.get("sigma", tracker.sigma))
        tracker.psi_star = float(state.get("psi_star", tracker.psi_star))
        tracker.metric = np.array(state.get("metric", tracker.metric.tolist()), dtype=np.float64)
        tracker.round_idx = int(state.get("round_idx", tracker.round_idx))
        tracker.best_snapshots = state.get("best_snapshots", {})
        tracker.round_history = state.get("round_history", [])
        return tracker


def _add_model_noise(model, sigma):
    if sigma <= 0:
        return
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.normal(0.0, sigma, size=param.shape, device=param.device))


def forget_client_with_sifu(global_model, clients, args, target_client_id=0, server=None):
    if getattr(args, "algorithm", "") != "FedAvg":
        raise ValueError("SIFU 对比实现当前仅支持 FedAvg，以保持与论文设定更一致。")

    tracker = getattr(server, "sifu_tracker", None)
    if tracker is None:
        raise ValueError("SIFU requires training-time history. Please train with forget_strategy=sifu first.")

    snapshot = tracker.best_snapshots.get(target_client_id)
    if snapshot is None:
        raise ValueError(
            f"SIFU未找到客户端 {target_client_id} 的可回跳快照。"
            "请增大 sifu_epsilon、减小 sifu_sigma，或重新训练保存历史。"
        )

    print("\n============= SIFU 对比方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")
    print(
        f"SIFU回跳轮次: {snapshot['round']}, "
        f"psi={snapshot['metric']:.6f}, psi*={tracker.psi_star:.6f}"
    )

    forgotten_model = copy.deepcopy(global_model)
    forgotten_model.load_state_dict(snapshot["model_state_dict"], strict=False)

    snapshot_buffer_states = snapshot.get("client_buffer_states")
    recovery_rounds, configured_recovery_rounds = _resolve_recovery_rounds(
        args,
        tracker,
        snapshot["round"],
    )
    skip_recovery = bool(getattr(args, "sifu_skip_recovery", False))
    recovery_mode = "scheduled_replay"

    post_recovery_model = None
    post_recovery_client_buffer_states = None
    recovery_results = []
    recovery_stage = "skipped_for_sifu"

    if (
        server is not None
        and recovery_rounds > 0
        and not skip_recovery
        and getattr(tracker, "round_history", None)
    ):
        restored = _restore_client_buffer_states(server, snapshot_buffer_states, args.device)
        if restored:
            print(f"SIFU已恢复回跳快照对应的客户端buffer状态: {restored} 个客户端")

        server.global_model = copy.deepcopy(forgotten_model)
        scheduled_round_records = _remaining_round_records(tracker, snapshot["round"])[:recovery_rounds]
        recovery_results, post_recovery_client_buffer_states, executed_rounds = _run_recovery_rounds(
            server,
            scheduled_round_records,
            target_client_id=target_client_id,
        )
        recovery_rounds = int(executed_rounds)
        post_recovery_model = copy.deepcopy(server.global_model)
        recovery_stage = "internal_sifu_recovery"
        _add_model_noise(post_recovery_model, tracker.sigma)
    elif not skip_recovery and recovery_rounds > 0:
        restored = _restore_client_buffer_states(server, snapshot_buffer_states, args.device)
        if restored:
            print(f"SIFU已恢复回跳快照对应的客户端buffer状态: {restored} 个客户端")
        print("SIFU未找到可回放的原始轮次历史，将退化为普通恢复轮数配置。")
        if server is not None and hasattr(server, "recovery_training"):
            server.global_model = copy.deepcopy(forgotten_model)
            recovery_results = server.recovery_training(
                target_client_id=target_client_id,
                recovery_rounds=recovery_rounds,
            )
            post_recovery_model = copy.deepcopy(server.global_model)
            post_recovery_client_buffer_states = _clone_client_buffer_states(server)
            recovery_stage = "fallback_recovery"
            recovery_mode = "stochastic_fallback"
            _add_model_noise(post_recovery_model, tracker.sigma)
    else:
        _add_model_noise(forgotten_model, tracker.sigma)

    return {
        "model": forgotten_model,
        "client_buffer_states": snapshot_buffer_states,
        "post_recovery_model": post_recovery_model,
        "post_recovery_client_buffer_states": post_recovery_client_buffer_states,
        "recovery_results": recovery_results,
        "recovery_rounds": recovery_rounds if not skip_recovery else 0,
        "recovery_stage": recovery_stage,
        "metadata": {
            "sifu_round": int(snapshot["round"]),
            "sifu_metric": float(snapshot["metric"]),
            "sifu_psi_star": float(tracker.psi_star),
            "sifu_total_rounds": int(tracker.round_idx),
            "sifu_recovery_mode": recovery_mode,
            "sifu_requested_recovery_rounds": int(configured_recovery_rounds),
            "sifu_recovery_rounds": recovery_rounds if not skip_recovery else 0,
        },
    }
