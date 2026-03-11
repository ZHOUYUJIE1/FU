import copy
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer


def _cpu_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _weighted_average_state_dicts(state_dicts, weights):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged.keys():
        tensor = averaged[key]
        if torch.is_floating_point(tensor):
            acc = torch.zeros_like(tensor, dtype=tensor.dtype)
            for state_dict, weight in zip(state_dicts, weights):
                acc += state_dict[key].to(acc.dtype) * (float(weight) / total_weight)
            averaged[key] = acc
        else:
            averaged[key] = state_dicts[0][key].clone()
    return averaged


def _state_fingerprint(state_dict):
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


class FedCSAComparison:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.min_clusters = getattr(args, "fedcsa_compare_min_clusters", 2)
        self.max_clusters = getattr(args, "fedcsa_compare_max_clusters", 5)
        self.probe_epochs = getattr(args, "fedcsa_probe_epochs", 1)
        self.probe_batches = getattr(
            args, "fedcsa_probe_batches", max(1, min(getattr(args, "max_batches", 4), 4))
        )
        self.slice_rounds = getattr(args, "fedcsa_slice_rounds", max(1, args.local_epochs))
        self.slice_local_epochs = getattr(args, "fedcsa_slice_local_epochs", max(1, args.local_epochs))
        self.slice_batches = getattr(args, "fedcsa_slice_batches", 0)
        self.forget_rounds = getattr(args, "fedcsa_forget_epochs", self.slice_rounds)
        self.use_cache = getattr(args, "fedcsa_cache_enabled", True)
        self.force_rebuild = getattr(args, "fedcsa_force_rebuild", False)
        self.use_scaffold_controls = getattr(args, "fedcsa_compare_use_scaffold", True)

    def forget_client(self, global_model, clients, target_client_id=0):
        print("\n============= FedCSA 对比方法 =============")
        print(f"目标遗忘客户端: {target_client_id}")

        cache = self._load_or_prepare_cache(global_model, clients)
        slice_members = cache["slice_members"]
        slice_model_states = cache["slice_model_states"]

        target_slice_id = None
        for slice_id, members in enumerate(slice_members):
            if target_client_id in members:
                target_slice_id = slice_id
                break

        if target_slice_id is None:
            raise RuntimeError(f"未找到目标客户端 {target_client_id} 所属的切片")

        print(f"目标客户端所属切片: {target_slice_id}, 成员: {slice_members[target_slice_id]}")

        retained_states = []
        retained_weights = []

        for slice_id, members in enumerate(slice_members):
            if slice_id == target_slice_id:
                continue
            retained_states.append(slice_model_states[slice_id])
            retained_weights.append(self._count_slice_samples(clients, members))

        target_members_after_forget = [cid for cid in slice_members[target_slice_id] if cid != target_client_id]
        if target_members_after_forget:
            print(f"仅重训练目标切片 {target_slice_id}，移除客户端 {target_client_id}")
            target_state = self._train_slice_model(
                global_model,
                clients,
                target_members_after_forget,
                rounds=self.forget_rounds,
            )
            retained_states.append(target_state)
            retained_weights.append(self._count_slice_samples(clients, target_members_after_forget))
        else:
            print("目标切片仅包含待遗忘客户端，将直接移除该切片")

        if not retained_states:
            print("警告：遗忘后没有剩余切片，返回原始模型")
            return copy.deepcopy(global_model)

        forgotten_state = _weighted_average_state_dicts(retained_states, retained_weights)
        forgotten_model = copy.deepcopy(global_model)
        forgotten_model.load_state_dict(forgotten_state, strict=True)

        print("FedCSA 对比方法遗忘完成")
        return forgotten_model

    def _load_or_prepare_cache(self, global_model, clients):
        cache_path = self._cache_path()
        current_fingerprint = _state_fingerprint(global_model.state_dict())

        if self.use_cache and not self.force_rebuild and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu")
            if self._cache_matches(cache, current_fingerprint, len(clients)):
                print(f"加载 FedCSA 切片缓存: {cache_path}")
                return cache
            print("检测到 FedCSA 缓存与当前模型或配置不匹配，重新构建")

        print("构建 FedCSA 切片缓存...")
        cache = self._prepare_cache(global_model, clients, current_fingerprint)
        if self.use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(cache, cache_path)
            print(f"FedCSA 切片缓存已保存到: {cache_path}")
        return cache

    def _prepare_cache(self, global_model, clients, fingerprint):
        features, client_metrics = self._extract_probe_features(global_model, clients)
        cluster_labels = self._cluster_clients(features)
        slice_members = self._cluster_labels_to_slices(cluster_labels)

        print(f"FedCSA 切片数量: {len(slice_members)}")
        for slice_id, members in enumerate(slice_members):
            print(f"  切片 {slice_id}: 客户端 {members}")

        slice_model_states = []
        slice_weights = []
        for slice_id, members in enumerate(slice_members):
            print(f"预训练切片 {slice_id} 子模型...")
            state = self._train_slice_model(global_model, clients, members, rounds=self.slice_rounds)
            slice_model_states.append(state)
            slice_weights.append(self._count_slice_samples(clients, members))

        return {
            "version": 1,
            "fingerprint": fingerprint,
            "num_clients": len(clients),
            "config": {
                "probe_epochs": self.probe_epochs,
                "probe_batches": self.probe_batches,
                "slice_rounds": self.slice_rounds,
                "slice_local_epochs": self.slice_local_epochs,
                "slice_batches": self.slice_batches,
                "forget_rounds": self.forget_rounds,
                "min_clusters": self.min_clusters,
                "max_clusters": self.max_clusters,
                "use_scaffold_controls": self.use_scaffold_controls,
            },
            "features": features.tolist(),
            "client_metrics": client_metrics,
            "cluster_labels": cluster_labels.tolist(),
            "slice_members": slice_members,
            "slice_model_states": slice_model_states,
            "slice_weights": slice_weights,
        }

    def _extract_probe_features(self, global_model, clients):
        base_vec = self._flatten_model_parameters(global_model)
        features = []
        metrics = []

        for client in clients:
            local_model = self._train_client_from_base(
                global_model,
                client,
                local_epochs=self.probe_epochs,
                max_batches=self.probe_batches,
            )
            local_vec = self._flatten_model_parameters(local_model)
            cosine = F.cosine_similarity(local_vec.unsqueeze(0), base_vec.unsqueeze(0)).item()
            cosine = max(-1.0, min(1.0, float(cosine)))
            drift = torch.norm(local_vec - base_vec).item() / (torch.norm(base_vec).item() + 1e-12)
            features.append([cosine, drift])
            metrics.append(
                {
                    "client_id": client.id,
                    "cosine_similarity": float(cosine),
                    "relative_drift": float(drift),
                }
            )
            print(
                f"客户端 {client.id} 探测特征: cosine={cosine:.6f}, "
                f"relative_drift={drift:.6f}"
            )

        return np.array(features, dtype=np.float32), metrics

    def _cluster_clients(self, features):
        num_clients = len(features)
        if num_clients == 1:
            return np.zeros(1, dtype=np.int64)

        upper_k = min(self.max_clusters, num_clients)
        lower_k = min(max(2, self.min_clusters), upper_k)

        if upper_k < 2:
            return np.zeros(num_clients, dtype=np.int64)

        best_labels = None
        best_k = lower_k
        best_score = -float("inf")

        for k in range(lower_k, upper_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.args.random_seed, n_init=10)
            labels = kmeans.fit_predict(features)
            if len(np.unique(labels)) < 2:
                continue
            score = calinski_harabasz_score(features, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            kmeans = KMeans(n_clusters=lower_k, random_state=self.args.random_seed, n_init=10)
            best_labels = kmeans.fit_predict(features)

        print(f"基于 CH 指数选择 {best_k} 个簇")
        return best_labels.astype(np.int64)

    def _cluster_labels_to_slices(self, cluster_labels):
        slices = {}
        for client_id, label in enumerate(cluster_labels):
            slices.setdefault(int(label), []).append(client_id)

        ordered = []
        for label in sorted(slices.keys()):
            ordered.append(sorted(slices[label]))
        return ordered

    def _train_slice_model(self, global_model, clients, client_ids, rounds):
        slice_model = copy.deepcopy(global_model)
        if not client_ids:
            return _cpu_state_dict(slice_model.state_dict())

        for round_idx in range(rounds):
            local_states = []
            local_weights = []
            print(f"  切片训练轮次 {round_idx + 1}/{rounds}, 客户端 {client_ids}")
            for client_id in client_ids:
                local_model = self._train_client_from_base(
                    slice_model,
                    clients[client_id],
                    local_epochs=self.slice_local_epochs,
                    max_batches=self.slice_batches,
                )
                local_states.append(_cpu_state_dict(local_model.state_dict()))
                local_weights.append(clients[client_id].train_samples)

            aggregated_state = _weighted_average_state_dicts(local_states, local_weights)
            slice_model.load_state_dict(aggregated_state, strict=True)

        return _cpu_state_dict(slice_model.state_dict())

    def _train_client_from_base(self, base_model, client, local_epochs, max_batches):
        local_model = copy.deepcopy(base_model).to(self.device)
        local_model.train()

        use_scaffold = (
            self.use_scaffold_controls
            and getattr(self.args, "algorithm", "").upper() == "SCAFFOLD"
            and hasattr(client, "client_c")
            and client.client_c is not None
        )

        if use_scaffold:
            server_cs = getattr(client, "global_c", None)
            if server_cs is None:
                server_cs = [torch.zeros_like(param, device=self.device) for param in local_model.parameters()]
            else:
                server_cs = [tensor.detach().clone().to(self.device) for tensor in server_cs]
            client_cs = [tensor.detach().clone().to(self.device) for tensor in client.client_c]
            optimizer = SCAFFOLDOptimizer(local_model.parameters(), lr=client.learning_rate)
        else:
            optimizer = torch.optim.SGD(local_model.parameters(), lr=client.learning_rate)
            server_cs = None
            client_cs = None

        trainloader = client.load_train_data()
        for _ in range(local_epochs):
            batch_count = 0
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = local_model(x)
                loss = client.loss(output, y)
                loss.backward()

                if use_scaffold:
                    optimizer.step(server_cs, client_cs)
                else:
                    optimizer.step()

                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    break

        return local_model.cpu()

    def _flatten_model_parameters(self, model):
        flat_params = []
        for _, param in model.named_parameters():
            flat_params.append(param.detach().float().cpu().reshape(-1))
        return torch.cat(flat_params, dim=0)

    def _count_slice_samples(self, clients, client_ids):
        return sum(clients[client_id].train_samples for client_id in client_ids)

    def _cache_matches(self, cache, fingerprint, num_clients):
        if cache.get("version") != 1:
            return False
        if cache.get("fingerprint") != fingerprint:
            return False
        if cache.get("num_clients") != num_clients:
            return False

        config = cache.get("config", {})
        expected = {
            "probe_epochs": self.probe_epochs,
            "probe_batches": self.probe_batches,
            "slice_rounds": self.slice_rounds,
            "slice_local_epochs": self.slice_local_epochs,
            "slice_batches": self.slice_batches,
            "forget_rounds": self.forget_rounds,
            "min_clusters": self.min_clusters,
            "max_clusters": self.max_clusters,
            "use_scaffold_controls": self.use_scaffold_controls,
        }
        return config == expected

    def _cache_path(self):
        base_name = f"{self.args.dataset}_{self.args.algorithm}_{self.args.goal}_{self.args.times}"
        if getattr(self.args, "load_saved_model", False) and getattr(self.args, "saved_model_path", ""):
            checkpoint_name = os.path.splitext(os.path.basename(self.args.saved_model_path))[0]
            base_name = f"{base_name}_{checkpoint_name}"
        return os.path.join("results", "exp_configs", f"fedcsa_compare_cache_{base_name}.pt")


def forget_client_with_fedcsa_compare(global_model, clients, args, target_client_id=0):
    method = FedCSAComparison(args)
    return method.forget_client(global_model, clients, target_client_id=target_client_id)
