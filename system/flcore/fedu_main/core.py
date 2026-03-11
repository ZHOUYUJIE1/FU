import copy
import random
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors]) if tensors else torch.zeros(1)


def _iter_model_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _to_device(x, y, device: torch.device):
    if isinstance(x, list):
        x[0] = x[0].to(device)
    else:
        x = x.to(device)
    y = y.to(device)
    return x, y


def _split_target_dataset(
    client,
    erased_ratio: float,
    split_seed: int,
    erased_max_samples: int,
) -> Tuple[Subset, Subset]:
    base_loader = client.load_train_data(batch_size=1)
    base_dataset = base_loader.dataset
    n = len(base_dataset)
    if n < 2:
        return Subset(base_dataset, list(range(n))), Subset(base_dataset, list(range(n)))

    n_erased = max(1, int(n * erased_ratio))
    if erased_max_samples > 0:
        n_erased = min(n_erased, erased_max_samples)
    n_erased = min(n_erased, n - 1)

    rng = random.Random(split_seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    erased_idx = all_idx[:n_erased]
    remain_idx = all_idx[n_erased:]
    return Subset(base_dataset, erased_idx), Subset(base_dataset, remain_idx)


def _batch_ce_loss(model: torch.nn.Module, batch, device: torch.device) -> torch.Tensor:
    x, y = batch
    x, y = _to_device(x, y, device)
    logits = model(x)
    return F.cross_entropy(logits, y)


def _sample_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def _hvp(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    vec: torch.Tensor,
    retain_graph: bool,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    gv = torch.zeros((), device=grads[0].device)
    offset = 0
    for g, p in zip(grads, params):
        numel = p.numel()
        v = vec[offset:offset + numel].reshape_as(p)
        gv = gv + torch.sum(g * v)
        offset += numel
    hv = torch.autograd.grad(gv, params, create_graph=False, retain_graph=retain_graph)
    return _flatten_tensors(hv)


def _hutchinson_diag(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    num_samples: int,
    eps: float,
) -> torch.Tensor:
    diag = None
    total_dim = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype
    for sample_idx in range(num_samples):
        z = torch.empty(total_dim, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        keep_graph = sample_idx < (num_samples - 1)
        hz = _hvp(loss, params, z, retain_graph=keep_graph)
        sample = z * hz
        if diag is None:
            diag = sample
        else:
            diag += sample
    diag = diag / float(max(1, num_samples))
    return torch.clamp(diag.abs(), min=eps)


def _min_norm_alpha(g1: torch.Tensor, g2: torch.Tensor) -> float:
    # Two-task min-norm closed-form (FedU-style dual-objective tradeoff).
    diff = g1 - g2
    denom = torch.dot(diff, diff).item()
    if denom <= 1e-18:
        return 0.5
    alpha = torch.dot(g2, g2 - g1).item() / denom
    return float(max(0.0, min(1.0, alpha)))


def _apply_update(model: torch.nn.Module, update_vec: torch.Tensor, lr: float):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.add_(-lr * update_vec[offset:offset + numel].reshape_as(p))
            offset += numel


def fedu_main_forget_client(global_model, clients, args, target_client_id=0):
    print("\n============= FedU-main 风格遗忘方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")

    target_client = clients[target_client_id]
    target_client.set_parameters(global_model)
    updated_model = copy.deepcopy(global_model).to(target_client.device)
    updated_model.train()
    params = _iter_model_params(updated_model)
    if not params:
        print("警告：模型无可训练参数，返回原模型")
        return global_model

    erased_ratio = float(getattr(args, "fedu_erased_ratio", 0.2))
    split_seed = int(getattr(args, "fedu_split_seed", 42))
    erased_max_samples = int(getattr(args, "fedu_erased_max_samples", 0))
    forget_rounds = int(getattr(args, "fedu_forget_rounds", 5))
    local_steps = int(getattr(args, "fedu_local_steps", 3))
    base_lr = float(getattr(args, "fedu_lr", 0.001))
    eps = float(getattr(args, "fedu_eps", 1e-3))
    hutchinson_samples = int(getattr(args, "fedu_hutchinson_samples", 4))
    utility_scale = float(getattr(args, "fedu_utility_scale", 1.0))

    erased_set, remain_set = _split_target_dataset(
        client=target_client,
        erased_ratio=erased_ratio,
        split_seed=split_seed,
        erased_max_samples=erased_max_samples,
    )
    if len(erased_set) == 0 or len(remain_set) == 0:
        print("警告：erased/remaining 划分失败，返回原模型")
        return global_model

    print(f"本地划分: erased={len(erased_set)}, remaining={len(remain_set)}")
    erased_loader = DataLoader(erased_set, batch_size=target_client.batch_size, shuffle=True, drop_last=False)
    remain_loader = DataLoader(remain_set, batch_size=target_client.batch_size, shuffle=True, drop_last=False)

    for r in range(forget_rounds):
        erased_iter = iter(erased_loader)
        remain_iter = iter(remain_loader)
        round_norm = 0.0
        round_alpha = 0.0

        for _ in range(local_steps):
            erased_batch, erased_iter = _sample_batch(erased_iter, erased_loader)
            remain_batch, remain_iter = _sample_batch(remain_iter, remain_loader)

            # IAF term: maximize CE on erased data -> minimize (-CE).
            loss_erase = _batch_ce_loss(updated_model, erased_batch, target_client.device)
            loss_iaf = -loss_erase
            g_iaf = _flatten_tensors(
                torch.autograd.grad(loss_iaf, params, create_graph=False, retain_graph=True)
            )

            # UP term: minimize CE on remaining data.
            # 先基于一张独立计算图估计Hessian，再重新前向一次计算g_up，
            # 避免在同一张图上多次autograd.grad导致图已释放错误。
            loss_up_hessian = _batch_ce_loss(updated_model, remain_batch, target_client.device)
            h_diag = _hutchinson_diag(loss_up_hessian, params, hutchinson_samples, eps)
            loss_up_grad = _batch_ce_loss(updated_model, remain_batch, target_client.device)
            g_up = _flatten_tensors(
                torch.autograd.grad(loss_up_grad, params, create_graph=False, retain_graph=False)
            )
            iaf_precond = g_iaf / h_diag

            # Dual objective tradeoff (two-task min-norm combination).
            alpha = _min_norm_alpha(iaf_precond, utility_scale * g_up)
            direction = alpha * iaf_precond + (1.0 - alpha) * utility_scale * g_up

            lr = base_lr * (1.0 + 0.1 * r)
            _apply_update(updated_model, direction.detach(), lr=lr)
            round_norm += direction.norm().item()
            round_alpha += alpha

        avg_alpha = round_alpha / float(max(1, local_steps))
        avg_norm = round_norm / float(max(1, local_steps))
        if r < forget_rounds - 1:
            print(f"  轮次 {r + 1}/{forget_rounds}: 平均更新范数 {avg_norm:.6f}, 平均alpha {avg_alpha:.4f}")

    print("FedU-main 风格遗忘完成")
    return updated_model
