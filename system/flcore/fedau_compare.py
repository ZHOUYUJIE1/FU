import copy

import torch


def forget_client_with_fedau(global_model, clients, args, target_client_id=0):
    print("\n============= FedAU 对比方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")

    target_client = clients[target_client_id]
    target_model = target_client.model
    if not hasattr(target_model, "head_ul"):
        raise ValueError(
            "FedAU需要训练期的辅助遗忘头。请使用 forget_strategy=fedau 从头训练，"
            "或者加载带 client_local_param_states 的 FedAU checkpoint。"
        )

    fedau_mode = str(getattr(args, "fedau_mode", "legacy")).strip().lower()
    if fedau_mode not in {"legacy", "corrected"}:
        raise ValueError(f"Unsupported fedau_mode={fedau_mode}, expected one of ['legacy', 'corrected'].")

    alpha = float(getattr(args, "fedau_alpha", 0.9))
    alpha = min(max(alpha, 0.0), 1.0)
    forget_scale = float(getattr(args, "fedau_forget_scale", 1.0))
    forgotten_model = copy.deepcopy(global_model)
    forgotten_state = forgotten_model.state_dict()
    target_state = target_model.state_dict()

    blended_keys = []
    for main_key in ("head.weight", "head.bias"):
        aux_key = main_key.replace("head.", "head_ul.")
        if main_key not in forgotten_state or aux_key not in target_state:
            continue

        if fedau_mode == "corrected" and main_key in target_state:
            # Corrected branch uses target-local main head as W_l, matching FedAU's W_hat=(1-a)W_l+aW_a.
            base_tensor = target_state[main_key].to(forgotten_state[main_key].device)
        else:
            # Legacy branch keeps historical behavior to preserve old experiments.
            base_tensor = forgotten_state[main_key]

        aux_tensor = target_state[aux_key].to(forgotten_state[main_key].device)
        forgotten_state[main_key] = base_tensor + alpha * forget_scale * (aux_tensor - base_tensor)
        blended_keys.append(main_key)

    forgotten_model.load_state_dict(forgotten_state, strict=False)

    print(f"FedAU模式: {fedau_mode}")
    print(f"FedAU线性组合系数 alpha: {alpha}")
    print(f"FedAU遗忘缩放系数 forget_scale: {forget_scale}")
    return {
        "model": forgotten_model,
        "metadata": {
            "fedau_mode": fedau_mode,
            "fedau_alpha": alpha,
            "fedau_forget_scale": forget_scale,
            "blended_keys": blended_keys,
        },
    }
