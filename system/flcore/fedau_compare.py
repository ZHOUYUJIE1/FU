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

    alpha = float(getattr(args, "fedau_alpha", 0.9))
    alpha = min(max(alpha, 0.0), 1.0)
    forgotten_model = copy.deepcopy(global_model)
    forgotten_state = forgotten_model.state_dict()
    target_state = target_model.state_dict()

    for main_key in ("head.weight", "head.bias"):
        aux_key = main_key.replace("head.", "head_ul.")
        if main_key in forgotten_state and aux_key in target_state:
            forgotten_state[main_key] = (
                (1.0 - alpha) * forgotten_state[main_key]
                + alpha * target_state[aux_key].to(forgotten_state[main_key].device)
            )

    forgotten_model.load_state_dict(forgotten_state, strict=False)

    print(f"FedAU线性组合系数 alpha: {alpha}")
    return {
        "model": forgotten_model,
        "metadata": {
            "fedau_alpha": alpha,
        },
    }
