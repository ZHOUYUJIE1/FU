import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import json
import random
import sys

# 设置环境变量来抑制警告
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 抑制不影响运行的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*threadpoolctl.*")
warnings.filterwarnings("ignore", message=".*sklearn.*")
warnings.filterwarnings("ignore", message=".*scipy.*")

# 重定向stderr来抑制特定的错误输出
import os
import sys

# 创建一个临时的stderr重定向类
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverscaffold import SCAFFOLD

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

# 导入遗忘模块
from flcore.forgetting import forget_client_wrapper
from flcore.eval_attack import membership_inference_attack, backdoor_attack

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_global_seed(seed):
    """设置全局随机种子以确保实验可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TRAINING_AWARE_STRATEGIES = {"fedau", "sifu"}


def get_result_suffix(args):
    """为需要训练期额外状态的方法生成隔离后缀，避免覆盖常规模型。"""
    result_tag = getattr(args, "result_tag", "").strip()
    if result_tag:
        safe_tag = result_tag.replace(os.sep, "_").replace(" ", "_")
        return f"_{safe_tag}"
    forget_strategy = getattr(args, "forget_strategy", "").strip()
    if forget_strategy in TRAINING_AWARE_STRATEGIES:
        return f"_{forget_strategy}"
    if forget_strategy == "fedcsa_compare":
        return "_fedcsa_compare"
    return ""


def build_result_artifact_path(prefix, args, extension):
    suffix = get_result_suffix(args)
    return f"results/exp_configs/{prefix}_{args.dataset}_{args.algorithm}_{args.goal}_{args.times}{suffix}.{extension}"


def resolve_saved_model_path(args):
    saved_model_path = getattr(args, "saved_model_path", "").strip()
    if saved_model_path:
        return saved_model_path
    return build_result_artifact_path("global_model", args, "pt")

def save_experiment_config(args, seed):
    """保存实验配置"""
    config = {
        "seed": seed,
        "goal": args.goal,
        "device": args.device,
        "device_id": args.device_id,
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "model": args.model,
        "batch_size": args.batch_size,
        "local_learning_rate": args.local_learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "learning_rate_decay_gamma": args.learning_rate_decay_gamma,
        "global_rounds": args.global_rounds,
        "top_cnt": args.top_cnt,
        "local_epochs": args.local_epochs,
        "algorithm": args.algorithm,
        "server_learning_rate": args.server_learning_rate,
        "join_ratio": args.join_ratio,
        "random_join_ratio": args.random_join_ratio,
        "num_clients": args.num_clients,
        "prev": args.prev,
        "times": args.times,
        "eval_gap": args.eval_gap,
        "save_folder_name": args.save_folder_name,
        "auto_break": args.auto_break,
        "dlg_eval": args.dlg_eval,
        "dlg_gap": args.dlg_gap,
        "batch_num_per_client": args.batch_num_per_client,
        "num_new_clients": args.num_new_clients,
        "fine_tuning_epoch_new": args.fine_tuning_epoch_new,
        "feature_dim": args.feature_dim,
        "vocab_size": args.vocab_size,
        "max_len": args.max_len,
        "few_shot": args.few_shot,
        "client_drop_rate": args.client_drop_rate,
        "train_slow_rate": args.train_slow_rate,
        "send_slow_rate": args.send_slow_rate,
        "time_select": args.time_select,
        "time_threthold": args.time_threthold,
        "fedosd_lr": getattr(args, "fedosd_lr", None),
        "fedosd_use_legacy": getattr(args, "fedosd_use_legacy", None),
        "fedosd_unlearn_rounds": getattr(args, "fedosd_unlearn_rounds", None),
        "fedosd_recovery_rounds": getattr(args, "fedosd_recovery_rounds", None),
        "fedosd_recovery_lr": getattr(args, "fedosd_recovery_lr", None),
        "fedosd_force_target_online": getattr(args, "fedosd_force_target_online", None),
        "fedau_aux_lr": getattr(args, "fedau_aux_lr", None),
        "fedau_alpha": getattr(args, "fedau_alpha", None),
        "fedau_client_gamma": getattr(args, "fedau_client_gamma", None),
        "fedau_skip_recovery": getattr(args, "fedau_skip_recovery", None),
        "fedau_forget_scale": getattr(args, "fedau_forget_scale", None),
        "sifu_epsilon": getattr(args, "sifu_epsilon", None),
        "sifu_sigma": getattr(args, "sifu_sigma", None),
        "sifu_lambd": getattr(args, "sifu_lambd", None),
        "sifu_skip_recovery": getattr(args, "sifu_skip_recovery", None),
        "sifu_recovery_rounds": getattr(args, "sifu_recovery_rounds", None),
        "fu_retain_calibration_rounds": getattr(args, "fu_retain_calibration_rounds", None),
        "fu_retain_calibration_lr": getattr(args, "fu_retain_calibration_lr", None),
        "fu_retain_calibration_batches": getattr(args, "fu_retain_calibration_batches", None),
        "fu_mask_retain_scale": getattr(args, "fu_mask_retain_scale", None),
        "fu_similarity_boost": getattr(args, "fu_similarity_boost", None),
        "fu_select_best_recovery": getattr(args, "fu_select_best_recovery", None),
        "fu_recovery_target_penalty": getattr(args, "fu_recovery_target_penalty", None),
        "protection_level": getattr(args, "protection_level", None)
    }
    
    # 创建结果目录
    os.makedirs("results/exp_configs", exist_ok=True)
    config_path = build_result_artifact_path("exp_config", args, "json")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"实验配置已保存到: {config_path}")
    return config_path


def extract_buffer_state(model):
    """提取模型中的非参数buffer，用于恢复客户端本地BN统计。"""
    param_keys = {name for name, _ in model.named_parameters()}
    buffer_state = {}
    for key, tensor in model.state_dict().items():
        if key not in param_keys:
            buffer_state[key] = tensor.detach().cpu().clone()
    return buffer_state


def restore_buffer_state(model, buffer_state, device):
    """仅恢复buffer，不覆盖模型参数。"""
    if not buffer_state:
        return

    model_state = model.state_dict()
    for key, tensor in buffer_state.items():
        if key in model_state:
            model_state[key] = tensor.to(device)
    model.load_state_dict(model_state, strict=False)


def get_client_buffer_sidecar_path(model_path):
    """FedAvg checkpoint 的客户端 buffer sidecar 路径。"""
    root, ext = os.path.splitext(model_path)
    if not ext:
        ext = ".pt"
    return f"{root}_client_buffers{ext}"


def has_batchnorm_buffers(model):
    return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


def recalibrate_batchnorm_buffers(model, data_loader, device, max_batches=None):
    """使用本地训练数据重建BN running stats，兼容旧FedAvg checkpoint。"""
    bn_modules = [module for module in model.modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return 0

    saved_momenta = []
    was_training = model.training
    for module in bn_modules:
        module.reset_running_stats()
        saved_momenta.append((module, module.momentum))
        module.momentum = None

    model.train()
    batches = 0
    with torch.no_grad():
        for x, _ in data_loader:
            if type(x) == type([]):
                x[0] = x[0].to(device)
                model(x)
            else:
                x = x.to(device)
                model(x)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    model.train(was_training)
    for module, momentum in saved_momenta:
        module.momentum = momentum

    return batches


def rebuild_client_buffer_states(global_model, clients, server=None, batch_size=None, max_batches=None):
    """旧FedAvg checkpoint 没有保存客户端BN统计时，基于本地训练集重建。"""
    if not has_batchnorm_buffers(global_model):
        return {}

    rebuilt_states = {}
    for client in clients:
        if hasattr(client, 'set_parameters'):
            if server is not None and hasattr(server, 'global_c'):
                client.set_parameters(global_model, global_c=server.global_c, copy_buffers=True)
            else:
                client.set_parameters(global_model, copy_buffers=True)
        else:
            client.model = copy.deepcopy(global_model)

        train_loader = client.load_train_data(batch_size=batch_size)
        num_batches = recalibrate_batchnorm_buffers(
            client.model, train_loader, client.device, max_batches=max_batches
        )
        if num_batches > 0:
            rebuilt_states[client.id] = extract_buffer_state(client.model)

    return rebuilt_states


def load_client_buffer_states_from_checkpoint(model_path, loaded_data, map_location="cpu"):
    """优先从checkpoint中读取；FedAvg旧格式则尝试读取sidecar。"""
    if isinstance(loaded_data, dict) and 'client_buffer_states' in loaded_data:
        return loaded_data['client_buffer_states'], "checkpoint"

    sidecar_path = get_client_buffer_sidecar_path(model_path)
    if os.path.exists(sidecar_path):
        sidecar_data = torch.load(sidecar_path, map_location=map_location)
        if isinstance(sidecar_data, dict) and 'client_buffer_states' in sidecar_data:
            return sidecar_data['client_buffer_states'], sidecar_path
        if isinstance(sidecar_data, dict):
            return sidecar_data, sidecar_path

    return None, None


def get_local_only_param_prefixes(model):
    return tuple(getattr(model, "local_only_param_prefixes", ()))


def extract_local_only_state(model):
    local_only_prefixes = get_local_only_param_prefixes(model)
    if not local_only_prefixes:
        return {}

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if any(key.startswith(prefix) for prefix in local_only_prefixes)
    }


def collect_client_local_param_states(clients):
    if not clients:
        return {}

    local_only_prefixes = get_local_only_param_prefixes(clients[0].model)
    if not local_only_prefixes:
        return {}

    states = {}
    for client in clients:
        local_state = extract_local_only_state(client.model)
        if local_state:
            states[client.id] = local_state
    return states


def restore_client_local_param_states(clients, client_local_param_states, device):
    if not client_local_param_states:
        return 0

    restored = 0
    for client in clients:
        state = client_local_param_states.get(client.id)
        if not state:
            continue
        model_state = client.model.state_dict()
        for key, tensor in state.items():
            if key in model_state:
                model_state[key] = tensor.to(device)
        client.model.load_state_dict(model_state, strict=False)
        restored += 1
    return restored


def load_model_state_with_local_only_support(model, state_dict):
    local_only_prefixes = get_local_only_param_prefixes(model)
    if not local_only_prefixes:
        model.load_state_dict(state_dict)
        return

    incompat = model.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key for key in incompat.missing_keys
        if not any(key.startswith(prefix) for prefix in local_only_prefixes)
    ]
    if missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"加载模型失败，missing_keys={missing_keys}, "
            f"unexpected_keys={incompat.unexpected_keys}"
        )


def normalize_forgetting_result(result):
    if isinstance(result, dict) and "model" in result:
        return result
    return {"model": result}


def restore_client_buffer_states(server, client_buffer_states, device):
    if not client_buffer_states:
        return 0

    restored = 0
    for client in server.clients:
        if client.id in client_buffer_states:
            restore_buffer_state(
                client.model,
                client_buffer_states[client.id],
                device,
            )
            restored += 1
    return restored


def clone_state_dict_to_device(state_dict, device):
    return {
        key: value.to(device)
        for key, value in state_dict.items()
    }


def capture_server_recovery_snapshot(server):
    snapshot = {
        "global_model_state": {
            key: value.detach().cpu().clone()
            for key, value in server.global_model.state_dict().items()
        },
        "client_buffer_states": {
            client.id: extract_buffer_state(client.model)
            for client in server.clients
        } if has_batchnorm_buffers(server.global_model) else {},
    }

    client_local_param_states = collect_client_local_param_states(server.clients)
    if client_local_param_states:
        snapshot["client_local_param_states"] = client_local_param_states

    if hasattr(server, "global_c"):
        snapshot["global_c"] = [param.detach().cpu().clone() for param in server.global_c]
    if any(hasattr(client, "client_c") for client in server.clients):
        snapshot["client_c"] = {
            client.id: [param.detach().cpu().clone() for param in client.client_c]
            for client in server.clients
            if hasattr(client, "client_c")
        }

    return snapshot


def restore_server_recovery_snapshot(server, snapshot, device):
    server.global_model.load_state_dict(
        clone_state_dict_to_device(snapshot["global_model_state"], device),
        strict=False,
    )
    restore_client_buffer_states(server, snapshot.get("client_buffer_states"), device)
    restore_client_local_param_states(
        server.clients,
        snapshot.get("client_local_param_states"),
        device,
    )

    if hasattr(server, "global_c") and snapshot.get("global_c"):
        for server_param, snapshot_param in zip(server.global_c, snapshot["global_c"]):
            server_param.data.copy_(snapshot_param.to(device))
    if snapshot.get("client_c"):
        for client in server.clients:
            if hasattr(client, "client_c") and client.id in snapshot["client_c"]:
                for client_param, snapshot_param in zip(client.client_c, snapshot["client_c"][client.id]):
                    client_param.data.copy_(snapshot_param.to(device))


def compute_target_and_retain_metrics(accs, target_client_id):
    target_acc = accs[target_client_id]
    retain_accs = [acc for idx, acc in enumerate(accs) if idx != target_client_id]
    retain_avg_acc = float(np.mean(retain_accs)) if retain_accs else 0.0
    return target_acc, retain_avg_acc


def select_best_fu_recovery_candidate(server, args, post_accs, post_eval_metrics, recovery_results):
    if not recovery_results:
        return None

    target_penalty = float(getattr(args, "fu_recovery_target_penalty", 1.0))
    post_target_acc, post_retain_avg_acc = compute_target_and_retain_metrics(post_accs, args.target_client_id)

    best_candidate = {
        "label": "post_forget",
        "round": 0,
        "score": post_retain_avg_acc - target_penalty * post_target_acc,
        "metrics": post_eval_metrics,
        "snapshot": None,
    }
    print(
        f"FU恢复候选基线（不恢复）: retain_avg={post_retain_avg_acc:.4f}, "
        f"target_acc={post_target_acc:.4f}, score={best_candidate['score']:.4f}"
    )

    for result in recovery_results:
        snapshot = result.get("snapshot")
        if not snapshot:
            continue

        restore_server_recovery_snapshot(server, snapshot, args.device)
        stage_name = f"FU恢复候选第{result['round']}轮评估"
        accs, aucs, avg_acc, avg_auc, std_acc, std_auc = evaluate_on_all_clients(
            server.global_model,
            server.clients,
            stage_name,
            server=server,
        )
        target_acc, retain_avg_acc = compute_target_and_retain_metrics(accs, args.target_client_id)
        score = retain_avg_acc - target_penalty * target_acc
        print(
            f"FU恢复候选第{result['round']}轮: avg_acc={avg_acc:.4f}, "
            f"retain_avg={retain_avg_acc:.4f}, target_acc={target_acc:.4f}, score={score:.4f}"
        )

        if score > best_candidate["score"]:
            best_candidate = {
                "label": f"recovery_round_{result['round']}",
                "round": int(result["round"]),
                "score": float(score),
                "metrics": (accs, aucs, avg_acc, avg_auc, std_acc, std_auc),
                "snapshot": snapshot,
            }

    return best_candidate


def apply_forgetting_result_side_effects(forget_result, server, args):
    if forget_result.get("client_buffer_states"):
        restored = restore_client_buffer_states(
            server,
            forget_result["client_buffer_states"],
            args.device,
        )
        print(f"已恢复遗忘结果附带的客户端buffer状态: {restored} 个客户端")


def build_base_model(args, model_str):
    if model_str == "MLR":  # convex
        if "MNIST" in args.dataset:
            model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
        else:
            model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)
    elif model_str == "CNN":  # non-convex
        if "MNIST" in args.dataset:
            model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "Omniglot" in args.dataset:
            model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
        elif "Digit5" in args.dataset:
            model = Digit5CNN().to(args.device)
        else:
            model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
    elif model_str == "DNN":  # non-convex
        if "MNIST" in args.dataset:
            model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
        else:
            model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
    elif model_str == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
    elif model_str == "ResNet10":
        model = resnet10(num_classes=args.num_classes).to(args.device)
    elif model_str == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
    elif model_str == "AlexNet":
        model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
    elif model_str == "GoogleNet":
        model = torchvision.models.googlenet(
            pretrained=False,
            aux_logits=False,
            num_classes=args.num_classes,
        ).to(args.device)
    elif model_str == "MobileNet":
        model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
    elif model_str == "LSTM":
        model = LSTMNet(
            hidden_dim=args.feature_dim,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
        ).to(args.device)
    elif model_str == "BiLSTM":
        model = BiLSTM_TextClassification(
            input_size=args.vocab_size,
            hidden_size=args.feature_dim,
            output_size=args.num_classes,
            num_layers=1,
            embedding_dropout=0,
            lstm_dropout=0,
            attention_dropout=0,
            embedding_length=args.feature_dim,
        ).to(args.device)
    elif model_str == "fastText":
        model = fastText(
            hidden_dim=args.feature_dim,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
        ).to(args.device)
    elif model_str == "TextCNN":
        model = TextCNN(
            hidden_dim=args.feature_dim,
            max_len=args.max_len,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes,
        ).to(args.device)
    elif model_str == "Transformer":
        model = TransformerModel(
            ntoken=args.vocab_size,
            d_model=args.feature_dim,
            nhead=8,
            nlayers=2,
            num_classes=args.num_classes,
            max_len=args.max_len,
        ).to(args.device)
    elif model_str == "AmazonMLP":
        model = AmazonMLP().to(args.device)
    elif model_str == "HARCNN":
        if args.dataset == 'HAR':
            model = HARCNN(
                9,
                dim_hidden=1664,
                num_classes=args.num_classes,
                conv_kernel_size=(1, 9),
                pool_kernel_size=(1, 2),
            ).to(args.device)
        elif args.dataset == 'PAMAP2':
            model = HARCNN(
                9,
                dim_hidden=3712,
                num_classes=args.num_classes,
                conv_kernel_size=(1, 9),
                pool_kernel_size=(1, 2),
            ).to(args.device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model


def create_server(args, times, model_str):
    args.model = build_base_model(args, model_str)

    if args.algorithm in {"FedAvg", "SCAFFOLD"}:
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        if getattr(args, "forget_strategy", "") == "fedau":
            args.model = FedAUHeadSplit(args.model, args.head, copy.deepcopy(args.head))
        else:
            args.model = BaseHeadSplit(args.model, args.head)
    else:
        raise NotImplementedError

    print(args.model)

    if args.algorithm == "FedAvg":
        server = FedAvg(args, times)
    elif args.algorithm == "SCAFFOLD":
        server = SCAFFOLD(args, times)
    else:
        raise NotImplementedError

    if getattr(args, "forget_strategy", "") == "sifu" and not getattr(args, "load_saved_model", False):
        from flcore.sifu_compare import SIFUTracker

        server.sifu_tracker = SIFUTracker(args)

    return server

def evaluate_on_all_clients(global_model, clients, stage_name="评估", server=None):
    """在所有客户端数据上评估全局模型"""
    print(f"\n============= {stage_name} =============", flush=True)
    accs = []
    aucs = []
    total_clients = len(clients)
    
    for i, client in enumerate(clients):
        print(
            f"[{stage_name}] 评估进度: 客户端 {i + 1}/{total_clients} (client_id={client.id})",
            flush=True,
        )
        # 设置客户端模型为全局模型
        # 对于SCAFFOLD，需要传递global_c（评估时可以为None）
        if hasattr(client, 'set_parameters'):
            if server is not None and hasattr(server, 'global_c'):
                # SCAFFOLD需要global_c参数
                client.set_parameters(global_model, global_c=server.global_c)
            else:
                client.set_parameters(global_model)
        else:
            client.model = copy.deepcopy(global_model)
        
        # 评估
        test_acc, test_num, auc = client.test_metrics()
        
        if test_num == 0:
            print(f"警告：客户端{i}的测试样本数为0，可能测试集为空或无法加载！")
            acc = 0.0
        else:
            acc = test_acc / test_num
            if acc == 0.0 and test_num > 0:
                print(f"提示：客户端{i}使用测试集评估，测试样本数={test_num}，但准确率为0（可能模型完全预测错误）")
                # 添加详细调试信息：检查预测分布
                if i == 5:  # 特别关注客户端5
                    print(f"  调试：检查客户端{i}的预测分布...")
                    client.model.eval()
                    testloader = client.load_test_data()
                    all_predictions = []
                    all_labels = []
                    all_outputs = []  # 保存原始输出
                    with torch.no_grad():
                        for x, y in testloader:
                            if type(x) == type([]):
                                x[0] = x[0].to(client.device)
                            else:
                                x = x.to(client.device)
                            output = client.model(x)
                            pred = torch.argmax(output, dim=1)
                            all_predictions.extend(pred.cpu().numpy())
                            all_labels.extend(y.numpy())
                            all_outputs.append(output.cpu().numpy())
                            if len(all_predictions) >= 100:  # 只检查前100个样本
                                break
                    
                    from collections import Counter
                    pred_counter = Counter(all_predictions)
                    label_counter = Counter(all_labels[:len(all_predictions)])
                    print(f"  客户端{i}前100个样本 - 真实标签分布: {dict(label_counter)}")
                    print(f"  客户端{i}前100个样本 - 预测标签分布: {dict(pred_counter)}")
                    
                    # 检查模型输出层对类别4和5的logits
                    if len(all_outputs) > 0:
                        outputs_array = np.concatenate(all_outputs, axis=0)[:len(all_labels)]  # 确保长度匹配
                        labels_array = np.array(all_labels[:len(outputs_array)])
                        # 找出标签为4和5的样本
                        idx_4 = np.where(labels_array == 4)[0]
                        idx_5 = np.where(labels_array == 5)[0]
                        if len(idx_4) > 0:
                            print(f"  标签4的样本数: {len(idx_4)}")
                            avg_logits_4 = outputs_array[idx_4].mean(axis=0)
                            print(f"  标签4样本的平均logits (前5个): {avg_logits_4[:5]}")
                            print(f"  标签4样本的平均logits (后5个): {avg_logits_4[-5:]}")
                            pred_4 = np.array(all_predictions[:len(outputs_array)])[idx_4]
                            print(f"  标签4样本的预测类别分布: {dict(Counter(pred_4))}")
                        if len(idx_5) > 0:
                            print(f"  标签5的样本数: {len(idx_5)}")
                            avg_logits_5 = outputs_array[idx_5].mean(axis=0)
                            print(f"  标签5样本的平均logits (前5个): {avg_logits_5[:5]}")
                            print(f"  标签5样本的平均logits (后5个): {avg_logits_5[-5:]}")
                            pred_5 = np.array(all_predictions[:len(outputs_array)])[idx_5]
                            print(f"  标签5样本的预测类别分布: {dict(Counter(pred_5))}")
                    
                    # 检查模型参数是否正确设置
                    print(f"  检查模型参数设置...")
                    print(f"  模型输出层形状: {list(client.model.parameters())[-1].shape}")
                    print(f"  模型是否在eval模式: {not client.model.training}")
        
        accs.append(acc)
        aucs.append(auc)
        
        print(f"客户端{i} - 准确率: {acc:.4f}, AUC: {auc:.4f}, 测试样本数: {test_num}")
    
    avg_acc = np.mean(accs)
    avg_auc = np.mean(aucs)
    std_acc = np.std(accs)
    std_auc = np.std(aucs)
    
    print(f"\n平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    
    return accs, aucs, avg_acc, avg_auc, std_acc, std_auc

def save_evaluation_results(accs, aucs, avg_acc, avg_auc, std_acc, std_auc, stage_name, args):
    """保存评估结果"""
    results = {
        "stage": stage_name,
        "client_accuracies": accs,
        "client_aucs": aucs,
        "average_accuracy": avg_acc,
        "average_auc": avg_auc,
        "std_accuracy": std_acc,
        "std_auc": std_auc
    }
    
    results_path = build_result_artifact_path(f"eval_{stage_name}", args, "json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"评估结果已保存到: {results_path}")


def build_unlearning_comparison_metrics(pre_accs, post_accs, target_client_id, mia_pre=None, mia_post=None):
    """构建统一对比指标，便于不同遗忘方法在同一场景下横向比较。"""
    target_pre_acc = float(pre_accs[target_client_id])
    target_post_acc = float(post_accs[target_client_id])

    retain_indices = [i for i in range(len(pre_accs)) if i != target_client_id]
    retain_pre_acc = float(np.mean([pre_accs[i] for i in retain_indices])) if retain_indices else 0.0
    retain_post_acc = float(np.mean([post_accs[i] for i in retain_indices])) if retain_indices else 0.0

    metrics = {
        "target_pre_acc": target_pre_acc,
        "target_post_acc": target_post_acc,
        "target_acc_drop": float(target_pre_acc - target_post_acc),
        "retain_pre_acc": retain_pre_acc,
        "retain_post_acc": retain_post_acc,
        "retain_acc_drop": float(retain_pre_acc - retain_post_acc),
        "utility_retention_ratio": float(retain_post_acc / (retain_pre_acc + 1e-12))
    }

    if mia_pre is not None:
        metrics["mia_pre_auc"] = float(mia_pre.get("auc", 0.0))
        metrics["mia_pre_sr"] = float(mia_pre.get("mia_sr", mia_pre.get("best_acc", 0.0)))
        metrics["mia_pre_fixed_acc"] = float(mia_pre.get("fixed_acc", 0.0))
    if mia_post is not None:
        metrics["mia_post_auc"] = float(mia_post.get("auc", 0.0))
        metrics["mia_post_sr"] = float(mia_post.get("mia_sr", mia_post.get("best_acc", 0.0)))
        metrics["mia_post_fixed_acc"] = float(mia_post.get("fixed_acc", 0.0))
    if mia_pre is not None and mia_post is not None:
        metrics["mia_sr_drop"] = float(metrics["mia_pre_sr"] - metrics["mia_post_sr"])
        metrics["mia_auc_drop"] = float(metrics["mia_pre_auc"] - metrics["mia_post_auc"])

    return metrics

def run(args):
    # 设置全局随机种子
    seed = args.random_seed  # 可配置的种子
    set_global_seed(seed)
    
    # 保存实验配置
    config_path = save_experiment_config(args, seed)
    
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if args.load_saved_model:
            # 只加载模型，直接评估和遗忘
            model_path = resolve_saved_model_path(args)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            print(f"从已保存的模型文件加载: {model_path}")
            server = create_server(args, i, model_str)

            # 如果只做“重训练基线”，则在训练阶段排除目标客户端，并跳过遗忘阶段
            if getattr(args, "retrain_only", False):
                print("\n============= 仅执行重训练基线（排除目标客户端参与训练） =============")
                # 这里的 server 还未训练，global_model 为随机初始化 / 预定义初始化
                # 调用各自服务器中的 retrain_without_client 方法
                if hasattr(server, "retrain_without_client"):
                    retrain_model = server.retrain_without_client(
                        target_client_id=args.target_client_id,
                        retrain_rounds=args.global_rounds
                    )
                else:
                    raise NotImplementedError("当前算法未实现 retrain_without_client 方法")

                # 对重训练后的模型在所有客户端上做一次完整评估
                print("\n============= 重训练后评估（所有客户端） =============")
                rt_accs, rt_aucs, rt_avg_acc, rt_avg_auc, rt_std_acc, rt_std_auc = evaluate_on_all_clients(
                    retrain_model, server.clients, "重训练后评估", server=server
                )

                # 保存评估结果
                save_evaluation_results(
                    rt_accs, rt_aucs, rt_avg_acc, rt_avg_auc, rt_std_acc, rt_std_auc,
                    "retrain_only", args
                )

                # 对目标客户端做同样的攻击评估（MIA + 后门）
                print("\n============= 重训练后攻击评估（目标客户端） =============")
                mia_rt = membership_inference_attack(retrain_model, server.clients[args.target_client_id], args.device)
                print(f"[重训练后MIA] AUC: {mia_rt['auc']:.4f}, 最佳准确率: {mia_rt['best_acc']:.4f}, 最佳阈值: {mia_rt['best_thr']:.2f}")
                backdoor_acc_rt = backdoor_attack(retrain_model, server.clients[args.target_client_id], args.device)
                print(f"[重训练后后门攻击] 后门样本准确率: {backdoor_acc_rt:.4f}")

                # 保存攻击评估结果
                with open(build_result_artifact_path("attack_retrain_only", args, "txt"), "w") as f:
                    f.write(f"MIA AUC: {mia_rt['auc']:.4f}\nMIA Best Acc: {mia_rt['best_acc']:.4f}\nMIA Best Thr: {mia_rt['best_thr']:.2f}\n")
                    f.write(f"Backdoor Acc: {backdoor_acc_rt:.4f}\n")

                time_list.append(time.time() - start)
                return
            
            # 然后加载参数
            loaded_data = torch.load(model_path, map_location=args.device)
            client_buffer_states, client_buffer_source = load_client_buffer_states_from_checkpoint(
                model_path, loaded_data, map_location=args.device
            )
            
            # 检查是否是包含额外元信息的保存格式（SCAFFOLD/FedAvg增强格式）
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                # 新格式：至少包含模型参数，可能还带控制变量或客户端buffer状态
                load_model_state_with_local_only_support(args.model, loaded_data['model_state_dict'])
                global_model = args.model
                print("模型参数已通过state_dict加载完毕！")
                
                # 如果是 SCAFFOLD 算法，加载控制变量
                if args.algorithm == "SCAFFOLD" and hasattr(server, 'global_c'):
                    if 'global_c' in loaded_data:
                        # 将 global_c 恢复到正确的设备
                        server.global_c = [c.to(args.device) for c in loaded_data['global_c']]
                        print("global_c 已加载！")
                    
                    if 'client_c' in loaded_data:
                        # 恢复每个客户端的 client_c
                        for client in server.clients:
                            if hasattr(client, 'client_c') and client.id in loaded_data['client_c']:
                                client.client_c = [c.to(args.device) for c in loaded_data['client_c'][client.id]]
                        print("所有客户端的 client_c 已加载！")
            else:
                # 旧格式：只有模型参数（向后兼容）
                load_model_state_with_local_only_support(args.model, loaded_data)
                global_model = args.model
                print("模型参数已通过state_dict加载完毕！")
                print("注意：这是旧格式的模型文件，不包含额外checkpoint元信息。")
                if args.algorithm == "SCAFFOLD":
                    print("警告：SCAFFOLD 的控制变量将使用默认初始值（零）。")

            client_local_param_states = None
            if isinstance(loaded_data, dict):
                client_local_param_states = loaded_data.get("client_local_param_states")
                if "sifu_state" in loaded_data and getattr(args, "forget_strategy", "") == "sifu":
                    from flcore.sifu_compare import SIFUTracker

                    server.sifu_tracker = SIFUTracker.from_state_dict(args, loaded_data["sifu_state"])
                    print("SIFU训练历史已加载！")

            if client_buffer_states:
                restored_client_count = 0
                for client in server.clients:
                    if client.id in client_buffer_states:
                        restore_buffer_state(
                            client.model,
                            client_buffer_states[client.id],
                            args.device,
                        )
                        restored_client_count += 1
                print(f"已为 {restored_client_count} 个客户端恢复本地buffer状态（来源: {client_buffer_source}）。")
            elif args.algorithm == "SCAFFOLD":
                # 兼容旧checkpoint：至少先用全局模型的buffer初始化客户端，避免随机初值污染评估。
                for client in server.clients:
                    client.model.load_state_dict(global_model.state_dict(), strict=False)
                print("未发现客户端buffer状态，已使用全局模型buffer初始化所有客户端。")
            elif has_batchnorm_buffers(global_model):
                print("未发现FedAvg客户端buffer状态，开始基于各客户端训练数据重建BN统计...")
                rebuilt_buffer_states = rebuild_client_buffer_states(
                    global_model,
                    server.clients,
                    server=server,
                    batch_size=args.batch_size,
                )
                if rebuilt_buffer_states:
                    sidecar_path = get_client_buffer_sidecar_path(model_path)
                    torch.save({'client_buffer_states': rebuilt_buffer_states}, sidecar_path)
                    print(f"已为 {len(rebuilt_buffer_states)} 个客户端重建BN统计，并缓存到: {sidecar_path}")
                else:
                    for client in server.clients:
                        client.model.load_state_dict(global_model.state_dict(), strict=False)
                    print("未能重建客户端BN统计，已回退为使用全局模型buffer初始化所有客户端。")

            if client_local_param_states:
                restored = restore_client_local_param_states(
                    server.clients,
                    client_local_param_states,
                    args.device,
                )
                print(f"已恢复客户端本地局部参数状态: {restored} 个客户端")

            if args.forget_strategy == "fedau" and not client_local_param_states:
                raise ValueError(
                    "FedAU无法直接基于普通checkpoint运行。"
                    "请使用 forget_strategy=fedau 重新训练并保存，或提供带 client_local_param_states 的checkpoint。"
                )
            if args.forget_strategy == "sifu" and getattr(server, "sifu_tracker", None) is None:
                raise ValueError(
                    "SIFU需要训练期历史信息。"
                    "请使用 forget_strategy=sifu 从头训练，或加载带 sifu_state 的checkpoint。"
                )
            
            print("跳过训练阶段，直接进行遗忘...")
            # 评估/遗忘流程
            # 遗忘前评估：在所有客户端数据上测试准确率
            print("\n============= 遗忘前评估 =============")
            pre_accs, pre_aucs, pre_avg_acc, pre_avg_auc, pre_std_acc, pre_std_auc = evaluate_on_all_clients(
                global_model, server.clients, "遗忘前评估", server=server)
            
            # 保存遗忘前评估结果
            save_evaluation_results(pre_accs, pre_aucs, pre_avg_acc, pre_avg_auc, pre_std_acc, pre_std_auc, 
                                  "pre_forget", args)

            # 遗忘前攻击评估
            print("\n============= 遗忘前攻击评估 =============")
            mia_pre = membership_inference_attack(global_model, server.clients[args.target_client_id], args.device)
            print(f"[遗忘前MIA] AUC: {mia_pre['auc']:.4f}, 最佳准确率: {mia_pre['best_acc']:.4f}, 最佳阈值: {mia_pre['best_thr']:.2f}")
            backdoor_acc_pre = backdoor_attack(global_model, server.clients[args.target_client_id], args.device)
            print(f"[遗忘前后门攻击] 后门样本准确率: {backdoor_acc_pre:.4f}")
            # 保存攻击评估结果
            with open(build_result_artifact_path("attack_pre_forget", args, "txt"), "w") as f:
                f.write(f"MIA AUC: {mia_pre['auc']:.4f}\nMIA Best Acc: {mia_pre['best_acc']:.4f}\nMIA Best Thr: {mia_pre['best_thr']:.2f}\n")
                f.write(f"Backdoor Acc: {backdoor_acc_pre:.4f}\n")

            # 遗忘阶段：遗忘客户端{args.target_client_id}
            print(f"\n============= 开始遗忘阶段 =============")
            print(f"目标遗忘客户端: {args.target_client_id}")
            with SuppressStderr():
                if args.forget_strategy == "adversarial":
                    from flcore.forgetting import forget_client_wrapper
                    global_model_forget = forget_client_wrapper(global_model, server.clients, args, target_client_id=args.target_client_id)
                    
                elif args.forget_strategy == "gradient_reversal":
                    from flcore.forgetting import forget_client_with_gradient_reversal
                    global_model_forget = forget_client_with_gradient_reversal(global_model, server.clients, args, target_client_id=args.target_client_id, pre_accuracies=pre_accs)
                
                elif args.forget_strategy == "fedcsa":
                    from flcore.forgetting import forget_client_with_fedcsa
                    global_model_forget = forget_client_with_fedcsa(global_model, server.clients, args, target_client_id=args.target_client_id)

                elif args.forget_strategy == "fedcsa_compare":
                    from flcore.fedcsa_compare import forget_client_with_fedcsa_compare
                    global_model_forget = forget_client_with_fedcsa_compare(global_model, server.clients, args, target_client_id=args.target_client_id)
                
                elif args.forget_strategy == "fedosd":
                    from flcore.forgetting import forget_client_with_fedosd
                    global_model_forget = forget_client_with_fedosd(
                        global_model,
                        server.clients,
                        args,
                        target_client_id=args.target_client_id,
                        server=server,
                    )
                
                elif args.forget_strategy == "fedu":
                    from flcore.forgetting import forget_client_with_fedu
                    global_model_forget = forget_client_with_fedu(global_model, server.clients, args, target_client_id=args.target_client_id)
                elif args.forget_strategy == "fedau":
                    from flcore.fedau_compare import forget_client_with_fedau
                    global_model_forget = forget_client_with_fedau(
                        global_model,
                        server.clients,
                        args,
                        target_client_id=args.target_client_id,
                    )
                elif args.forget_strategy == "sifu":
                    from flcore.sifu_compare import forget_client_with_sifu
                    global_model_forget = forget_client_with_sifu(
                        global_model,
                        server.clients,
                        args,
                        target_client_id=args.target_client_id,
                        server=server,
                    )
                else:
                    raise ValueError(f"Unknown forgetting strategy: {args.forget_strategy}")

            forget_result = normalize_forgetting_result(global_model_forget)
            global_model_forget = forget_result["model"]
            apply_forgetting_result_side_effects(forget_result, server, args)
            method_recovery_results = list(forget_result.get("recovery_results", []))
            method_recovery_rounds = int(forget_result.get("recovery_rounds", 0) or 0)
            method_recovery_stage = forget_result.get("recovery_stage")
            method_post_recovery_model = forget_result.get("post_recovery_model")
            method_post_recovery_client_buffer_states = forget_result.get("post_recovery_client_buffer_states")
            
            # 保存遗忘后的模型
            forget_model_save_path = build_result_artifact_path("forget_model", args, "pt")
            torch.save(global_model_forget, forget_model_save_path)
            print(f"遗忘后模型已保存到: {forget_model_save_path}")

            # 遗忘后评估：在所有客户端数据上测试准确率
            print("\n============= 遗忘后评估 =============")
            post_accs, post_aucs, post_avg_acc, post_avg_auc, post_std_acc, post_std_auc = evaluate_on_all_clients(
                global_model_forget, server.clients, "遗忘后评估", server=server)
            
            # 保存遗忘后评估结果
            save_evaluation_results(post_accs, post_aucs, post_avg_acc, post_avg_auc, post_std_acc, post_std_auc, 
                                  "post_forget", args)

            # 计算遗忘效果（以遗忘前准确率为基准）
            print("\n============= 遗忘效果分析 =============")
            per_client_abs_change = [post_accs[i] - pre_accs[i] for i in range(len(pre_accs))]
            per_client_rel_change = [
                (post_accs[i] - pre_accs[i]) / (pre_accs[i] + 1e-12) for i in range(len(pre_accs))
            ]
            target_abs_change = per_client_abs_change[args.target_client_id]
            target_rel_change = per_client_rel_change[args.target_client_id]
            # 计算其他客户端的平均变化（排除目标客户端）
            other_indices = [i for i in range(len(pre_accs)) if i != args.target_client_id]
            others_abs_change_avg = np.mean([per_client_abs_change[i] for i in other_indices]) if len(other_indices) > 0 else 0.0
            valid_other_rel_changes = [per_client_rel_change[i] for i in other_indices if pre_accs[i] > 1e-6]
            others_rel_change_avg = np.mean(valid_other_rel_changes) if len(valid_other_rel_changes) > 0 else 0.0

            print(f"目标客户端({args.target_client_id}) 遗忘前: {pre_accs[args.target_client_id]:.4f} 遗忘后: {post_accs[args.target_client_id]:.4f}")
            print(f"目标客户端({args.target_client_id}) 变化量: {target_abs_change:.4f} (相对 {target_rel_change*100:.2f}%)")
            print(f"其他客户端 平均变化量: {others_abs_change_avg:.4f} (相对 {others_rel_change_avg*100:.2f}%)")

            # 直接进入恢复阶段（无需阈值判断）
            skip_external_recovery = (
                (args.forget_strategy == "fedosd" and not getattr(args, "fedosd_use_legacy", False))
                or (args.forget_strategy == "fedau" and getattr(args, "fedau_skip_recovery", True))
                or args.forget_strategy == "sifu"
            )
            if args.forget_strategy == "fedosd" and skip_external_recovery:
                recovery_rounds = getattr(args, 'fedosd_recovery_rounds', 0)
            elif args.forget_strategy == "sifu" and skip_external_recovery:
                recovery_rounds = method_recovery_rounds
            else:
                recovery_rounds = 0
            recovery_results = list(method_recovery_results)
            recovery_accs = None
            recovery_aucs = None
            recovery_avg_acc = None
            recovery_avg_auc = None
            recovery_effect = None

            if skip_external_recovery:
                print(f"\n============= 跳过外层恢复阶段 =============")
                if args.forget_strategy == "fedosd":
                    print("FedOSD 对比实现已在遗忘函数内部完成 post-training，外层通用恢复不再重复执行。")
                elif args.forget_strategy == "fedau":
                    print("FedAU 默认不执行额外服务器恢复阶段，以保持与官方方法更一致。")
                elif args.forget_strategy == "sifu":
                    if method_post_recovery_model is not None:
                        print("SIFU 对比实现已在方法内部完成回跳后的恢复训练，外层通用恢复不再重复执行。")
                        server.global_model = method_post_recovery_model
                        restored = restore_client_buffer_states(
                            server,
                            method_post_recovery_client_buffer_states,
                            args.device,
                        )
                        if restored:
                            print(f"已恢复 SIFU 内部恢复后的客户端buffer状态: {restored} 个客户端")

                        recovery_model_save_path = build_result_artifact_path("recovery_model", args, "pt")
                        torch.save(server.global_model, recovery_model_save_path)
                        print(f"恢复后模型已保存到: {recovery_model_save_path}")

                        print("\n============= 恢复后评估 =============")
                        recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc = evaluate_on_all_clients(
                            server.global_model, server.clients, "恢复后评估", server=server)
                        save_evaluation_results(
                            recovery_accs,
                            recovery_aucs,
                            recovery_avg_acc,
                            recovery_avg_auc,
                            recovery_std_acc,
                            recovery_std_auc,
                            "post_recovery",
                            args,
                        )
                        recovery_effect = recovery_avg_acc - post_avg_acc
                        print(f"恢复效果: 平均准确率提升 {recovery_effect:.4f} ({recovery_effect/post_avg_acc*100:.2f}%)")
                        global_model_forget = server.global_model
                    else:
                        print("SIFU 已按当前配置跳过恢复阶段。")
            else:
                print(f"\n============= 开始恢复阶段 =============")
                print(f"其他客户端性能变化: {others_rel_change_avg*100:.2f}%")
                
                recovery_rounds = getattr(args, 'recovery_rounds', 5)  # 默认恢复5轮
                print(f"恢复训练轮数: {recovery_rounds}")
                
                # 使用遗忘后的模型进行恢复训练
                server.global_model = global_model_forget
                recovery_results = server.recovery_training(target_client_id=args.target_client_id, recovery_rounds=recovery_rounds)
                
                # 保存恢复后的模型
                recovery_model_save_path = build_result_artifact_path("recovery_model", args, "pt")
                torch.save(server.global_model, recovery_model_save_path)
                print(f"恢复后模型已保存到: {recovery_model_save_path}")
                
                # 恢复后评估
                print("\n============= 恢复后评估 =============")
                recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc = evaluate_on_all_clients(
                    server.global_model, server.clients, "恢复后评估", server=server)
                
                # 保存恢复后评估结果
                save_evaluation_results(recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc, 
                                      "post_recovery", args)
                
                # 计算恢复效果
                recovery_effect = recovery_avg_acc - post_avg_acc
                print(f"恢复效果: 平均准确率提升 {recovery_effect:.4f} ({recovery_effect/post_avg_acc*100:.2f}%)")
                
                # 更新最终模型
                global_model_forget = server.global_model

            # 保存遗忘效果汇总
            forget_summary = {
                "pre_accs": pre_accs,
                "post_accs": post_accs,
                "per_client_abs_change": per_client_abs_change,
                "per_client_rel_change": per_client_rel_change,
                "target_abs_change": target_abs_change,
                "target_rel_change": target_rel_change,
                "others_abs_change_avg": float(others_abs_change_avg),
                "others_rel_change_avg": float(others_rel_change_avg),
                "recovery_rounds": recovery_rounds,
                "recovery_triggered": recovery_rounds > 0,
                "recovery_stage": (
                    method_recovery_stage
                    if args.forget_strategy == "sifu" and method_recovery_stage
                    else
                    "internal_fedosd_post_training"
                    if args.forget_strategy == "fedosd" and skip_external_recovery
                    else "skipped_for_fedau"
                    if args.forget_strategy == "fedau" and skip_external_recovery
                    else "external_server_recovery"
                )
            }
            if forget_result.get("metadata"):
                forget_summary["method_metadata"] = forget_result["metadata"]
            
            # 添加恢复结果
            forget_summary["recovery_results"] = recovery_results
            if recovery_accs is not None:
                forget_summary["recovery_accs"] = recovery_accs
                forget_summary["recovery_aucs"] = recovery_aucs
                forget_summary["recovery_avg_acc"] = float(recovery_avg_acc)
                forget_summary["recovery_avg_auc"] = float(recovery_avg_auc)
                forget_summary["recovery_effect"] = float(recovery_effect)
            
            forget_summary_path = build_result_artifact_path("forget_effect", args, "json")
            with open(forget_summary_path, "w") as f:
                json.dump(forget_summary, f, indent=2)
            print(f"遗忘效果汇总已保存到: {forget_summary_path}")


            # 遗忘后攻击评估
            print("\n============= 遗忘后攻击评估 =============")
            mia_post = membership_inference_attack(global_model_forget, server.clients[args.target_client_id], args.device)
            print(f"[遗忘后MIA] AUC: {mia_post['auc']:.4f}, 最佳准确率: {mia_post['best_acc']:.4f}, 最佳阈值: {mia_post['best_thr']:.2f}")
            backdoor_acc_post = backdoor_attack(global_model_forget, server.clients[args.target_client_id], args.device)
            print(f"[遗忘后后门攻击] 后门样本准确率: {backdoor_acc_post:.4f}")
            # 保存攻击评估结果
            with open(build_result_artifact_path("attack_post_forget", args, "txt"), "w") as f:
                f.write(f"MIA AUC: {mia_post['auc']:.4f}\nMIA Best Acc: {mia_post['best_acc']:.4f}\nMIA Best Thr: {mia_post['best_thr']:.2f}\n")
                f.write(f"Backdoor Acc: {backdoor_acc_post:.4f}\n")

            time_list.append(time.time()-start)
            return
        else:
            # 正常的训练流程
            server = create_server(args, i, model_str)

        # 如果只做"重训练基线"，则在训练阶段排除目标客户端，并跳过遗忘阶段
        if getattr(args, "retrain_only", False):
            print("\n============= 仅执行重训练基线（排除目标客户端参与训练） =============")
            # 这里的 server 还未训练，global_model 为随机初始化 / 预定义初始化
            # 调用各自服务器中的 retrain_without_client 方法
            if hasattr(server, "retrain_without_client"):
                retrain_model = server.retrain_without_client(
                    target_client_id=args.target_client_id,
                    retrain_rounds=args.global_rounds
                )
            else:
                raise NotImplementedError("当前算法未实现 retrain_without_client 方法")

            # 对重训练后的模型在所有客户端上做一次完整评估
            print("\n============= 重训练后评估（所有客户端） =============")
            rt_accs, rt_aucs, rt_avg_acc, rt_avg_auc, rt_std_acc, rt_std_auc = evaluate_on_all_clients(
                retrain_model, server.clients, "重训练后评估", server=server
            )

            # 保存评估结果
            save_evaluation_results(
                rt_accs, rt_aucs, rt_avg_acc, rt_avg_auc, rt_std_acc, rt_std_auc,
                "retrain_only", args
            )

            # 对目标客户端做同样的攻击评估（MIA + 后门）
            print("\n============= 重训练后攻击评估（目标客户端） =============")
            mia_rt = membership_inference_attack(retrain_model, server.clients[args.target_client_id], args.device)
            print(f"[重训练后MIA] AUC: {mia_rt['auc']:.4f}, 最佳准确率: {mia_rt['best_acc']:.4f}, 最佳阈值: {mia_rt['best_thr']:.2f}")
            backdoor_acc_rt = backdoor_attack(retrain_model, server.clients[args.target_client_id], args.device)
            print(f"[重训练后后门攻击] 后门样本准确率: {backdoor_acc_rt:.4f}")

            # 保存攻击评估结果
            with open(build_result_artifact_path("attack_retrain_only", args, "txt"), "w") as f:
                f.write(f"MIA AUC: {mia_rt['auc']:.4f}\nMIA Best Acc: {mia_rt['best_acc']:.4f}\nMIA Best Thr: {mia_rt['best_thr']:.2f}\n")
                f.write(f"Backdoor Acc: {backdoor_acc_rt:.4f}\n")

            # 保存重训练后的模型
            retrain_model_save_path = build_result_artifact_path("retrain_model", args, "pt")
            torch.save(retrain_model.state_dict(), retrain_model_save_path)
            print(f"重训练后模型已保存到: {retrain_model_save_path}")

            time_list.append(time.time() - start)
            return

        # 训练联邦学习模型
        server.train()
        
        # 保存训练后的全局模型
        global_model = copy.deepcopy(server.global_model)
        if args.algorithm == "FedAvg" and getattr(server, 'best_model_state_dict', None) is not None:
            global_model.load_state_dict(server.best_model_state_dict)
            server.global_model.load_state_dict(server.best_model_state_dict)
            if getattr(server, 'best_client_buffer_states', None):
                for client in server.clients:
                    if client.id in server.best_client_buffer_states:
                        restore_buffer_state(
                            client.model,
                            server.best_client_buffer_states[client.id],
                            args.device,
                        )
            if getattr(server, 'best_client_local_param_states', None):
                restore_client_local_param_states(
                    server.clients,
                    server.best_client_local_param_states,
                    args.device,
                )
            print(
                f"FedAvg将使用最佳评估快照作为保存结果："
                f"round={server.best_eval_round}, acc={server.best_test_acc:.4f}"
            )

        model_save_path = build_result_artifact_path("global_model", args, "pt")
        client_buffer_states = {
            client.id: extract_buffer_state(client.model)
            for client in server.clients
        } if has_batchnorm_buffers(global_model) else {}
        client_local_param_states = collect_client_local_param_states(server.clients)
        extra_checkpoint = {}
        if client_local_param_states:
            extra_checkpoint['client_local_param_states'] = client_local_param_states
        if getattr(server, 'sifu_tracker', None) is not None:
            extra_checkpoint['sifu_state'] = server.sifu_tracker.state_dict()
        
        # 如果是 SCAFFOLD 算法，需要同时保存控制变量
        if args.algorithm == "SCAFFOLD" and hasattr(server, 'global_c'):
            save_dict = {
                'model_state_dict': global_model.state_dict(),
                'global_c': [c.cpu().clone() for c in server.global_c],  # 保存到 CPU 以节省 GPU 内存
                'client_c': {client.id: [c.cpu().clone() for c in client.client_c] 
                            for client in server.clients if hasattr(client, 'client_c')},
                'client_buffer_states': client_buffer_states,
            }
            save_dict.update(extra_checkpoint)
            torch.save(save_dict, model_save_path)
            print(f"全局模型和控制变量已保存到: {model_save_path}")
        elif extra_checkpoint:
            save_dict = {
                'model_state_dict': global_model.state_dict(),
                'client_buffer_states': client_buffer_states,
            }
            save_dict.update(extra_checkpoint)
            torch.save(save_dict, model_save_path)
            print(f"全局模型和扩展元信息已保存到: {model_save_path}")
        else:
            torch.save(global_model.state_dict(), model_save_path)
            print(f"全局模型已保存到: {model_save_path}")
            if client_buffer_states:
                sidecar_path = get_client_buffer_sidecar_path(model_save_path)
                torch.save({'client_buffer_states': client_buffer_states}, sidecar_path)
                print(f"客户端buffer状态已保存到: {sidecar_path}")

        # 遗忘前评估：在所有客户端数据上测试准确率
        print("\n============= 遗忘前评估 =============")
        pre_accs, pre_aucs, pre_avg_acc, pre_avg_auc, pre_std_acc, pre_std_auc = evaluate_on_all_clients(
            global_model, server.clients, "遗忘前评估", server=server)
        
        # 保存遗忘前评估结果
        save_evaluation_results(pre_accs, pre_aucs, pre_avg_acc, pre_avg_auc, pre_std_acc, pre_std_auc, 
                              "pre_forget", args)

        # 遗忘前攻击评估
        print("\n============= 遗忘前攻击评估 =============")
        mia_pre = membership_inference_attack(global_model, server.clients[args.target_client_id], args.device)
        print(f"[遗忘前MIA] AUC: {mia_pre['auc']:.4f}, 最佳准确率: {mia_pre['best_acc']:.4f}, 最佳阈值: {mia_pre['best_thr']:.2f}")
        backdoor_acc_pre = backdoor_attack(global_model, server.clients[args.target_client_id], args.device)
        print(f"[遗忘前后门攻击] 后门样本准确率: {backdoor_acc_pre:.4f}")
        # 保存攻击评估结果
        with open(build_result_artifact_path("attack_pre_forget", args, "txt"), "w") as f:
            f.write(f"MIA AUC: {mia_pre['auc']:.4f}\nMIA Best Acc: {mia_pre['best_acc']:.4f}\nMIA Best Thr: {mia_pre['best_thr']:.2f}\n")
            f.write(f"Backdoor Acc: {backdoor_acc_pre:.4f}\n")

        # 遗忘阶段：遗忘客户端{args.target_client_id}
        print(f"\n============= 开始遗忘阶段 =============")
        print(f"目标遗忘客户端: {args.target_client_id}")
        with SuppressStderr():
            if args.forget_strategy == "adversarial":
                from flcore.forgetting import forget_client_wrapper
                global_model_forget = forget_client_wrapper(global_model, server.clients, args, target_client_id=args.target_client_id)
            elif args.forget_strategy == "gradient_reversal":
                from flcore.forgetting import forget_client_with_gradient_reversal
                global_model_forget = forget_client_with_gradient_reversal(global_model, server.clients, args, target_client_id=args.target_client_id, pre_accuracies=pre_accs)
            
            elif args.forget_strategy == "fedcsa":
                from flcore.forgetting import forget_client_with_fedcsa
                global_model_forget = forget_client_with_fedcsa(global_model, server.clients, args, target_client_id=args.target_client_id)

            elif args.forget_strategy == "fedcsa_compare":
                from flcore.fedcsa_compare import forget_client_with_fedcsa_compare
                global_model_forget = forget_client_with_fedcsa_compare(global_model, server.clients, args, target_client_id=args.target_client_id)
            
            elif args.forget_strategy == "fedosd":
                from flcore.forgetting import forget_client_with_fedosd
                global_model_forget = forget_client_with_fedosd(
                    global_model,
                    server.clients,
                    args,
                    target_client_id=args.target_client_id,
                    server=server,
                )
            
            elif args.forget_strategy == "fedu":
                from flcore.forgetting import forget_client_with_fedu
                global_model_forget = forget_client_with_fedu(global_model, server.clients, args, target_client_id=args.target_client_id)
            elif args.forget_strategy == "fedau":
                from flcore.fedau_compare import forget_client_with_fedau
                global_model_forget = forget_client_with_fedau(
                    global_model,
                    server.clients,
                    args,
                    target_client_id=args.target_client_id,
                )
            elif args.forget_strategy == "sifu":
                from flcore.sifu_compare import forget_client_with_sifu
                global_model_forget = forget_client_with_sifu(
                    global_model,
                    server.clients,
                    args,
                    target_client_id=args.target_client_id,
                    server=server,
                )
            else:
                raise ValueError(f"Unknown forgetting strategy: {args.forget_strategy}")

        forget_result = normalize_forgetting_result(global_model_forget)
        global_model_forget = forget_result["model"]
        apply_forgetting_result_side_effects(forget_result, server, args)
        method_recovery_results = list(forget_result.get("recovery_results", []))
        method_recovery_rounds = int(forget_result.get("recovery_rounds", 0) or 0)
        method_recovery_stage = forget_result.get("recovery_stage")
        method_post_recovery_model = forget_result.get("post_recovery_model")
        method_post_recovery_client_buffer_states = forget_result.get("post_recovery_client_buffer_states")
        
        # 保存遗忘后的模型
        forget_model_save_path = build_result_artifact_path("forget_model", args, "pt")
        torch.save(global_model_forget, forget_model_save_path)
        print(f"遗忘后模型已保存到: {forget_model_save_path}")

        # 遗忘后评估：在所有客户端数据上测试准确率
        print("\n============= 遗忘后评估 =============")
        post_accs, post_aucs, post_avg_acc, post_avg_auc, post_std_acc, post_std_auc = evaluate_on_all_clients(
            global_model_forget, server.clients, "遗忘后评估", server=server)
        
        # 保存遗忘后评估结果
        save_evaluation_results(post_accs, post_aucs, post_avg_acc, post_avg_auc, post_std_acc, post_std_auc, 
                              "post_forget", args)

        # 计算遗忘效果（以遗忘前准确率为基准）
        print("\n============= 遗忘效果分析 =============")
        per_client_abs_change = [post_accs[i] - pre_accs[i] for i in range(len(pre_accs))]
        per_client_rel_change = [
            (post_accs[i] - pre_accs[i]) / (pre_accs[i] + 1e-12) for i in range(len(pre_accs))
        ]
        target_abs_change = per_client_abs_change[args.target_client_id]
        target_rel_change = per_client_rel_change[args.target_client_id]
        # 计算其他客户端的平均变化（排除目标客户端）
        other_indices = [i for i in range(len(pre_accs)) if i != args.target_client_id]
        others_abs_change_avg = np.mean([per_client_abs_change[i] for i in other_indices]) if len(other_indices) > 0 else 0.0
        valid_other_rel_changes = [per_client_rel_change[i] for i in other_indices if pre_accs[i] > 1e-6]
        others_rel_change_avg = np.mean(valid_other_rel_changes) if len(valid_other_rel_changes) > 0 else 0.0

        print(f"目标客户端({args.target_client_id}) 遗忘前: {pre_accs[args.target_client_id]:.4f} 遗忘后: {post_accs[args.target_client_id]:.4f}")
        print(f"目标客户端({args.target_client_id}) 变化量: {target_abs_change:.4f} (相对 {target_rel_change*100:.2f}%)")
        print(f"其他客户端 平均变化量: {others_abs_change_avg:.4f} (相对 {others_rel_change_avg*100:.2f}%)")

        # 直接进入恢复阶段（无需阈值判断）
        skip_external_recovery = (
            (args.forget_strategy == "fedosd" and not getattr(args, "fedosd_use_legacy", False))
            or (args.forget_strategy == "fedau" and getattr(args, "fedau_skip_recovery", True))
            or args.forget_strategy == "sifu"
        )
        if args.forget_strategy == "fedosd" and skip_external_recovery:
            recovery_rounds = getattr(args, 'fedosd_recovery_rounds', 0)
        elif args.forget_strategy == "sifu" and skip_external_recovery:
            recovery_rounds = method_recovery_rounds
        else:
            recovery_rounds = 0
        recovery_results = list(method_recovery_results)
        recovery_accs = None
        recovery_aucs = None
        recovery_avg_acc = None
        recovery_avg_auc = None
        recovery_effect = None

        if skip_external_recovery:
            print(f"\n============= 跳过外层恢复阶段 =============")
            if args.forget_strategy == "fedosd":
                print("FedOSD 对比实现已在遗忘函数内部完成 post-training，外层通用恢复不再重复执行。")
            elif args.forget_strategy == "fedau":
                print("FedAU 默认不执行额外服务器恢复阶段，以保持与官方方法更一致。")
            elif args.forget_strategy == "sifu":
                if method_post_recovery_model is not None:
                    print("SIFU 对比实现已在方法内部完成回跳后的恢复训练，外层通用恢复不再重复执行。")
                    server.global_model = method_post_recovery_model
                    restored = restore_client_buffer_states(
                        server,
                        method_post_recovery_client_buffer_states,
                        args.device,
                    )
                    if restored:
                        print(f"已恢复 SIFU 内部恢复后的客户端buffer状态: {restored} 个客户端")

                    recovery_model_save_path = build_result_artifact_path("recovery_model", args, "pt")
                    torch.save(server.global_model, recovery_model_save_path)
                    print(f"恢复后模型已保存到: {recovery_model_save_path}")

                    print("\n============= 恢复后评估 =============")
                    recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc = evaluate_on_all_clients(
                        server.global_model, server.clients, "恢复后评估", server=server)
                    save_evaluation_results(
                        recovery_accs,
                        recovery_aucs,
                        recovery_avg_acc,
                        recovery_avg_auc,
                        recovery_std_acc,
                        recovery_std_auc,
                        "post_recovery",
                        args,
                    )
                    recovery_effect = recovery_avg_acc - post_avg_acc
                    print(f"恢复效果: 平均准确率提升 {recovery_effect:.4f} ({recovery_effect/post_avg_acc*100:.2f}%)")
                    global_model_forget = server.global_model
                else:
                    print("SIFU 已按当前配置跳过恢复阶段。")
        else:
            print(f"\n============= 开始恢复阶段 =============")
            print(f"其他客户端性能变化: {others_rel_change_avg*100:.2f}%")
            
            recovery_rounds = getattr(args, 'recovery_rounds', 5)  # 默认恢复5轮
            print(f"恢复训练轮数: {recovery_rounds}")
            
            # 使用遗忘后的模型进行恢复训练
            server.global_model = global_model_forget
            recovery_results = server.recovery_training(target_client_id=args.target_client_id, recovery_rounds=recovery_rounds)
            
            # 保存恢复后的模型
            recovery_model_save_path = build_result_artifact_path("recovery_model", args, "pt")
            torch.save(server.global_model, recovery_model_save_path)
            print(f"恢复后模型已保存到: {recovery_model_save_path}")
            
            # 恢复后评估
            print("\n============= 恢复后评估 =============")
            recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc = evaluate_on_all_clients(
                server.global_model, server.clients, "恢复后评估", server=server)
            
            # 保存恢复后评估结果
            save_evaluation_results(recovery_accs, recovery_aucs, recovery_avg_acc, recovery_avg_auc, recovery_std_acc, recovery_std_auc, 
                                  "post_recovery", args)
            
            # 计算恢复效果
            recovery_effect = recovery_avg_acc - post_avg_acc
            print(f"恢复效果: 平均准确率提升 {recovery_effect:.4f} ({recovery_effect/post_avg_acc*100:.2f}%)")
            
            # 更新最终模型
            global_model_forget = server.global_model

        # 保存遗忘效果汇总
        forget_summary = {
            "pre_accs": pre_accs,
            "post_accs": post_accs,
            "per_client_abs_change": per_client_abs_change,
            "per_client_rel_change": per_client_rel_change,
            "target_abs_change": target_abs_change,
            "target_rel_change": target_rel_change,
            "others_abs_change_avg": float(others_abs_change_avg),
            "others_rel_change_avg": float(others_rel_change_avg),
            "recovery_rounds": recovery_rounds,
            "recovery_triggered": recovery_rounds > 0,
            "recovery_stage": (
                method_recovery_stage
                if args.forget_strategy == "sifu" and method_recovery_stage
                else
                "internal_fedosd_post_training"
                if args.forget_strategy == "fedosd" and skip_external_recovery
                else "skipped_for_fedau"
                if args.forget_strategy == "fedau" and skip_external_recovery
                else "external_server_recovery"
            )
        }
        if forget_result.get("metadata"):
            forget_summary["method_metadata"] = forget_result["metadata"]
        
        # 添加恢复结果
        forget_summary["recovery_results"] = recovery_results
        if recovery_accs is not None:
            forget_summary["recovery_accs"] = recovery_accs
            forget_summary["recovery_aucs"] = recovery_aucs
            forget_summary["recovery_avg_acc"] = float(recovery_avg_acc)
            forget_summary["recovery_avg_auc"] = float(recovery_avg_auc)
            forget_summary["recovery_effect"] = float(recovery_effect)
        
        forget_summary_path = build_result_artifact_path("forget_effect", args, "json")
        with open(forget_summary_path, "w") as f:
            json.dump(forget_summary, f, indent=2)
        print(f"遗忘效果汇总已保存到: {forget_summary_path}")


        # 遗忘后攻击评估
        print("\n============= 遗忘后攻击评估 =============")
        mia_post = membership_inference_attack(global_model_forget, server.clients[args.target_client_id], args.device)
        print(f"[遗忘后MIA] AUC: {mia_post['auc']:.4f}, 最佳准确率: {mia_post['best_acc']:.4f}, 最佳阈值: {mia_post['best_thr']:.2f}")
        backdoor_acc_post = backdoor_attack(global_model_forget, server.clients[args.target_client_id], args.device)
        print(f"[遗忘后后门攻击] 后门样本准确率: {backdoor_acc_post:.4f}")
        # 保存攻击评估结果
        with open(build_result_artifact_path("attack_post_forget", args, "txt"), "w") as f:
            f.write(f"MIA AUC: {mia_post['auc']:.4f}\nMIA Best Acc: {mia_post['best_acc']:.4f}\nMIA Best Thr: {mia_post['best_thr']:.2f}\n")
            f.write(f"Backdoor Acc: {backdoor_acc_post:.4f}\n")

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="ResNet18")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg", choices=["FedAvg", "SCAFFOLD"])
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0, help="Server learning rate for SCAFFOLD aggregation")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=str2bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # 添加随机种子参数
    parser.add_argument('-seed', "--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--result_tag', type=str, default="",
                        help="Optional suffix for result artifacts to avoid overwriting other experiment runs")
    # 添加遗忘策略选择参数
    parser.add_argument('-fst', "--forget_strategy", type=str, default="gradient_reversal",
                        choices=["adversarial", "gradient_reversal", "fedcsa", "fedcsa_compare", "fedosd", "fedu", "fedau", "sifu"],
                        help="Forgetting strategy: adversarial, gradient_reversal, fedcsa, fedcsa_compare, fedosd, fedu, fedau, or sifu")
    
    # FedCSA参数
    parser.add_argument('--fedcsa_forget_epochs', type=int, default=5,
                        help="Number of epochs for FedCSA forgetting")
    parser.add_argument('--fedcsa_probe_epochs', type=int, default=1,
                        help="Local epochs for FedCSA comparison probing")
    parser.add_argument('--fedcsa_probe_batches', type=int, default=4,
                        help="Max batches per client when probing FedCSA comparison features")
    parser.add_argument('--fedcsa_slice_rounds', type=int, default=3,
                        help="Training rounds used to build FedCSA comparison slice submodels")
    parser.add_argument('--fedcsa_slice_local_epochs', type=int, default=1,
                        help="Local epochs per client inside each FedCSA comparison slice-training round")
    parser.add_argument('--fedcsa_slice_batches', type=int, default=0,
                        help="Max batches per client inside FedCSA comparison slice training; 0 means full epoch")
    parser.add_argument('--fedcsa_compare_min_clusters', type=int, default=2,
                        help="Minimum cluster count for FedCSA comparison")
    parser.add_argument('--fedcsa_compare_max_clusters', type=int, default=5,
                        help="Maximum cluster count for FedCSA comparison")
    parser.add_argument('--fedcsa_cache_enabled', type=str2bool, default=True,
                        help="Whether to cache prepared FedCSA comparison slice models")
    parser.add_argument('--fedcsa_force_rebuild', type=str2bool, default=False,
                        help="Whether to rebuild FedCSA comparison cache even if a cache file exists")
    parser.add_argument('--fedcsa_compare_use_scaffold', type=str2bool, default=True,
                        help="Whether to use SCAFFOLD control variates during FedCSA comparison local updates")
    
    # FedOSD参数
    parser.add_argument('--fedosd_lr', type=float, default=0.0004,
                        help="Learning rate for FedOSD forgetting")
    parser.add_argument('--fedosd_use_legacy', type=str2bool, default=False,
                        help="Use the previous direct-gradient FedOSD approximation")
    parser.add_argument('--fedosd_unlearn_rounds', type=int, default=20,
                        help="Number of official-like FedOSD unlearning rounds")
    parser.add_argument('--fedosd_recovery_rounds', type=int, default=1,
                        help="Number of official-like FedOSD post-training rounds")
    parser.add_argument('--fedosd_recovery_lr', type=float, default=1e-6,
                        help="Learning rate for official-like FedOSD post-training")
    parser.add_argument('--fedosd_force_target_online', type=str2bool, default=True,
                        help="Ensure the target client participates in every FedOSD unlearning round")

    # FedAU参数
    parser.add_argument('--fedau_aux_lr', type=float, default=0.0,
                        help="Legacy compatibility argument; official-like FedAU uses the main local learning rate")
    parser.add_argument('--fedau_alpha', type=float, default=0.9,
                        help="Whole-client FedAU auxiliary-head coefficient alpha in W_hat = (1-alpha) * W_l + alpha * W_a")
    parser.add_argument('--fedau_client_gamma', type=float, default=0.5,
                        help="FedAU whole-client head blending factor used before each local epoch")
    parser.add_argument('--fedau_skip_recovery', type=str2bool, default=True,
                        help="Skip the extra server-side recovery stage for FedAU to stay closer to the official method")
    parser.add_argument('--fedau_forget_scale', type=float, default=1.0,
                        help="Legacy compatibility argument for the previous simplified FedAU implementation")

    # SIFU参数
    parser.add_argument('--sifu_epsilon', type=float, default=10.0,
                        help="Privacy/unlearning budget epsilon used by SIFU")
    parser.add_argument('--sifu_sigma', type=float, default=0.05,
                        help="Noise std used by SIFU")
    parser.add_argument('--sifu_lambd', type=float, default=0.0,
                        help="Regularization coefficient used in the SIFU psi recursion")
    parser.add_argument('--sifu_skip_recovery', type=str2bool, default=False,
                        help="Skip the recovery stage inside the SIFU comparison method")
    parser.add_argument('--sifu_recovery_rounds', type=int, default=-1,
                        help="Recovery rounds run inside SIFU; negative values replay the remaining original training rounds")

    # FedU参数
    parser.add_argument('--fedu_forget_epochs', type=int, default=1,
                        help="Number of epochs for FedU forgetting")
    parser.add_argument('--fedu_lr', type=float, default=0.003,
                        help="Learning rate for FedU forgetting")
    parser.add_argument('--fedu_eps', type=float, default=0.005,
                        help="Epsilon for FedU Hessian computation")
    parser.add_argument('--fedu_alpha', type=float, default=0.02,
                        help="Alpha for FedU update")
    parser.add_argument('--fedu_erased_ratio', type=float, default=0.35 ,
                        help="Ratio of target-client local samples treated as erased set in FedU-main style")
    parser.add_argument('--fedu_erased_max_samples', type=int, default=1200,
                        help="Upper bound of erased samples (0 means no cap)")
    parser.add_argument('--fedu_split_seed', type=int, default=42,
                        help="Random seed for erased/remaining split")
    parser.add_argument('--fedu_forget_rounds', type=int, default=12,
                        help="FedU-main style local forgetting rounds on target client")
    parser.add_argument('--fedu_local_steps', type=int, default=6,
                        help="Inner local steps per FedU forgetting round")
    parser.add_argument('--fedu_hutchinson_samples', type=int, default=8,
                        help="Number of Hutchinson samples for Hessian diagonal approximation")
    parser.add_argument('--fedu_utility_scale', type=float, default=0.35,
                        help="Scale for utility-preservation gradient in FedU dual objective")
    # 添加加载已保存模型参数
    parser.add_argument('-lsm', "--load_saved_model", type=str2bool, default=False,
                        help="Whether to load a saved model instead of training")
    parser.add_argument('-smp', "--saved_model_path", type=str, default="/home/siguangchen/zyj/FUcopy/system/results/exp_configs/global_model_Cifar10_FedAvg_test_1_fedau.pt",
                        help="Path to the saved model file to load; empty means resolving the default path from current args")
    # 添加调试参数
    parser.add_argument('-dml', "--debug_model_loading", type=str2bool, default=False,
                        help="Enable debug mode for model loading")
    # 添加恢复阶段参数
    parser.add_argument('-rr', "--recovery_rounds", type=int, default=5,
                        help="Number of rounds for recovery training")
    
    # 添加优化选项参数
    parser.add_argument('-opt', "--use_optimized_forgetting", type=str2bool, default=True,
                        help="Whether to use optimized forgetting algorithm to reduce computation cost")
    parser.add_argument('-sr', "--sample_ratio", type=float, default=0.3,
                        help="Sample ratio for gradient computation (0.0-1.0)")
    parser.add_argument('-mb', "--max_batches", type=int, default=8,
                        help="Maximum number of batches for gradient computation")
    parser.add_argument('-np', "--num_processes", type=int, default=4,
                        help="Number of processes for parallel gradient computation")
    parser.add_argument('--fu_retain_calibration_rounds', type=int, default=2,
                        help="Extra masked retain-calibration rounds inside FU after gradient reversal")
    parser.add_argument('--fu_retain_calibration_lr', type=float, default=4e-4,
                        help="Learning rate for FU retain calibration")
    parser.add_argument('--fu_retain_calibration_batches', type=int, default=3,
                        help="Max batches per retain client in each FU retain calibration round")
    parser.add_argument('--fu_mask_retain_scale', type=float, default=0.12,
                        help="Gradient scale kept on FU forget-mask parameters during retain calibration")
    parser.add_argument('--fu_similarity_boost', type=float, default=1.2,
                        help="Protection weight boost for retain clients close to the target client")
    parser.add_argument('--protection_level', type=str, default='weak',
                        choices=['strong', 'moderate', 'weak'],
                        help="Protection strength for retain clients inside FU")
    
    # 添加目标客户端ID参数
    parser.add_argument('-tci', "--target_client_id", type=int, default=5,
                        help="Target client ID to forget (0-based index)")
    # 是否只做“重训练基线”（在训练阶段排除目标客户端，不执行遗忘算法）
    parser.add_argument('-rt', "--retrain_only", type=str2bool, default=False,
                        help="If True, only retrain from scratch excluding target client, without running forgetting algorithm")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    run(args)
