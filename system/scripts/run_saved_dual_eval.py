import argparse
import json
import os
import sys
import types

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.dirname(SCRIPT_DIR)
if SYSTEM_DIR not in sys.path:
    sys.path.insert(0, SYSTEM_DIR)

import main


def _load_args_from_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("target_client_id", 5)
    cfg.setdefault("times", 1)
    cfg.setdefault("goal", "test")
    cfg.setdefault("algorithm", "FedAvg")
    cfg.setdefault("model", "ResNet18")
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.setdefault("device_id", "0")
    cfg.setdefault("sgd_momentum", None)
    cfg.setdefault("weight_decay", None)
    return cfg


def run_saved_dual_eval(config_path, model_path, result_tag, stage_name, target_client_id=None):
    cfg = _load_args_from_config(config_path)
    if target_client_id is not None:
        cfg["target_client_id"] = int(target_client_id)
    cfg["result_tag"] = result_tag

    args = types.SimpleNamespace(**cfg)

    print("=" * 50)
    print(f"config_path = {config_path}")
    print(f"model_path = {model_path}")
    print(f"dataset = {args.dataset}")
    print(f"result_tag = {args.result_tag}")
    print(f"stage_name = {stage_name}")
    print(f"target_client_id = {args.target_client_id}")
    print("=" * 50)

    server = main.create_server(args, args.times, args.model)

    state_dict = torch.load(model_path, map_location=args.device)
    main.load_model_state_with_local_only_support(server.global_model, state_dict)
    server.send_models()

    client_buffer_states, client_buffer_source = main.load_client_buffer_states_from_checkpoint(
        model_path,
        state_dict,
        map_location="cpu",
    )
    if client_buffer_states:
        restored = main.restore_client_buffer_states(server, client_buffer_states, args.device)
        print(f"restored_client_buffers = {restored}")
        print(f"client_buffer_source = {client_buffer_source}")
    else:
        print("restored_client_buffers = 0")
        print("client_buffer_source = None")

    accs, aucs, avg_acc, avg_auc, std_acc, std_auc = main.evaluate_on_all_clients(
        server.global_model,
        server.clients,
        stage_name=stage_name,
        server=server,
        target_client_id=args.target_client_id,
    )
    main.save_evaluation_results(
        accs,
        aucs,
        avg_acc,
        avg_auc,
        std_acc,
        std_auc,
        stage_name,
        args,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run dual-buffer evaluation on a saved FU model.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--stage-name", default="dual_eval_rerun")
    parser.add_argument("--target-client-id", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_saved_dual_eval(
        config_path=cli_args.config_path,
        model_path=cli_args.model_path,
        result_tag=cli_args.result_tag,
        stage_name=cli_args.stage_name,
        target_client_id=cli_args.target_client_id,
    )
