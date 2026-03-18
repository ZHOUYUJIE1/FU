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
    cfg.setdefault("mia_mode", "target_lr")
    cfg.setdefault("mia_reference_mode", "target_train_vs_other_train")
    cfg.setdefault("mia_pre_model_source", "target_local_last")
    cfg.setdefault("mia_client_oriented", False)
    cfg.setdefault("attack_group_by_client", False)
    cfg.setdefault("enable_backdoor_attack", False)
    cfg.setdefault("backdoor_client_oriented", False)
    return cfg


def _load_checkpoint_into_server(server, args, model_path):
    loaded_data = torch.load(model_path, map_location=args.device)
    client_buffer_states, client_buffer_source = main.load_client_buffer_states_from_checkpoint(
        model_path,
        loaded_data,
        map_location=args.device,
    )

    if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
        main.load_model_state_with_local_only_support(server.global_model, loaded_data["model_state_dict"])
        if args.algorithm == "SCAFFOLD" and hasattr(server, "global_c"):
            if "global_c" in loaded_data:
                server.global_c = [c.to(args.device) for c in loaded_data["global_c"]]
            if "client_c" in loaded_data:
                for client in server.clients:
                    if hasattr(client, "client_c") and client.id in loaded_data["client_c"]:
                        client.client_c = [c.to(args.device) for c in loaded_data["client_c"][client.id]]
    elif hasattr(loaded_data, "state_dict") and callable(loaded_data.state_dict):
        main.load_model_state_with_local_only_support(server.global_model, loaded_data.state_dict())
    else:
        main.load_model_state_with_local_only_support(server.global_model, loaded_data)

    server.send_models()

    if client_buffer_states:
        restored = main.restore_client_buffer_states(server, client_buffer_states, args.device)
        print(f"restored_client_buffers = {restored}")
        print(f"client_buffer_source = {client_buffer_source}")
    else:
        print("restored_client_buffers = 0")
        print("client_buffer_source = None")


def run_saved_attack_eval(
    config_path,
    model_path,
    result_tag,
    stage_name,
    display_name=None,
    target_client_id=None,
    forget_strategy=None,
    mia_client_oriented=None,
    mia_mode=None,
):
    cfg = _load_args_from_config(config_path)
    if target_client_id is not None:
        cfg["target_client_id"] = int(target_client_id)
    if forget_strategy is not None:
        cfg["forget_strategy"] = str(forget_strategy)
    if mia_client_oriented is not None:
        cfg["mia_client_oriented"] = bool(mia_client_oriented)
    if mia_mode is not None:
        cfg["mia_mode"] = str(mia_mode)
    cfg["result_tag"] = result_tag

    args = types.SimpleNamespace(**cfg)
    if display_name is None:
        display_name = stage_name

    print("=" * 50)
    print(f"config_path = {config_path}")
    print(f"model_path = {model_path}")
    print(f"dataset = {args.dataset}")
    print(f"result_tag = {args.result_tag}")
    print(f"stage_name = {stage_name}")
    print(f"display_name = {display_name}")
    print(f"target_client_id = {args.target_client_id}")
    print(f"mia_mode = {args.mia_mode}")
    print("=" * 50)

    server = main.create_server(args, args.times, args.model)
    _load_checkpoint_into_server(server, args, model_path)

    args._attack_eval_clients = server.clients
    attack_reference = main._build_target_attack_reference(
        args,
        server.clients[args.target_client_id],
        eval_clients=server.clients,
    )

    mia_model = None
    mia_model_source = "global_model"
    if stage_name.startswith("pre"):
        mia_model, mia_model_source = main._load_target_local_model_last_for_mia(
            model_path,
            server,
            args,
        )

    main.run_attack_evaluation(
        server.global_model,
        server.clients[args.target_client_id],
        args,
        stage_name=stage_name,
        display_name=display_name,
        attack_reference=attack_reference,
        mia_model=mia_model,
        mia_model_source=mia_model_source,
    )


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {"true", "1", "yes", "y", "on"}:
            return True
        if value in {"false", "0", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = argparse.ArgumentParser(description="Run attack evaluation on a saved model checkpoint.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--target-client-id", type=int, default=None)
    parser.add_argument("--forget-strategy", default=None)
    parser.add_argument("--mia-client-oriented", type=str2bool, default=None)
    parser.add_argument("--mia-mode", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_saved_attack_eval(
        config_path=cli_args.config_path,
        model_path=cli_args.model_path,
        result_tag=cli_args.result_tag,
        stage_name=cli_args.stage_name,
        display_name=cli_args.display_name,
        target_client_id=cli_args.target_client_id,
        forget_strategy=cli_args.forget_strategy,
        mia_client_oriented=cli_args.mia_client_oriented,
        mia_mode=cli_args.mia_mode,
    )
