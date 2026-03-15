#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Digit5_supp_d5_hetero_severe --model CNN --num_clients 5 --result_tag fu_severe --forget_strategy gradient_reversal --target_client_id 0
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Digit5_supp_d5_hetero_severe --model CNN --num_clients 5 --result_tag fedau_severe --forget_strategy fedau --target_client_id 0
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Digit5_supp_d5_hetero_severe --model CNN --num_clients 5 --result_tag fedcsa_severe --forget_strategy fedcsa --target_client_id 0 --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Digit5_supp_d5_hetero_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Digit5_supp_d5_hetero_severe --model CNN --num_clients 5 --result_tag fedosd_severe --forget_strategy fedosd --target_client_id 0 --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Digit5_supp_d5_hetero_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Digit5_supp_d5_hetero_severe --model CNN --num_clients 5 --result_tag retrain_severe --forget_strategy gradient_reversal --target_client_id 0 --retrain_only true
