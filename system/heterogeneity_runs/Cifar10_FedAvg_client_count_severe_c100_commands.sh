#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_clientscale_c100_severe --batch_size 16 --num_clients 100 --result_tag fedosd_severe --forget_strategy fedosd --fedosd_max_online_clients 20 --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_clientscale_c100_severe_FedAvg_test_1_fu_severe.pt
