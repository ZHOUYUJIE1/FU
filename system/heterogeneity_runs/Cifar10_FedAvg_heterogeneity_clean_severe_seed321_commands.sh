#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321_severe --global_rounds 10 --random_seed 321 --result_tag fu_severe --forget_strategy gradient_reversal
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321_severe --global_rounds 10 --random_seed 321 --result_tag fedau_severe --forget_strategy fedau
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321_severe --global_rounds 10 --random_seed 321 --result_tag fedcsa_severe --forget_strategy fedcsa --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_clean_s321_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321_severe --global_rounds 10 --random_seed 321 --result_tag fedosd_severe --forget_strategy fedosd --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_clean_s321_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321_severe --global_rounds 10 --random_seed 321 --result_tag retrain_severe --forget_strategy gradient_reversal --retrain_only true
