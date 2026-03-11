#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_mild --result_tag fu_mild --forget_strategy gradient_reversal
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_mild --result_tag fedau_mild --forget_strategy fedau
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_mild --result_tag fedcsa_mild --forget_strategy fedcsa --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_mild_FedAvg_test_1_fu_mild.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_mild --result_tag fedosd_mild --forget_strategy fedosd --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_mild_FedAvg_test_1_fu_mild.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_mild --result_tag retrain_mild --forget_strategy gradient_reversal --retrain_only true
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_moderate --result_tag fu_moderate --forget_strategy gradient_reversal
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_moderate --result_tag fedau_moderate --forget_strategy fedau
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_moderate --result_tag fedcsa_moderate --forget_strategy fedcsa --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_moderate_FedAvg_test_1_fu_moderate.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_moderate --result_tag fedosd_moderate --forget_strategy fedosd --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_moderate_FedAvg_test_1_fu_moderate.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_moderate --result_tag retrain_moderate --forget_strategy gradient_reversal --retrain_only true
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --result_tag fu_severe --forget_strategy gradient_reversal
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --result_tag fedau_severe --forget_strategy fedau
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --result_tag fedcsa_severe --forget_strategy fedcsa --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --result_tag fedosd_severe --forget_strategy fedosd --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --result_tag retrain_severe --forget_strategy gradient_reversal --retrain_only true
