#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset TinyImagenet_hetero_clean_s42_severe --num_classes 200 --sgd_momentum 0.9 --weight_decay 0.0005 --result_tag fu_severe --forget_strategy gradient_reversal --enable_backdoor_attack false
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset TinyImagenet_hetero_clean_s42_severe --num_classes 200 --sgd_momentum 0.9 --weight_decay 0.0005 --result_tag fedau_severe --forget_strategy fedau --enable_backdoor_attack false
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset TinyImagenet_hetero_clean_s42_severe --num_classes 200 --sgd_momentum 0.9 --weight_decay 0.0005 --result_tag fedcsa_severe --forget_strategy fedcsa --enable_backdoor_attack false --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_TinyImagenet_hetero_clean_s42_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset TinyImagenet_hetero_clean_s42_severe --num_classes 200 --sgd_momentum 0.9 --weight_decay 0.0005 --result_tag fedosd_severe --forget_strategy fedosd --enable_backdoor_attack false --load_saved_model true --saved_model_path /home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_TinyImagenet_hetero_clean_s42_severe_FedAvg_test_1_fu_severe.pt
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset TinyImagenet_hetero_clean_s42_severe --num_classes 200 --sgd_momentum 0.9 --weight_decay 0.0005 --mia_pre_model_source global_model --result_tag retrain_severe --forget_strategy gradient_reversal --enable_backdoor_attack false --retrain_only true
