#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321cc_c50_severe --num_clients 50 --eval_gap 10 --mia_pre_model_source global_model --random_seed 321 --result_tag fu_severe --forget_strategy gradient_reversal --enable_backdoor_attack false
env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_clean_s321cc_c50_severe --num_clients 50 --eval_gap 10 --mia_pre_model_source global_model --random_seed 321 --result_tag retrain_severe --forget_strategy gradient_reversal --enable_backdoor_attack false --retrain_only true
