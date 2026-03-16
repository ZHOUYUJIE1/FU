#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset Cifar10_hetero_severe --eval_gap 10 --result_tag fu_severe --forget_strategy gradient_reversal
