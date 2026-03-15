#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/main.py --dataset STL10_crossds_paper_sanity2_severe --batch_size 64 --global_rounds 10 --result_tag fu_severe --forget_strategy gradient_reversal
