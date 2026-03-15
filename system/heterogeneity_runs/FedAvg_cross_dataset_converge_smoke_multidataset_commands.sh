#!/usr/bin/env bash
set -euo pipefail

cd /home/siguangchen/zyj/FUcopy1

env -u LD_LIBRARY_PATH /home/siguangchen/anaconda3/envs/zyj/bin/python -u system/run_heterogeneity_comparison.py --dataset-base STL10 --dataset-prefix crossds_converge_smoke --experiment-name cross_dataset_converge_smoke --goal test --algorithm FedAvg --model ResNet18 --device cuda --device-id 0 --python /home/siguangchen/anaconda3/envs/zyj/bin/python --levels severe --methods fu --partition dir --num-clients 10 --num-classes 10 --global-rounds 120 --top-cnt 20 --local-epochs 1 --batch-size 64 --local-learning-rate 0.005 --learning-rate-decay true --learning-rate-decay-gamma 0.995 --join-ratio 1.0 --times 1 --eval-gap 10 --auto-break true --target-client-id 5 --seed 42 --recovery-rounds 5 --max-batches 8 --num-processes 4 --alpha-mild 1.0 --alpha-moderate 0.3 --alpha-severe 0.1 --class-per-client-mild 5 --class-per-client-moderate 3 --class-per-client-severe 2 --balance
