# FU Hyperparameter Sensitivity Summary

## Setup

- dataset: `Cifar10_hetero_clean_s321_severe`
- seed: `321`
- target_client_id: `5`
- saved_model_path: `/home/siguangchen/zyj/FUcopy1/results/exp_configs/global_model_Cifar10_hetero_clean_s321_severe_FedAvg_test_1_fu_severe.pt`
- default lambda_reversal: `0.3`
- default initial_mask_ratio: `0.1`

## 基础反转系数

| Value | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc | Gap | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1500 | 0.7307 | 0.0500 | 0.8063 | 0.7496 | 0.4814 | 0.1979 | 0.7563 | complete |
| 0.2000 | 0.7307 | 0.0473 | 0.8067 | 0.7510 | 0.4891 | 0.1771 | 0.7593 | complete |
| 0.3000 | 0.7315 | 0.0480 | 0.8074 | 0.7518 | 0.4985 | 0.1771 | 0.7594 | complete |
| 0.4000 | 0.7307 | 0.0480 | 0.8065 | 0.7517 | 0.4971 | 0.1823 | 0.7585 | complete |
| 0.5000 | 0.7301 | 0.0473 | 0.8060 | 0.7513 | 0.5028 | 0.1875 | 0.7587 | complete |

### Analysis

- 最强遗忘点: `0.2`，目标客户端准确率 `0.0473`，保留客户端平均准确率 `0.8067`。
- 最佳保留性能点: `0.3`，保留客户端平均准确率 `0.8074`，目标客户端准确率 `0.0480`。
- 综合折中最优点: `0.3`，`retain-target gap = 0.7594`。
- 默认值 `0.3` 的基线结果: target `0.0480`，retain `0.8074`，MIA AUC `0.4985`。
- 遗忘趋势: 随着参数增大，目标客户端准确率整体下降，遗忘更强，趋势中等。
- 保留趋势: 保留客户端性能随参数变化有波动，但单调趋势较弱。

## 初始掩码比例

| Value | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc | Gap | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0500 | 0.7399 | 0.0773 | 0.8135 | 0.7589 | 0.4941 | 0.2552 | 0.7361 | complete |
| 0.0800 | 0.7340 | 0.0593 | 0.8090 | 0.7570 | 0.4998 | 0.2188 | 0.7496 | complete |
| 0.1000 | 0.7315 | 0.0480 | 0.8074 | 0.7518 | 0.4985 | 0.1771 | 0.7594 | complete |
| 0.1200 | 0.7275 | 0.0393 | 0.8040 | 0.7456 | 0.5060 | 0.1562 | 0.7647 | complete |
| 0.1500 | 0.7185 | 0.0313 | 0.7949 | 0.7314 | 0.5037 | 0.1510 | 0.7636 | complete |

### Analysis

- 最强遗忘点: `0.15`，目标客户端准确率 `0.0313`，保留客户端平均准确率 `0.7949`。
- 最佳保留性能点: `0.05`，保留客户端平均准确率 `0.8135`，目标客户端准确率 `0.0773`。
- 综合折中最优点: `0.12`，`retain-target gap = 0.7647`。
- 默认值 `0.1` 的基线结果: target `0.0480`，retain `0.8074`，MIA AUC `0.4985`。
- 遗忘趋势: 随着参数增大，目标客户端准确率整体下降，遗忘更强，趋势较强。
- 保留趋势: 随着参数增大，保留客户端性能整体下降，趋势较强。
