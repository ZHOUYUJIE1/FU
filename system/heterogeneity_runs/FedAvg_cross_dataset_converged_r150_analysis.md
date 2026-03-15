# 跨数据集对比分析（FedAvg）

- 实验名称: `cross_dataset_converged_r150`
- 数据集: STL-10, Tiny-ImageNet
- 方法: fu, fedau, fedcsa, fedosd, retrain
- 层级: severe

## 按数据集结论

### STL-10 / severe
- 最佳可用性方法: `fedcsa`，`final_avg_acc=0.3412`。
- 最强遗忘方法: `retrain`，`final_target_acc=0.0030`。
- FU 指标: `final_avg_acc=0.3377`，`final_target_acc=0.0788`。

### Tiny-ImageNet / severe
- 最佳可用性方法: `fedau`，`final_avg_acc=0.1682`。
- 最强遗忘方法: `retrain`，`final_target_acc=0.0000`。
- FU 指标: `final_avg_acc=0.1369`，`final_target_acc=0.0110`。

## 方法总体趋势

- `fedcsa`: 平均 `final_avg_acc=0.2523`，平均 `final_target_acc=0.1316`（样本数=2）。
- `fedau`: 平均 `final_avg_acc=0.2414`，平均 `final_target_acc=0.0872`（样本数=2）。
- `retrain`: 平均 `final_avg_acc=0.2401`，平均 `final_target_acc=0.0015`（样本数=2）。
- `fu`: 平均 `final_avg_acc=0.2373`，平均 `final_target_acc=0.0449`（样本数=2）。
- `fedosd`: 平均 `final_avg_acc=0.2152`，平均 `final_target_acc=0.0657`（样本数=2）。