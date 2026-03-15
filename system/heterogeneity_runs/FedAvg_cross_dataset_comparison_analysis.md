# 跨数据集对比分析（FedAvg）

- 实验名称: `cross_dataset_comparison`
- 数据集: STL-10, Tiny-ImageNet
- 方法: fu, fedau, fedcsa, fedosd, retrain
- 层级: severe

## 按数据集结论

### STL-10 / severe
- 最佳可用性方法: `fedcsa`，`final_avg_acc=0.1284`。
- 最强遗忘方法: `fu`，`final_target_acc=0.0006`。
- FU 指标: `final_avg_acc=0.1151`，`final_target_acc=0.0006`。

### Tiny-ImageNet / severe
- 最佳可用性方法: `fedcsa`，`final_avg_acc=0.0078`。
- 最强遗忘方法: `fedosd`，`final_target_acc=0.0000`。
- FU 指标: `final_avg_acc=0.0054`，`final_target_acc=0.0001`。

## 方法总体趋势

- `fedcsa`: 平均 `final_avg_acc=0.0681`，平均 `final_target_acc=0.0205`（样本数=2）。
- `fedau`: 平均 `final_avg_acc=0.0604`，平均 `final_target_acc=0.0494`（样本数=2）。
- `retrain`: 平均 `final_avg_acc=0.0604`，平均 `final_target_acc=0.0054`（样本数=2）。
- `fu`: 平均 `final_avg_acc=0.0603`，平均 `final_target_acc=0.0003`（样本数=2）。
- `fedosd`: 平均 `final_avg_acc=0.0601`，平均 `final_target_acc=0.0003`（样本数=2）。