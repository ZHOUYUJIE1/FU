# FedAU 检查/修正日志

## 1. 检查范围
- `system/flcore/fedau_compare.py`
- `system/flcore/clients/clientfedau.py`
- `system/flcore/trainmodel/models.py`
- `system/flcore/servers/serverbase.py`
- `system/flcore/servers/serveravg.py`
- `system/flcore/servers/serverscaffold.py`
- `system/main.py`（FedAU 参数与调用入口）

## 2. 当前实现概述
- 训练阶段：`FedAUHeadSplit` 使用 `base + head + head_ul`，其中 `head_ul` 标记为本地参数（`local_only_param_prefixes = ("head_ul.",)`）。
- 客户端阶段：目标客户端用主头真标签 + 辅助头错标签进行联合训练；非目标客户端按常规 CE 训练。
- 聚合阶段：服务端聚合时跳过本地参数前缀（`head_ul.*`），保证辅助头不进入全局聚合。
- 遗忘阶段（修正前）：仅执行 `head <- (1-alpha)*global_head + alpha*target_head_ul`。

## 3. FedAU 核心思想理解
> 说明：仓库内未提供 FedAU 论文正文与公式推导，本节为“基于代码注释/命名的推断”，存在不确定性。

- 从参数说明 `W_hat = (1-alpha) * W_l + alpha * W_a` 可推断：
  - `W_l` 应为目标客户端本地主头（local/main head）；
  - `W_a` 应为目标客户端辅助遗忘头（aux/unlearning head）。
- 因此遗忘时更合理的融合应在目标客户端本地头与辅助头之间进行，而不是以全局头代替 `W_l`。

## 4. 一致/不一致项列表

### 4.1 一致项
- 一致：辅助头 `head_ul` 仅本地保存，不参与服务端聚合。
- 一致：目标客户端训练使用辅助头机制，非目标客户端走正常训练。
- 一致：遗忘入口对 `head_ul` 进行显式使用，且与 `alpha` 融合系数相连。

### 4.2 不一致项（修正前）
1. 不一致点
- 遗忘融合使用的是“全局主头 + 目标辅助头”，即 `global_head` 充当 `W_l`。

风险
- 当目标客户端本地主头与全局头存在偏移时，遗忘更新可能偏离 FedAU 机制，导致：
  - 目标遗忘强度不稳定；
  - 不同异构强度下结果可解释性变差；
  - 与“FedAU名义机制”的对齐性不足。

最小修正方案
- 增加 `fedau_mode: legacy | corrected`：
  - `legacy`：保持旧行为；
  - `corrected`：按 `W_hat = W_l + alpha * (W_a - W_l)`，其中 `W_l` 来自目标客户端本地主头。

## 5. 修正前后行为差异
- 修正前（legacy）：`head <- global_head + alpha * (target_aux_head - global_head)`。
- 修正后（corrected）：`head <- target_local_head + alpha * (target_aux_head - target_local_head)`。
- 两者均保留 `fedau_forget_scale`，最终形式为：`base + alpha * forget_scale * (aux - base)`。

## 6. 兼容选项说明
- 新增参数：`--fedau_mode {legacy,corrected}`。
- 默认值：`legacy`。
- 默认设为 `legacy` 的原因：
  - 最小破坏历史实验结果与已保存对比表；
  - 保证旧脚本/旧 checkpoint 流程不变。
- 当需要更贴近 FedAU 注释公式时，显式设为 `--fedau_mode corrected`。

## 7. 风险与仍未解决的问题
- 风险1：由于缺少论文原文对照，本次“核心思想”判断带推断性质。
- 风险2：`fedau_client_gamma` 与错标签构造策略（固定错标签）是否与论文完全一致，仓库证据不足。
- 风险3：当前 backdoor/MIA 的 client-oriented 评估为扩展能力，不影响默认路径，但会增加评估耗时。

## 8. 建议在实验报告中的描述
- 建议明确写出：
  - “为保证历史结果复现，FedAU 默认沿用 legacy 实现；”
  - “同时提供 corrected 分支，以更贴近仓库注释中的 `W_hat=(1-alpha)W_l+alpha W_a` 机制。”
- 报告中建议同时给出 `legacy` 与 `corrected` 对比（至少目标客户端准确率下降、retained 平均准确率、MIA AUC、backdoor ASR）。
- 若仅报告一种模式，应在方法章节显式注明 `fedau_mode` 取值。
