# LB/GB Paper Assets

## 产出文件

- 主表 CSV: `paper_lbgb_main_summary.csv`
- 主表 LaTeX: `paper_lbgb_main_table.tex`
- 主图 PNG/PDF: `paper_lbgb_main_metrics.png`, `paper_lbgb_main_metrics.pdf`
- Trade-off 图 PNG/PDF: `paper_lbgb_tradeoff.png`, `paper_lbgb_tradeoff.pdf`
- 严重异质性跨数据集表: `paper_lbgb_cross_dataset_summary.csv`, `paper_lbgb_cross_dataset_table.tex`

## 关键观察（中文）

- CIFAR-10 mild 下，FU 与 Retrain 最近，距离仅 0.130。FU 的遗忘后 LB 目标客户端准确率为 0.151，GB 保留客户端平均准确率为 0.791。
- CIFAR-10 moderate 下，与 Retrain 最近的方法是 FedOSD，其距离为 0.068。这一设置下，方法间差异更多体现在保留效用而非 MIA。
- CIFAR-10 severe 下，FU 的 LB 目标客户端准确率从 0.496 降到 0.067，同时 GB 保留客户端平均准确率达到 0.598，与 Retrain 的距离仅 0.078。
- 同样在 CIFAR-10 severe 下，FedAU 的目标压制更强（0.033），但 GB 保留客户端平均准确率只有 0.324，明显低于 FU 的 0.598。
- MIA AUC 在多数设置中都集中在 0.48 到 0.50 左右，说明这些方法的差距主要体现在忘记-效用权衡，而不是形成非常大的隐私攻击差距。
- 在这些实验里，FU、FedCSA 和 FedOSD 共享完全相同的遗忘前 checkpoint，因此它们的 `LB Target Before` 与 `GB Global Before` 可以直接横向比较；FedAU 由于需要训练期缓存，必须单独训练，所以其遗忘前指标应单独解释。

## 跨数据集观察（中文）

- CIFAR-10 severe 下，按与 Retrain 的二维距离衡量，最接近 Retrain 的非重训练方法是 FU（距离 0.078，LB 目标准确率 0.067，GB 保留客户端平均准确率 0.598）。
- GTSRB severe 下，按与 Retrain 的二维距离衡量，最接近 Retrain 的非重训练方法是 FedAU（距离 0.121，LB 目标准确率 0.008，GB 保留客户端平均准确率 0.837）。
- STL10 severe 下，按与 Retrain 的二维距离衡量，最接近 Retrain 的非重训练方法是 FedAU（距离 0.145，LB 目标准确率 0.062，GB 保留客户端平均准确率 0.329）。
- Tiny-ImageNet severe 下，按与 Retrain 的二维距离衡量，最接近 Retrain 的非重训练方法是 FedAU（距离 0.066，LB 目标准确率 0.066，GB 保留客户端平均准确率 0.317）。

## 论文主文可直接使用的中文表述

在 CIFAR-10 的 mild, moderate 和 severe 异质性设置下，我们采用 local buffer 评估目标客户端遗忘效果，并采用 global buffer 评估遗忘后保留客户端的整体效用。结果表明，不同方法在“遗忘强度”和“保留效用”之间存在明显权衡。在 mild 设置下，FU 将目标客户端的 local-buffer 准确率降低到 0.151，同时保持 0.791 的 global-buffer 保留客户端平均准确率，是非重训练方法中最接近 Retrain 的方案。在 moderate 设置下，FedOSD 与 Retrain 的距离最小，说明其在该设置下更接近重训练基线；而在 severe 设置下，FU 重新取得最优的 Retrain 接近度（0.078），并在明显优于 FedAU 的保留客户端效用前提下，将目标客户端准确率从 0.496 压低到 0.067。整体来看，FU 在 severe 异质性下呈现出最稳定的忘记-效用平衡，而 moderate 设置下则需要结合具体目标决定是否采用更接近 Retrain 的 FedOSD。
此外，在本组实验中，FU、FedCSA 和 FedOSD 使用同一遗忘前全局模型作为起点，因此它们的遗忘前 local-buffer 与 global-buffer 指标可以直接横向比较；只有 FedAU 由于方法本身依赖训练阶段缓存，需要重新训练得到专属起点。

## English Paragraph for the Paper

Table~\ref{tab:lbgb_main_results} reports the heterogeneous-data comparison using the exact evaluation protocol emphasized in our paper: target-client forgetting is measured on local buffers, whereas retained utility is measured on global buffers. In our setup, FU, FedCSA, and FedOSD start from the same pre-unlearning checkpoint, so their pre-unlearning LB/GB metrics are directly comparable; FedAU is trained separately because it depends on training-time cache states. Under mild heterogeneity, FU reduces the local-buffer target-client accuracy to 0.151 while preserving 0.791 global-buffer retained-client accuracy, making it the closest non-retraining baseline to Retrain in this setting. Under moderate heterogeneity, FedOSD becomes the closest method to Retrain, indicating that the best utility-forgetting balance is setting dependent. Under severe heterogeneity, FU lowers the target-client accuracy from 0.496 to 0.067 and retains 0.598 accuracy on the retained clients, achieving the smallest distance to Retrain among the non-retraining methods. Across most settings, the post-unlearning MIA AUC remains around 0.48--0.50, suggesting that the major differences among methods come from the forgetting-utility trade-off rather than from large separations in attack AUC.

## Figure Caption (English)

Main metrics under CIFAR-10 heterogeneity. Target forgetting is measured by the post-unlearning local-buffer target-client accuracy, retained utility is measured by the post-unlearning global-buffer retained-client accuracy, and Retrain distance is computed in the two-dimensional (local-buffer target accuracy, global-buffer retained accuracy) plane. Lower target accuracy, Retrain distance, and MIA AUC are better.

## Supplementary Table Caption (English)

Cross-dataset severe-heterogeneity comparison using local-buffer target accuracy and global-buffer retained-client accuracy. Dist. to Retrain denotes the Euclidean distance to the Retrain point in the same evaluation space.