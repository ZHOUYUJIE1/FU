# Best-FU Heterogeneity Analysis

Final summary: [Cifar10_FedAvg_heterogeneity_summary_fu_best.csv](./Cifar10_FedAvg_heterogeneity_summary_fu_best.csv)
Main figure: [heterogeneity_paper_figure_fu_best.png](./heterogeneity_paper_figure_fu_best.png)
LaTeX table: [heterogeneity_paper_table_fu_best.tex](./heterogeneity_paper_table_fu_best.tex)

## Key observations

- FU uses the best per-level configuration: mild=`fu_mild_best`, moderate=`fu_moderate_best`, severe=`fu_severe_best`.
- Compared with the original FU setting, the best-per-level FU reduces final target-client accuracy from 0.4471/0.4747/0.1957 to 0.3398/0.4035/0.0942 under mild/moderate/severe heterogeneity.
- The corresponding final average accuracy changes from 0.5475/0.5677/0.4136 to 0.5079/0.5529/0.4490.

## Paper-ready paragraph

After selecting the best FU configuration for each heterogeneity level, FU exhibits a stronger utility-forgetting tradeoff across heterogeneous federated settings. Under mild heterogeneity, FU improves forgetting strength by reducing the final target-client accuracy from 0.4471 to 0.3398, while preserving a relatively high final average accuracy of 0.5079. Under moderate heterogeneity, the adaptive FU configuration remains the best practical choice, achieving 0.5529 final average accuracy with 0.4035 target-client accuracy, indicating that further strengthening forgetting would incur a disproportionate utility loss under the current recovery mechanism. Under severe heterogeneity, FU achieves the best final average accuracy among all compared methods (0.4490) and reduces the final target-client accuracy to 0.0942, approaching the retraining baseline (0.0435). These results suggest that FU is most convincingly positioned as a balanced federated unlearning method that preserves model utility while providing competitive forgetting strength, with particularly strong evidence under severe heterogeneity.

## Conservative claim

FU should be described as providing a better utility-forgetting balance rather than uniformly dominating all baselines on every metric. This claim is strongest in the severe heterogeneity setting, where FU combines the highest final average accuracy with substantially lower target-client accuracy than FedAU, FedCSA, and FedOSD.
