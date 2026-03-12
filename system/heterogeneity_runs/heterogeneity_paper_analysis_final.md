# Heterogeneity Analysis for Paper

Final summary: [Cifar10_FedAvg_heterogeneity_summary_final.csv](./Cifar10_FedAvg_heterogeneity_summary_final.csv)

Main figure: [heterogeneity_paper_figure_final.png](./heterogeneity_paper_figure_final.png)

LaTeX table: [heterogeneity_paper_table_final.tex](./heterogeneity_paper_table_final.tex)

## Key results

- Under mild heterogeneity, FU achieves `0.5035` final average accuracy and reduces the target-client accuracy to `0.3279`.
- Under moderate heterogeneity, FU achieves `0.5529` final average accuracy with `0.4035` target-client accuracy.
- Under severe heterogeneity, FU achieves `0.4490` final average accuracy and reduces the target-client accuracy to `0.0942`.
- In the severe heterogeneity setting, FU attains the highest final average accuracy among all compared methods while maintaining substantially lower target-client accuracy than FedAU, FedCSA, and FedOSD.

## Paper-ready paragraph

Table~\ref{tab:heterogeneity_results} reports the results under mild, moderate, and severe heterogeneity settings on CIFAR-10. FU consistently maintains strong utility across all heterogeneous scenarios while improving the forgetting effectiveness on the target client. Under mild heterogeneity, FU achieves a final average accuracy of `0.5035` and reduces the target-client accuracy to `0.3279`, outperforming FedAU in both utility retention and target suppression. Under moderate heterogeneity, FU reaches `0.5529` final average accuracy with `0.4035` target-client accuracy, indicating a favorable balance between model availability and forgetting strength. Under severe heterogeneity, FU obtains the best final average accuracy among all compared methods (`0.4490`) while lowering the target-client accuracy to `0.0942`, which is markedly better than FedAU, FedCSA, and FedOSD and approaches the retraining baseline. These results show that FU is particularly competitive as a balanced federated unlearning method under heterogeneous data distributions.

## Conservative wording

The most defensible claim is that FU provides a stronger utility-forgetting tradeoff, rather than strictly dominating all baselines on every single metric. This statement is especially well supported in the severe heterogeneity setting.
