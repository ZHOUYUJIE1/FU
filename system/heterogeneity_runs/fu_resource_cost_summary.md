# FU Resource Cost Summary

## Scope

- Representative setting: `CIFAR-10`, `Dirichlet alpha=0.1`, `10` clients.
- Shared pretraining cost is separated from method-specific unlearning cost.
- Backbone: standard `torchvision` `ResNet-18` with `pretrained=False`, i.e., trained from scratch without ImageNet pretraining.

## Shared Backbone Cost

- Parameters: `11.18M`
- FP32 model size: `42.65 MB`
- Single-forward FLOPs on `32x32` input: `74.03 MFLOPs`

## Shared Pretraining Cost

- Common global training rounds before unlearning: `100`
- Empirical round time on this machine: about `22.2 s / round`
- Shared pretraining wall-clock cost: about `2,220-2,260 s` (`37-38 min`)

This cost is common to all methods and should not be repeatedly charged to each method in the main comparison table.

## Method-Specific Additional Cost

| Method | Additional communication rounds | Additional wall-clock time | Basis | Notes |
|---|---:|---:|---|---|
| FU | `5` | `112.22 s` | measured | Paper result uses `fu_select_best_recovery=True`, i.e. evaluate `5` recovery rounds and keep the best checkpoint. |
| FedAU | `0` | `N/A` | no explicit round-level log | No extra global recovery stage is triggered in the current pipeline. |
| FedCSA | `10` | `~217.5 s` | partially measured + partially estimated | `5` FedCSA unlearning rounds plus `5` outer recovery rounds; the latter are measured (`106.73 s`), the former are estimated from the common round time. |
| FedOSD | `21` | `~465.2 s` | estimated | `20` unlearning rounds plus `1` post-training round; log records rounds but not per-round wall-clock. |
| Retrain | `100` | `2196.91 s` | measured | Full retraining from scratch after removing the target client. |

## Paper-Safe Reading

- `FU` is much cheaper than `Retrain` in additional communication and wall-clock overhead.
- `FU` is also substantially lighter than `FedOSD` under the severe heterogeneity setting.
- `FedAU` has the lowest communication overhead, but its utility under severe heterogeneity is much lower than `FU`.
- `FedCSA` remains cheaper than `Retrain`, but its target forgetting is much weaker than `FU` under severe heterogeneity.

## Suggested Wording

Under the severe CIFAR-10 setting, the additional cost of `FU` remains modest: the method introduces only `5` extra communication rounds and about `112 s` of extra wall-clock time on top of the shared pretraining stage, which is far below `Retrain` (`100` rounds, about `2197 s`) and also below `FedOSD` (`21` rounds, about `465 s`). Since all methods share the same `ResNet-18` backbone trained from scratch, the observed cost gap mainly comes from the extra unlearning/recovery procedure rather than model size differences.
