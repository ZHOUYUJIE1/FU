# Target-LR MIA Paper Tables

Lower post-unlearning MIA AUC is better. $\Delta$AUC = post - pre; negative values indicate reduced privacy leakage after unlearning.

## Main Table: Seed42 Heterogeneity

| Level | Method | Pre MIA AUC | Post MIA AUC | $\Delta$AUC |
| --- | --- | ---: | ---: | ---: |
| Mild | FU | 0.5206 | 0.5041 | -0.0165 |
|  | FedAU | 0.5210 | **0.4885** | -0.0325 |
|  | FedCSA | 0.5206 | 0.5171 | -0.0034 |
|  | FedOSD | 0.5206 | 0.5074 | -0.0132 |
|  | Retrain | -- | 0.5049 | -- |

| Moderate | FU | 0.5077 | 0.5146 | +0.0069 |
|  | FedAU | 0.5099 | 0.4989 | -0.0110 |
|  | FedCSA | 0.5077 | 0.5043 | -0.0034 |
|  | FedOSD | 0.5077 | 0.4900 | -0.0177 |
|  | Retrain | -- | **0.4766** | -- |

| Severe | FU | 0.4856 | 0.5151 | +0.0295 |
|  | FedAU | 0.5044 | 0.5128 | +0.0084 |
|  | FedCSA | 0.4856 | 0.5100 | +0.0244 |
|  | FedOSD | 0.4856 | 0.4860 | +0.0004 |
|  | Retrain | -- | **0.4693** | -- |

## Supplementary Table: Seed321 Severe

| Setting | Method | Pre MIA AUC | Post MIA AUC | $\Delta$AUC |
| --- | --- | ---: | ---: | ---: |
| Seed321 Severe | FedAU | 0.4896 | 0.5002 | +0.0107 |
|  | FedCSA | 0.5210 | 0.4901 | -0.0309 |
|  | FedOSD | 0.5210 | 0.4896 | -0.0314 |
|  | FU | 0.5210 | 0.5046 | -0.0164 |
|  | Retrain | -- | 0.4885 | -- |
|  | FU (r200) | 0.5225 | **0.4678** | -0.0547 |
|  | Retrain (r200) | -- | 0.5175 | -- |

## Supplementary Table: Severe Client Count

| Client # | Method | Pre MIA AUC | Post MIA AUC | $\Delta$AUC |
| --- | --- | ---: | ---: | ---: |
| 20 | FU | 0.4938 | **0.4862** | -0.0076 |
|  | Retrain | -- | 0.4899 | -- |
| 50 | FU | 0.5697 | **0.4908** | -0.0789 |
|  | Retrain | -- | 0.5167 | -- |
| 100 | FU | 0.4215 | 0.5387 | +0.1172 |
|  | Retrain | -- | **0.4971** | -- |