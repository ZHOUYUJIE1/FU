# CIFAR-10 Severe FU Ablation Summary

Metrics follow the current FU evaluation pipeline.
- `final_*`: local-buffer metrics from `eval_post_forget`.
- `global_buffer_*`: global-buffer utility metrics from `eval_post_forget`.

## Seed 42

| Variant | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.5435 | 0.0713 | 0.5959 | 0.6321 | 0.5160 | 0.0990 |
| w/o H | 0.5875 | 0.2073 | 0.6297 | 0.6358 | 0.4887 | 0.0833 |
| w/o A | 0.1143 | 0.0100 | 0.1259 | 0.0750 | 0.5077 | 0.4271 |
| w/o U-mask | 0.1944 | 0.0133 | 0.2145 | 0.1764 | 0.5110 | 0.0938 |
| w/o retain | 0.0907 | 0.0160 | 0.0990 | 0.1476 | 0.5199 | 0.0000 |

## Seed 321

| Variant | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.7315 | 0.0480 | 0.8074 | 0.7518 | 0.4985 | 0.1771 |
| w/o H | 0.7271 | 0.0493 | 0.8024 | 0.7589 | 0.4997 | 0.3125 |
| w/o A | 0.3063 | 0.0220 | 0.3379 | 0.4468 | 0.5051 | 0.3750 |
| w/o U-mask | 0.4239 | 0.0167 | 0.4692 | 0.5653 | 0.4983 | 0.2083 |
| w/o retain | 0.1227 | 0.0000 | 0.1364 | 0.3569 | 0.5009 | 0.0000 |

## Mean Across Seeds

| Variant | Final Avg Acc | Target Acc | Retain Avg Acc | Global Retain Acc | MIA AUC | Backdoor Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0.6375 | 0.0597 | 0.7017 | 0.6920 | 0.5072 | 0.1380 |
| w/o H | 0.6573 | 0.1283 | 0.7160 | 0.6973 | 0.4942 | 0.1979 |
| w/o A | 0.2103 | 0.0160 | 0.2319 | 0.2609 | 0.5064 | 0.4010 |
| w/o U-mask | 0.3092 | 0.0150 | 0.3419 | 0.3709 | 0.5047 | 0.1510 |
| w/o retain | 0.1067 | 0.0080 | 0.1177 | 0.2522 | 0.5104 | 0.0000 |
