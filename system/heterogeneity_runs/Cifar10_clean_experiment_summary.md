# CIFAR10 Clean Heterogeneity Experiment Summary

## Notes
- dataset: Cifar10
- partition: dir
- global_rounds: 10
- clean setting: backdoor_train_poison_enabled=false, backdoor_train_poison_rate=0.0, backdoor_label_mode=fixed_target

## Seed 42 (mild/moderate/severe, 5 methods)

| level    | method   |   final_avg_acc |   final_target_acc |   final_retain_avg_acc | status   |
|:---------|:---------|----------------:|-------------------:|-----------------------:|:---------|
| mild     | fedau    |        0.292    |         0.0813333  |               0.315407 | complete |
| mild     | fedcsa   |        0.5762   |         0.28       |               0.609111 | complete |
| mild     | fedosd   |        0.485133 |         0.453333   |               0.488667 | complete |
| mild     | fu       |        0.495267 |         0.159333   |               0.532593 | complete |
| mild     | retrain  |        0.469867 |         0.04       |               0.51763  | complete |
| moderate | fedau    |        0.264533 |         0.00466667 |               0.293407 | complete |
| moderate | fedcsa   |        0.436933 |         0.272667   |               0.455185 | complete |
| moderate | fedosd   |        0.3632   |         0.166      |               0.385111 | complete |
| moderate | fu       |        0.4528   |         0.117333   |               0.490074 | complete |
| moderate | retrain  |        0.416867 |         0.169333   |               0.44437  | complete |
| severe   | fedau    |        0.2814   |         0          |               0.312667 | complete |
| severe   | fedcsa   |        0.220333 |         0.446      |               0.195259 | complete |
| severe   | fedosd   |        0.2784   |         0.307333   |               0.275185 | complete |
| severe   | fu       |        0.277333 |         0.0133333  |               0.306667 | complete |
| severe   | retrain  |        0.205733 |         0.022      |               0.226148 | complete |

## Seed 321 (severe only, 5 methods)

| level   | method   |   final_avg_acc |   final_target_acc |   final_retain_avg_acc | status   |
|:--------|:---------|----------------:|-------------------:|-----------------------:|:---------|
| severe  | fedau    |        0.2078   |         0          |               0.230889 | complete |
| severe  | fedcsa   |        0.320267 |         0.006      |               0.355185 | complete |
| severe  | fedosd   |        0.316333 |         0.264667   |               0.322074 | complete |
| severe  | fu       |        0.346533 |         0.00866667 |               0.384074 | complete |
| severe  | retrain  |        0.309067 |         0.002      |               0.343185 | complete |

## Severe: seed 42 vs 321 (final_avg_acc)

| method   |       42 |      321 |   delta_321_minus_42 |
|:---------|---------:|---------:|---------------------:|
| fedau    | 0.2814   | 0.2078   |           -0.0736    |
| fedcsa   | 0.220333 | 0.320267 |            0.0999333 |
| fedosd   | 0.2784   | 0.316333 |            0.0379333 |
| fu       | 0.277333 | 0.346533 |            0.0692    |
| retrain  | 0.205733 | 0.309067 |            0.103333  |
