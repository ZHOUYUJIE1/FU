# Client Count Comparison Analysis

- Available client counts: [10, 20, 50, 100]
- Methods: FU, FedAU, FedCSA, FedOSD, Retrain

## Final Avg Accuracy
- clients=10: best=FU (0.4305)
- clients=20: best=FU (0.4221)
- clients=50: best=FU (0.3207)
- clients=100: best=Retrain (0.2402)

## Target Client Accuracy
- clients=10: best=Retrain (0.0435)
- clients=20: best=Retrain (0.0080)
- clients=50: best=FedAU (0.0128)
- clients=100: best=FU (0.0000)

## Post-Unlearning MIA AUC
- clients=10: best=FedAU (0.4518)
- clients=20: best=FedAU (0.4678)
- clients=50: best=FedAU (0.4847)
- clients=100: best=Retrain (0.4641)

## Post-Unlearning Backdoor Accuracy
- clients=10: best=FedAU (0.2292)
- clients=20: best=FedAU (0.0052)
- clients=50: best=FedAU (0.0052)
- clients=100: best=FU (0.0000)
