# Severe Cross-Dataset Analysis

## Overall trends

- FU won utility on 2 of the 4 datasets and was a near-tie on STL10, making it the most consistent utility-preserving unlearning method in this suite.
- FedAU was the strongest fast forgetting method on all 4 datasets when measured by final target-client accuracy, but it repeatedly paid for that strength with the largest utility drop.
- FedCSA was most attractive when the goal was preserving the original model, especially on GTSRB, but its forgetting effect was often too mild.
- FedOSD usually sat in the middle: it forgot more than FedCSA, but it rarely dominated on utility, retain preservation, or privacy.
- Retrain remained the strongest absolute forgetting baseline, but it was not uniformly the best privacy baseline and often left noticeable utility on the table relative to FU or FedCSA.

## Dataset-wise observations

- On CIFAR-10, FU achieved the best utility (0.527) and retain accuracy (0.578), while FedAU pushed the target client lower among fast methods (0.033).
- Retrain remained the strongest forgetting baseline with final target accuracy 0.022, but its utility (0.326) stayed well below FU.
- Within CIFAR-10, the lowest final MIA AUC came from Retrain (0.489).

- On GTSRB, FedCSA almost preserved the original model (0.966 final average accuracy), but forgetting was weak (0.726 target accuracy).
- FedAU delivered the most aggressive fast forgetting (0.008), whereas FU provided the best trade-off between strong utility (0.894) and meaningful forgetting (0.411).
- Within GTSRB, the lowest final MIA AUC came from FU (0.477).

- On STL10, FU and FedCSA were essentially tied on utility (0.586 vs. 0.587), but FU forgot the target client much more strongly (0.209 vs. 0.351).
- FedAU was the most aggressive fast method on both forgetting (0.062) and privacy (0.427), at the cost of a sharp utility drop to 0.368.
- Within STL10, the lowest final MIA AUC came from FedAU (0.427).

- On Tiny-ImageNet, FU and FedCSA clearly dominated utility (0.368 and 0.365), with FU holding a slight edge on both target forgetting (0.118) and privacy (0.527).
- FedAU still forgot more aggressively (0.066), but its utility dropped to 0.266; FedOSD was dominated on both utility and privacy.
- Within Tiny-ImageNet, the lowest final MIA AUC came from Retrain (0.512).

## Writing suggestions

- If you want a single headline statement for the main paper, the cleanest one is: `FU achieves the most stable utility-forgetting trade-off across datasets, while FedAU is the most aggressive fast unlearning method and FedCSA is the most conservative.`
- If you want to emphasize dataset dependence, the strongest contrast is GTSRB vs. Tiny-ImageNet: GTSRB exposes FedCSA's utility advantage and forgetting weakness most clearly, while Tiny-ImageNet highlights FU's balance under a harder 200-class regime.
- STL10 is useful as the small-data case: it shows that aggressive forgetting can still be achieved, but the utility gap between FU/FedCSA and FedAU widens quickly.
