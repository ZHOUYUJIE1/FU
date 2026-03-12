# Client Count Experiments Under Severe Heterogeneity

## 1. Completion Status

- Target matrix: `4` client counts (`10, 20, 50, 100`) x `5` methods (`FU, FedAU, FedCSA, FedOSD, Retrain`) = `20` runs.
- Completed: `20 / 20`.
- `FedOSD @ 100` has been successfully backfilled.
- Re-run note:
  - To avoid CUDA OOM at the first unlearning round, `FedOSD @ 100` was run with `--fedosd-max-online-clients 20` (all other core settings unchanged).

## 2. Main Quantitative Results

### Utility (`final_avg_acc`, higher is better)

- `10` clients: best `FU` (`0.4305`)
- `20` clients: best `FU` (`0.4221`)
- `50` clients: best `FU` (`0.3207`)
- `100` clients: best `Retrain` (`0.2402`)

### Forgetting Strength (`final_target_acc`, lower is better)

- `10` clients: best `Retrain` (`0.0435`)
- `20` clients: best `Retrain` (`0.0080`)
- `50` clients: best `FedAU` (`0.0128`)
- `100` clients: `FU`, `FedAU`, `FedOSD`, `Retrain` all reach `0.0000`

### Retained-Client Utility (`final_retain_avg_acc`, higher is better)

- `10/20/50` clients: best `FU` (`0.4655 / 0.4436 / 0.3248`)
- `100` clients: best `Retrain` (`0.2426`)

### Privacy / Backdoor (`mia_post_auc`, `backdoor_post_acc`, lower is better)

- `post-MIA AUC`:
  - `10`: best `FedAU` (`0.4518`)
  - `20`: best `FedAU` (`0.4678`)
  - `50`: best `FedAU` (`0.4847`)
  - `100`: among methods with available `mia_post_auc`, best `FedAU` (`0.5055`)
- `post-backdoor accuracy`:
  - `10/20/50`: best `FedAU` (`0.2292 / 0.0052 / 0.0052`)
  - `100`: `FU`, `FedAU`, and `FedOSD` tie at `0.0000`

## 3. Scaling Trend (`10 -> 100` clients)

- `FU`: `0.4305 -> 0.2039` (`-52.7%`)
- `FedAU`: `0.3500 -> 0.1907` (`-45.5%`)
- `FedCSA`: `0.4289 -> 0.1762` (`-58.9%`)
- `FedOSD`: `0.3559 -> 0.2008` (`-43.6%`)
- `Retrain`: `0.3796 -> 0.2402` (`-36.7%`)

Interpretation:

- Utility decreases with more clients for all methods under fixed total data.
- This is expected because each client receives fewer samples and local distributions become more fragmented.
- The key comparison is relative degradation: at `100` clients, `Retrain` is most utility-stable; `FU` still wins at `10/20/50`.

## 4. Method-Level Reading

- `FU`:
  - Strongest utility and retained-client utility at `10/20/50`.
  - Not consistently best on privacy metrics.
  - Loses utility lead at `100`.
- `FedAU`:
  - Strongest and most stable on privacy/backdoor metrics across client counts.
  - Best forgetting at `50`.
- `FedOSD`:
  - `100`-client point is now available and comparable in table/figure.
  - Utility at `100` (`0.2008`) is below `Retrain` and close to `FU` (`0.2039`).
- `Retrain`:
  - Best utility at `100`.
  - Strong forgetting at low-to-mid client counts.

## 5. Paper-Safe Claims

- Recommended claim:
  - Under severe heterogeneity, `FU` provides the best utility for moderate federation scales (`10-50` clients), while `Retrain` becomes stronger at very large scale (`100` clients).
  - `FedAU` remains strongest on privacy/backdoor indicators.
- Avoid claiming:
  - One method dominates all metrics for all client counts.
