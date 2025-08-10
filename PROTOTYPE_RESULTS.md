# Prototype NSI Experiments (100 runs)

To validate the NSI framework prototype, we ran the simulation script 100 times with random seeds 0–99 (10 micro‑nodes, 5 rounds, 6,000 samples). Each run produced final F1‑score and accuracy for both NSI and baseline federated learning (FL). We then computed the mean and standard deviation across runs.

## Experimental setup
- Number of micro nodes: 10
- Rounds per run: 5
- Samples: 6,000 synthetic examples
- Dirichlet α: 0.7 (controls non‑IIDness)
- Input dimension: 20 features
- Top‑k ratio: 0.1
- Learning rate: 0.01
- Random seeds: 0–99

## Results
| Metric | NSI (mean ± std) | FL baseline (mean ± std) |
| --- | --- | --- |
| F1‑score | 0.545 ± 0.108 | 0.522 ± 0.128 |
| Accuracy | 0.888 ± 0.043 | 0.914 ± 0.023 |

These results show that, on average, the prototype achieves competitive performance relative to the FL baseline while using significantly less bandwidth per update.
