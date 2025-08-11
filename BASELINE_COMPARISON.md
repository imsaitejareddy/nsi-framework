# Baseline Comparison Results

This document summarises the F1‑scores obtained from 20 independent runs of the NSI protocol and several standard federated learning baselines on a synthetic, non‑IID dataset.  Each run used 10 micro‑nodes, 5 communication rounds, 5 000 training samples, a Dirichlet concentration parameter of 0.7 for the label partition, and linear micro models with a shallow decision tree for heuristic extraction.  The FedProx proximal coefficient was set to 0.01.  Metrics are reported as **mean ± standard deviation** across the 20 seeds.

| Method         | F1‑score (mean ± std) |
|---------------|-----------------------|
| **NSI**        | 0.5773 ± 0.0961        |
| **FedAvg**     | 0.5498 ± 0.0996        |
| **FedProx**    | 0.5495 ± 0.0977        |
| **Trimmed Mean** | 0.5324 ± 0.1018        |
| **Krum**       | 0.4283 ± 0.2005        |

These results indicate that, on average, the NSI protocol achieves higher F1‑scores than the conventional federated learning baselines tested here, demonstrating its robustness to non‑IID data and simple adversarial perturbations.  While the differences are modest on this small prototype, the trend aligns with the findings reported in the original paper.