# Neuro‑Symbiotic Intelligence (NSI) Prototype

This repository contains a research prototype implementation of the
**Neuro‑Symbiotic Intelligence** (NSI) framework described in the paper:

> *Neuro‑Symbiotic Intelligence: A Convergent Learning Ecosystem via
> Co‑Evolution of Micro‑ and Macro‑Scale Language Models*.

NSI proposes a co‑evolutionary alternative to conventional federated
learning in which edge devices (``MicroLLMs``) and a central
coordinator (``MacroLLM``) exchange **cognition capsules** that
compress model updates and heuristic information.  The macro performs
**heuristic resonance analysis** and **Dynamic Aptitude Graph
Pruning** (DAGP) to adapt the communication graph and aggregate
updates in a resilient way.  The paper demonstrates dramatic
reductions in communication overhead and improved robustness to
poisoning attacks compared to state‑of‑the‑art federated learning.

## Components

The code in the `nsi_framework` package provides the following
components:

* `MicroLLM`: A lightweight linear classifier that trains on local
  data, tracks weight updates and extracts heuristic features using
  shallow decision trees.  It produces a `CognitionCapsule` containing
  a sparse embedding (`H_a`), an 8‑bit quantised gradient (`G_q`) and
  a binary heuristic vector (`H_p`).
* `MacroLLM`: A coordinator that receives capsules from multiple
  micro nodes, computes pairwise resonance between their heuristics,
  prunes the communication graph via DAGP, aggregates dequantised
  gradients, and updates a shared global weight vector.
* `dagp_prune`: A standalone implementation of the DAGP algorithm that
  can be applied to arbitrary graphs given a resonance matrix and
  threshold.
* `continuum_trust`: A simple behavioural security check that flags
  anomalous updates by computing the KL divergence between a gradient
  and a baseline distribution.
* `simulate.py`: A script that generates a synthetic non‑IID dataset,
  partitions it among a configurable number of micro nodes, runs
  multiple communication rounds of the NSI protocol, and reports
  accuracy, F1 and approximate bandwidth consumption for both NSI and
  a naive federated learning baseline.

The simulation is designed for reproducibility and can be run with

```bash
python -m nsi_framework.simulate --num-nodes 10 --rounds 5
```

to emulate the experiments described in the paper on a smaller
scale.  Because this prototype uses linear classifiers instead of
large language models, the absolute performance numbers differ from
those reported in the paper, but the relative trends (communication
savings, resilience to poisoning) hold.

## Limitations

This implementation is a simplified research prototype and does not
attempt to reproduce the full scale of the original experiments.  In
particular:

* `MicroLLM` uses a linear classifier and a shallow decision tree
  rather than quantised Transformers and discrete reasoning modules.
* Gradient quantisation and capsule compression are approximate and do
  not implement entropy coding.
* The macro does not host its own language model; it simply
  aggregates gradients.
* Poisoning and anti‑fragility experiments are not included in the
  default simulation but can be implemented by perturbing local
  gradients before capsule generation and using the continuum trust
  functions.

Despite these simplifications, the code faithfully follows the
algorithmic descriptions in the paper and can serve as a starting
point for further research and experimentation.

## Citation

If you use this code or ideas from the paper in your own work, please
cite the original paper.