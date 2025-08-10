"""Evaluate the NSI framework against common federated learning baselines.

This script runs a small‑scale federated learning experiment using
``nsi_framework`` micro models and compares the Neuro‑Symbiotic
Intelligence (NSI) protocol to several baseline aggregators:

* **FedAvg** – simple average of full gradients across nodes.
* **Trimmed‑Mean** – robust mean that discards a fraction of the largest
  and smallest gradients on each coordinate before averaging.
* **Krum** – Byzantine‑resilient aggregator that selects the gradient
  closest to its neighbours.

The underlying local model is the ``MicroLLM`` class defined in
``nsi_framework``.  A synthetic dataset is generated and partitioned
non‑IIDly using a Dirichlet distribution.  Each aggregator runs for a
number of communication rounds, and the resulting accuracy and F1
scores on a hold‑out test set are reported.

Run this script with default parameters to reproduce a small experiment:

```
python nsi_vs_baselines.py --num-nodes 10 --rounds 5
```

Note
----
This comparison is illustrative only.  The implementations here are
simplified and do not incorporate all of the nuances described in
the NSI paper.  They provide a baseline for understanding how the
NSI protocol differs from standard federated learning schemes.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification

from nsi_framework.micro import MicroLLM
from nsi_framework.macro import MacroLLM


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Non‑IID label partitioning via Dirichlet sampling."""
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for idxs in class_indices:
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * len(idxs)).astype(int)
        counts[-1] = len(idxs) - np.sum(counts[:-1])
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(idxs[start:start + count].tolist())
            start += count
    return [np.array(sorted(c_idx)) for c_idx in client_indices]


def trimmed_mean(grads: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
    """Compute the coordinate‑wise trimmed mean of a set of gradients.

    A fraction ``trim_ratio`` of the largest and smallest values on
    each coordinate is discarded before computing the mean.  If the
    discard window removes all but zero elements, the untrimmed mean
    is returned.
    """
    n = grads.shape[0]
    k = int(trim_ratio * n)
    if k == 0 or 2 * k >= n:
        return np.mean(grads, axis=0)
    sorted_grads = np.sort(grads, axis=0)
    trimmed = sorted_grads[k:n - k]
    return np.mean(trimmed, axis=0)


def krum(grads: np.ndarray, f: int = 1) -> np.ndarray:
    """Select a gradient using the Krum robust aggregation rule.

    ``f`` is the maximum number of Byzantine gradients assumed.  The
    Krum score for each candidate gradient is the sum of squared
    distances to its closest ``n - f - 2`` neighbours.  The gradient with
    the lowest score is returned.
    """
    n = grads.shape[0]
    m = n - f - 2
    if m <= 0:
        return np.mean(grads, axis=0)
    scores = np.zeros(n)
    for i in range(n):
        dists = np.sum((grads[i] - grads) ** 2, axis=1)
        dists_sorted = np.sort(dists)
        scores[i] = np.sum(dists_sorted[1:m + 1])
    return grads[np.argmin(scores)]


def run_baseline(aggregator: str,
                 micros: List[MicroLLM],
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 indices_per_client: List[np.ndarray],
                 global_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run one communication round using the specified baseline aggregator.

    Parameters
    ----------
    aggregator : str
        Either ``"fedavg"``, ``"trimmed"`` or ``"krum"``.
    micros : list of MicroLLM
        Micro nodes participating in training.
    X_train : numpy.ndarray
        Training features.
    y_train : numpy.ndarray
        Training labels.
    indices_per_client : list of numpy.ndarray
        Partition of samples among clients.
    global_weights : numpy.ndarray
        Current global weight vector to compute local gradients relative
        to.

    Returns
    -------
    new_global : numpy.ndarray
        Updated global weight vector after aggregation.
    local_preds : numpy.ndarray
        Concatenated predictions from all micro models on their local
        data (unused but returned for completeness).
    """
    grads = []
    local_preds = []
    for i, node in enumerate(micros):
        idx = indices_per_client[i]
        X_local = X_train[idx]
        y_local = y_train[idx]
        node.train_local(X_local, y_local, epochs=1)
        w = node._get_weights()
        grad = w - global_weights
        grads.append(grad)
        # store predictions (not used downstream)
        local_preds.append(node.predict(X_local))
    grads_arr = np.stack(grads, axis=0)
    if aggregator == "fedavg":
        update = np.mean(grads_arr, axis=0)
    elif aggregator == "trimmed":
        update = trimmed_mean(grads_arr, trim_ratio=0.1)
    elif aggregator == "krum":
        update = krum(grads_arr, f=1)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")
    return global_weights + update, np.concatenate(local_preds)


def simulate_vs_baselines(num_nodes: int = 10,
                          rounds: int = 5,
                          samples: int = 5000,
                          input_dim: int = 20,
                          alpha: float = 0.5,
                          lr: float = 0.01,
                          dt_depth: int = 3,
                          random_state: int | None = None) -> None:
    """Run NSI and baseline aggregators on the same data and report metrics."""
    rng = np.random.default_rng(random_state)
    # synthetic dataset
    X, y = make_classification(n_samples=samples, n_features=input_dim,
                               n_informative=int(0.6 * input_dim), n_redundant=int(0.2 * input_dim),
                               n_clusters_per_class=2, weights=[0.9, 0.1], flip_y=0.01,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=random_state)
    indices = dirichlet_partition(y_train, num_nodes, alpha, rng)
    classes = list(np.unique(y_train))
    # micro nodes for NSI
    micros_nsi: List[MicroLLM] = []
    micros_fedavg: List[MicroLLM] = []
    micros_trimmed: List[MicroLLM] = []
    micros_krum: List[MicroLLM] = []
    for i in range(num_nodes):
        micros_nsi.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                   dt_depth=dt_depth, top_k_ratio=0.1,
                                   random_state=(None if random_state is None else random_state + i)))
        # baseline micros share configuration but have separate weights
        micros_fedavg.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                      dt_depth=dt_depth, top_k_ratio=0.1,
                                      random_state=(None if random_state is None else random_state + 100 + i)))
        micros_trimmed.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                       dt_depth=dt_depth, top_k_ratio=0.1,
                                       random_state=(None if random_state is None else random_state + 200 + i)))
        micros_krum.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                    dt_depth=dt_depth, top_k_ratio=0.1,
                                    random_state=(None if random_state is None else random_state + 300 + i)))
    macro = MacroLLM(num_nodes=num_nodes, input_dim=input_dim, prune_threshold=0.5)
    # global weights for baselines
    gw_fedavg = np.zeros(input_dim + 1)
    gw_trimmed = np.zeros(input_dim + 1)
    gw_krum = np.zeros(input_dim + 1)
    # metrics storage
    acc_nsi = []
    f1_nsi = []
    acc_fedavg = []
    f1_fedavg = []
    acc_trimmed = []
    f1_trimmed = []
    acc_krum = []
    f1_krum = []
    for r in range(rounds):
        # NSI round
        caps = []
        for i, node in enumerate(micros_nsi):
            idx = indices[i]
            X_local = X_train[idx]
            y_local = y_train[idx]
            node.train_local(X_local, y_local, epochs=1)
            cap = node.generate_capsule(X_local, y_local)
            caps.append(cap)
        R = macro.heuristic_resonance(caps)
        macro.dagp(R)
        macro.aggregate(caps)
        # baseline rounds
        gw_fedavg, _ = run_baseline("fedavg", micros_fedavg, X_train, y_train, indices, gw_fedavg)
        gw_trimmed, _ = run_baseline("trimmed", micros_trimmed, X_train, y_train, indices, gw_trimmed)
        gw_krum, _ = run_baseline("krum", micros_krum, X_train, y_train, indices, gw_krum)
        # evaluate
        X_bias = np.hstack([X_test, np.ones((len(X_test), 1))])
        preds_nsi = (X_bias @ macro.global_weights >= 0).astype(int)
        preds_fedavg = (X_bias @ gw_fedavg >= 0).astype(int)
        preds_trimmed = (X_bias @ gw_trimmed >= 0).astype(int)
        preds_krum = (X_bias @ gw_krum >= 0).astype(int)
        acc_nsi.append(accuracy_score(y_test, preds_nsi))
        f1_nsi.append(f1_score(y_test, preds_nsi))
        acc_fedavg.append(accuracy_score(y_test, preds_fedavg))
        f1_fedavg.append(f1_score(y_test, preds_fedavg))
        acc_trimmed.append(accuracy_score(y_test, preds_trimmed))
        f1_trimmed.append(f1_score(y_test, preds_trimmed))
        acc_krum.append(accuracy_score(y_test, preds_krum))
        f1_krum.append(f1_score(y_test, preds_krum))
        print(f"Round {r+1}/{rounds} completed.")
    # report
    def report(name: str, vals: List[float]) -> str:
        arr = np.array(vals)
        return f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}"
    print("\nSummary of experiments:\n")
    print(report("NSI Accuracy", acc_nsi))
    print(report("NSI F1", f1_nsi))
    print(report("FedAvg Accuracy", acc_fedavg))
    print(report("FedAvg F1", f1_fedavg))
    print(report("Trimmed Accuracy", acc_trimmed))
    print(report("Trimmed F1", f1_trimmed))
    print(report("Krum Accuracy", acc_krum))
    print(report("Krum F1", f1_krum))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NSI against federated learning baselines")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dt-depth", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()
    simulate_vs_baselines(num_nodes=args.num_nodes,
                          rounds=args.rounds,
                          samples=args.samples,
                          input_dim=args.input_dim,
                          alpha=args.alpha,
                          lr=args.lr,
                          dt_depth=args.dt_depth,
                          random_state=args.random_state)


if __name__ == "__main__":
    main()
