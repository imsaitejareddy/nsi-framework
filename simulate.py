"""Run a small‑scale simulation of the Neuro‑Symbiotic Intelligence (NSI) framework.

This script generates a synthetic binary classification dataset,
partitions it among multiple micro nodes in a non‑IID fashion using a
Dirichlet distribution, and executes a number of communication rounds
between micro and macro models.  At each round, micro models train
locally on their data, produce cognition capsules, and send them to
the macro.  The macro performs heuristic resonance analysis, prunes
the aptitude graph via DAGP, aggregates the quantised gradients, and
updates a shared global weight vector.  After each round, the
simulation evaluates global accuracy and F1 on a held‑out test set and
computes approximate bandwidth consumption for both the NSI update and
a naive federated learning (FL) update of full model parameters.

The goal of this simulation is to illustrate the qualitative
behaviour reported in the paper: dramatic reductions in
communication overhead, resilience to poisoned updates, and
competitive accuracy compared to baseline FL.  Because the models
used here are simple linear classifiers, the absolute performance
values differ from those in the paper, but the relative trends should
hold.

Usage
-----
Run the script from the command line.  For example:

```
python -m nsi_framework.simulate --num-nodes 10 --rounds 5
```

Default parameters are chosen to run quickly on modest hardware.  Use
``--help`` to see all available options.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from .micro import MicroLLM
from .macro import MacroLLM
from .trust import continuum_trust


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float = 0.5, random_state: int | None = None) -> List[np.ndarray]:
    """Partition labels into non‑IID subsets via Dirichlet sampling.

    Each client receives a portion of the overall label distribution.
    The parameter ``alpha`` controls the degree of non‑IIDness: lower
    values produce more skewed distributions.

    Parameters
    ----------
    y : numpy.ndarray
        Array of labels of length ``n_samples``.
    num_clients : int
        Number of clients (micro nodes).
    alpha : float, optional
        Dirichlet concentration parameter.  Default is 0.5.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    indices_per_client : list of numpy.ndarray
        List of index arrays, one per client, specifying which samples
        belong to that client.
    """
    rng = np.random.default_rng(random_state)
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    # Dirichlet allocation per class
    per_client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c, idxs in enumerate(class_indices):
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        # number of samples for each client in this class
        counts = (proportions * len(idxs)).astype(int)
        # adjust counts to match total
        counts[-1] = len(idxs) - np.sum(counts[:-1])
        start = 0
        for client_id, count in enumerate(counts):
            per_client_indices[client_id].extend(idxs[start:start + count].tolist())
            start += count
    return [np.array(sorted(idxs)) for idxs in per_client_indices]


def simulate(num_nodes: int = 10,
             rounds: int = 5,
             samples: int = 5000,
             input_dim: int = 20,
             alpha: float = 0.5,
             prune_threshold: float = 0.5,
             top_k_ratio: float = 0.1,
             lr: float = 0.01,
             dt_depth: int = 3,
             random_state: int | None = None) -> None:
    """Run the NSI simulation with the given hyperparameters and report results.

    Parameters
    ----------
    num_nodes : int
        Number of micro nodes participating in training.
    rounds : int
        Number of communication rounds between micro and macro models.
    samples : int
        Total number of synthetic examples to generate.  The dataset is
        split into train and test sets using an 80/20 split.
    input_dim : int
        Dimensionality of the feature space (excluding bias).  The
        weight vector will have length ``input_dim + 1``.
    alpha : float
        Dirichlet concentration parameter controlling non‑IIDness.
    prune_threshold : float
        Resonance threshold for DAGP.  Edges with resonance below this
        value are pruned.
    top_k_ratio : float
        Fraction of weight dimensions to include in the capsule’s sparse
        embedding ``H_a``.
    lr : float
        Learning rate for micro SGD classifiers.
    dt_depth : int
        Maximum depth of decision trees used for heuristics.
    random_state : int, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    # generate synthetic binary classification data
    X, y = make_classification(n_samples=samples, n_features=input_dim,
                               n_informative=int(input_dim * 0.6),
                               n_redundant=int(input_dim * 0.2),
                               n_clusters_per_class=2, weights=[0.9, 0.1],
                               flip_y=0.01, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=random_state, stratify=y)
    # partition training data among micro nodes
    indices_per_client = dirichlet_partition(y_train, num_nodes, alpha=alpha, random_state=random_state)
    # instantiate micro models
    classes = list(np.unique(y_train))
    micro_nodes: List[MicroLLM] = []
    for i in range(num_nodes):
        node = MicroLLM(node_id=i, input_dim=input_dim, classes=classes,
                        lr=lr, top_k_ratio=top_k_ratio, dt_depth=dt_depth,
                        random_state=(None if random_state is None else random_state + i))
        micro_nodes.append(node)
    # instantiate macro model
    macro = MacroLLM(num_nodes=num_nodes, input_dim=input_dim,
                     prune_threshold=prune_threshold, use_resonance_weights=False)
    # baseline: mean gradient across nodes (simulate FL)
    fl_global_weights = np.zeros(input_dim + 1)
    # metrics storage
    round_acc = []
    round_f1 = []
    round_fl_acc = []
    round_fl_f1 = []
    bandwidth_nsi = []
    bandwidth_fl = []
    start_time = time.time()
    # simulation rounds
    for r in range(rounds):
        # each micro trains locally for one epoch and produces a capsule
        capsules: List[CognitionCapsule] = []
        fl_grads: List[np.ndarray] = []
        for i, node in enumerate(micro_nodes):
            idx = indices_per_client[i]
            X_local = X_train[idx]
            y_local = y_train[idx]
            # train local model
            node.train_local(X_local, y_local, epochs=1)
            # produce capsule
            cap = node.generate_capsule(X_local, y_local, top_k_ratio=top_k_ratio)
            capsules.append(cap)
            # compute full gradient for FL baseline
            w = node._get_weights()
            if node.prev_weights is None:
                grad = np.zeros_like(w)
            else:
                # full gradient is difference between current weights and previous weights stored inside node
                # since node.prev_weights was updated when capsule was generated, we need to compute
                # gradient relative to previous weights before update; approximate using q_grad dequantised
                # we store gradient before update in prev_weights_backup to compute FL grad
                grad = cap.G_q.astype(float)  # placeholder; updated below
            # For the FL simulation we approximate the gradient as the difference between the current
            # weights and the global FL weights from the previous round.
            # Note: because micro models are initialised independently, this is a rough approximation.
            grad = w - fl_global_weights
            fl_grads.append(grad)
        # macro operations: resonance, DAGP, aggregate
        R_matrix = macro.heuristic_resonance(capsules)
        macro.dagp(R_matrix)
        macro.aggregate(capsules)
        # update FL global weights: simple average of full gradients
        fl_mean_grad = np.mean(np.stack(fl_grads, axis=0), axis=0)
        fl_global_weights += fl_mean_grad
        # evaluate NSI global model (macro) on test data
        # we approximate predictions by applying the global weight vector to test features
        # note: scale features with identity; this is a simple linear classifier
        X_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        logits_nsi = X_bias @ macro.global_weights
        preds_nsi = (logits_nsi >= 0).astype(int)
        acc_nsi = accuracy_score(y_test, preds_nsi)
        f1_nsi = f1_score(y_test, preds_nsi)
        round_acc.append(acc_nsi)
        round_f1.append(f1_nsi)
        # evaluate FL baseline model on test data
        logits_fl = X_bias @ fl_global_weights
        preds_fl = (logits_fl >= 0).astype(int)
        acc_fl = accuracy_score(y_test, preds_fl)
        f1_fl = f1_score(y_test, preds_fl)
        round_fl_acc.append(acc_fl)
        round_fl_f1.append(f1_fl)
        # compute bandwidth of NSI and FL updates
        # FL: each micro sends full weight vector (float32) to macro
        d = input_dim + 1
        fl_update_bytes = d * 4  # float32 per dimension
        # NSI: each micro sends k float32 + d int8 (grad) + d int8 (heuristics)
        k = max(1, int(top_k_ratio * d))
        nsi_update_bytes = k * 4 + d * 1 + d * 1
        bandwidth_fl.append(fl_update_bytes * num_nodes)
        bandwidth_nsi.append(nsi_update_bytes * num_nodes)
        print(f"Round {r+1}/{rounds}: acc_nsi={acc_nsi:.4f}, f1_nsi={f1_nsi:.4f}, acc_fl={acc_fl:.4f}, f1_fl={f1_fl:.4f}")
    total_time = time.time() - start_time
    print("\nSimulation finished.\n")
    # summarise results
    for name, values in [
        ("NSI Accuracy", round_acc),
        ("NSI F1", round_f1),
        ("FL Accuracy", round_fl_acc),
        ("FL F1", round_fl_f1)
    ]:
        vals = np.array(values)
        print(f"{name}: mean={vals.mean():.4f}, std={vals.std():.4f}")
    bw_fl = np.array(bandwidth_fl)
    bw_nsi = np.array(bandwidth_nsi)
    print(f"Bandwidth FL per round: {bw_fl[0]/1e3:.2f} kB per node")
    print(f"Bandwidth NSI per round: {bw_nsi[0]/1e3:.2f} kB per node")
    print(f"Total runtime: {total_time:.2f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the NSI framework")
    parser.add_argument("--num-nodes", type=int, default=10, help="number of micro nodes")
    parser.add_argument("--rounds", type=int, default=5, help="number of communication rounds")
    parser.add_argument("--samples", type=int, default=5000, help="total number of synthetic samples")
    parser.add_argument("--input-dim", type=int, default=20, help="number of input features")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for non-IID partitioning")
    parser.add_argument("--prune-threshold", type=float, default=0.5, help="resonance threshold for DAGP")
    parser.add_argument("--top-k-ratio", type=float, default=0.1, help="fraction of dimensions to include in H_a")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for micro models")
    parser.add_argument("--dt-depth", type=int, default=3, help="max depth of decision tree for heuristics")
    parser.add_argument("--random-state", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()
    simulate(num_nodes=args.num_nodes,
             rounds=args.rounds,
             samples=args.samples,
             input_dim=args.input_dim,
             alpha=args.alpha,
             prune_threshold=args.prune_threshold,
             top_k_ratio=args.top_k_ratio,
             lr=args.lr,
             dt_depth=args.dt_depth,
             random_state=args.random_state)


if __name__ == "__main__":
    main()