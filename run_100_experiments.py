"""
Script to reproduce the 100‑run prototype experiment described in
``PROTOTYPE_RESULTS.md``.  This utility generates a synthetic binary
classification dataset, partitions it among a fixed number of micro
clients, executes the NSI simulation for a specified number of
communication rounds, and records the final F1‑score and accuracy for
both the NSI aggregator and a federated learning (FL) baseline.  The
process is repeated for seeds 0‑99, and the mean and standard
deviation of each metric are reported at the end.

The simulation settings mirror those used to generate the summary
statistics in the accompanying documentation:

* 10 micro nodes
* 5 communication rounds per run
* 6 000 total samples with 20 features
* Dirichlet α = 0.7 to control non‑IIDness
* Top‑k ratio of 0.1 for the capsule’s sparse embedding
* Learning rate 0.01

To run the experiment, execute this module directly with Python::

    python run_100_experiments.py

The script will print intermediate progress every 10 runs and final
aggregated statistics at the end.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from nsi_framework.micro import MicroLLM
from nsi_framework.macro import MacroLLM
from nsi_framework.simulate import dirichlet_partition


def run_single(
    seed: int,
    *,
    num_nodes: int = 10,
    rounds: int = 5,
    samples: int = 6000,
    input_dim: int = 20,
    alpha: float = 0.7,
    prune_threshold: float = 0.5,
    top_k_ratio: float = 0.1,
    lr: float = 0.01,
    dt_depth: int = 3,
) -> tuple[float, float, float, float] | None:
    """Run a single NSI/FL experiment and return final metrics.

    Parameters are chosen to match the prototype configuration used
    in the ``PROTOTYPE_RESULTS.md`` summary.  If a given random seed
    happens to produce an empty client (due to the Dirichlet
    partition), the function returns ``None`` and the run is skipped.

    Returns
    -------
    (f1_nsi, f1_fl, acc_nsi, acc_fl) : tuple of floats or None
        Final F1‑score and accuracy for both NSI and FL.  If the
        experiment is skipped, ``None`` is returned.
    """
    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=samples,
        n_features=input_dim,
        n_informative=int(input_dim * 0.6),
        n_redundant=int(input_dim * 0.2),
        n_clusters_per_class=2,
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=seed,
    )
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Non‑IID partitioning
    indices_per_client = dirichlet_partition(
        y_train, num_nodes, alpha=alpha, random_state=seed
    )
    classes = list(np.unique(y_train))
    # Initialise micro and macro models
    micro_nodes: list[MicroLLM] = []
    for i in range(num_nodes):
        node = MicroLLM(
            node_id=i,
            input_dim=input_dim,
            classes=classes,
            lr=lr,
            top_k_ratio=top_k_ratio,
            dt_depth=dt_depth,
            random_state=seed + i,
        )
        micro_nodes.append(node)
    macro = MacroLLM(
        num_nodes=num_nodes,
        input_dim=input_dim,
        prune_threshold=prune_threshold,
        use_resonance_weights=False,
    )
    # Initialise FL global weights
    fl_global = np.zeros(input_dim + 1)
    # Simulation rounds
    for _ in range(rounds):
        capsules = []
        fl_grads = []
        for i, node in enumerate(micro_nodes):
            idx = indices_per_client[i].astype(int)
            if idx.size == 0:
                # Skip nodes with no data for this seed
                continue
            X_local, y_local = X_train[idx], y_train[idx]
            node.train_local(X_local, y_local, epochs=1)
            cap = node.generate_capsule(X_local, y_local, top_k_ratio=top_k_ratio)
            capsules.append(cap)
            fl_grads.append(node._get_weights() - fl_global)
        if not capsules:
            # All clients empty; skip this run
            return None
        # NSI aggregation
        R_matrix = macro.heuristic_resonance(capsules)
        macro.dagp(R_matrix)
        macro.aggregate(capsules)
        # FL aggregation
        if fl_grads:
            fl_global += np.mean(np.stack(fl_grads, axis=0), axis=0)
    # Evaluate on test set
    X_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    preds_nsi = (X_bias @ macro.global_weights >= 0).astype(int)
    preds_fl = (X_bias @ fl_global >= 0).astype(int)
    f1_nsi = f1_score(y_test, preds_nsi)
    f1_fl = f1_score(y_test, preds_fl)
    acc_nsi = accuracy_score(y_test, preds_nsi)
    acc_fl = accuracy_score(y_test, preds_fl)
    return (f1_nsi, f1_fl, acc_nsi, acc_fl)


def main() -> None:
    """Run 100 experiments and print aggregated statistics."""
    results: list[tuple[float, float, float, float]] = []
    for seed in range(100):
        res = run_single(seed)
        if res is not None:
            results.append(res)
        # progress indicator every 10 runs
        if (seed + 1) % 10 == 0:
            print(f"Completed {seed + 1} runs")
    # Convert to arrays
    f1_nsi_vals = np.array([r[0] for r in results])
    f1_fl_vals = np.array([r[1] for r in results])
    acc_nsi_vals = np.array([r[2] for r in results])
    acc_fl_vals = np.array([r[3] for r in results])
    print(f"\nCollected {len(results)} runs")
    print(f"NSI F1 mean = {f1_nsi_vals.mean():.4f}, std = {f1_nsi_vals.std():.4f}")
    print(f"FL  F1 mean = {f1_fl_vals.mean():.4f}, std = {f1_fl_vals.std():.4f}")
    print(f"NSI ACC mean = {acc_nsi_vals.mean():.4f}, std = {acc_nsi_vals.std():.4f}")
    print(f"FL  ACC mean = {acc_fl_vals.mean():.4f}, std = {acc_fl_vals.std():.4f}")


if __name__ == "__main__":
    main()