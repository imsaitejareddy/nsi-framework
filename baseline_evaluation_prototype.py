"""Evaluate NSI against multiple federated learning baselines over many seeds.

This script runs the NSI protocol and several baseline federated learning
aggregators on synthetic data over multiple random seeds.  For each run
it records the F1‑score after the final communication round.  At the
end it reports the mean and standard deviation of the F1‑scores across
all seeds for each method.  This provides a more robust estimate of
performance than a single simulation.

The baselines considered are:

* **FedAvg** – standard federated averaging.
* **FedProx** – FedAvg with a proximal regularisation term controlled
  by ``mu``.
* **Trimmed‑Mean** – robust aggregation that discards a fraction of
  extreme values on each coordinate.
* **Krum** – Byzantine‑resilient selection of a single gradient.

Usage::

    python baseline_evaluation_prototype.py --seeds 20 --num-nodes 10 --rounds 5

"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from nsi_framework.micro import MicroLLM
from nsi_framework.macro import MacroLLM
from nsi_vs_baselines import dirichlet_partition, run_baseline, trimmed_mean, krum


def evaluate_once(num_nodes: int,
                  rounds: int,
                  samples: int,
                  input_dim: int,
                  alpha: float,
                  lr: float,
                  dt_depth: int,
                  mu: float,
                  random_state: int) -> Tuple[float, float, float, float, float]:
    """Run one simulation and return final F1 for each method.

    Returns
    -------
    tuple
        (f1_nsi, f1_fedavg, f1_fedprox, f1_trimmed, f1_krum)
    """
    rng = np.random.default_rng(random_state)
    X, y = make_classification(n_samples=samples, n_features=input_dim,
                               n_informative=int(0.6 * input_dim), n_redundant=int(0.2 * input_dim),
                               n_clusters_per_class=2, weights=[0.9, 0.1], flip_y=0.01,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=random_state)
    indices = dirichlet_partition(y_train, num_nodes, alpha, rng)
    classes = list(np.unique(y_train))
    # instantiate micros for each scheme
    micros_nsi: List[MicroLLM] = []
    micros_fedavg: List[MicroLLM] = []
    micros_fedprox: List[MicroLLM] = []
    micros_trimmed: List[MicroLLM] = []
    micros_krum: List[MicroLLM] = []
    for i in range(num_nodes):
        micros_nsi.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                   dt_depth=dt_depth, top_k_ratio=0.1,
                                   random_state=random_state + i))
        micros_fedavg.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                      dt_depth=dt_depth, top_k_ratio=0.1,
                                      random_state=random_state + 100 + i))
        micros_fedprox.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                       dt_depth=dt_depth, top_k_ratio=0.1,
                                       random_state=random_state + 200 + i))
        micros_trimmed.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                       dt_depth=dt_depth, top_k_ratio=0.1,
                                       random_state=random_state + 300 + i))
        micros_krum.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes, lr=lr,
                                    dt_depth=dt_depth, top_k_ratio=0.1,
                                    random_state=random_state + 400 + i))
    macro = MacroLLM(num_nodes=num_nodes, input_dim=input_dim, prune_threshold=0.5)
    gw_fedavg = np.zeros(input_dim + 1)
    gw_fedprox = np.zeros(input_dim + 1)
    gw_trimmed = np.zeros(input_dim + 1)
    gw_krum = np.zeros(input_dim + 1)
    # training rounds
    for r in range(rounds):
        # NSI round
        caps = []
        for i, node in enumerate(micros_nsi):
            idx = indices[i]
            X_local = X_train[idx]
            y_local = y_train[idx]
            node.train_local(X_local, y_local, epochs=1)
            caps.append(node.generate_capsule(X_local, y_local))
        R = macro.heuristic_resonance(caps)
        macro.dagp(R)
        macro.aggregate(caps)
        # FedAvg
        gw_fedavg, _ = run_baseline("fedavg", micros_fedavg, X_train, y_train, indices, gw_fedavg)
        # FedProx
        gw_fedprox, _ = run_baseline("fedprox", micros_fedprox, X_train, y_train, indices, gw_fedprox, mu=mu)
        # Trimmed
        gw_trimmed, _ = run_baseline("trimmed", micros_trimmed, X_train, y_train, indices, gw_trimmed)
        # Krum
        gw_krum, _ = run_baseline("krum", micros_krum, X_train, y_train, indices, gw_krum)
    # evaluate final round
    X_bias = np.hstack([X_test, np.ones((len(X_test), 1))])
    preds_nsi = (X_bias @ macro.global_weights >= 0).astype(int)
    preds_fedavg = (X_bias @ gw_fedavg >= 0).astype(int)
    preds_fedprox = (X_bias @ gw_fedprox >= 0).astype(int)
    preds_trimmed = (X_bias @ gw_trimmed >= 0).astype(int)
    preds_krum = (X_bias @ gw_krum >= 0).astype(int)
    f1_nsi = f1_score(y_test, preds_nsi)
    f1_fedavg = f1_score(y_test, preds_fedavg)
    f1_fedprox = f1_score(y_test, preds_fedprox)
    f1_trimmed = f1_score(y_test, preds_trimmed)
    f1_krum = f1_score(y_test, preds_krum)
    return f1_nsi, f1_fedavg, f1_fedprox, f1_trimmed, f1_krum


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate NSI vs baseline evaluations over multiple seeds")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dt-depth", type=int, default=3)
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal regularisation parameter")
    args = parser.parse_args()
    f1_nsi_list = []
    f1_fedavg_list = []
    f1_fedprox_list = []
    f1_trimmed_list = []
    f1_krum_list = []
    for seed in range(args.seeds):
        res = evaluate_once(num_nodes=args.num_nodes,
                            rounds=args.rounds,
                            samples=args.samples,
                            input_dim=args.input_dim,
                            alpha=args.alpha,
                            lr=args.lr,
                            dt_depth=args.dt_depth,
                            mu=args.mu,
                            random_state=seed)
        f1_nsi_list.append(res[0])
        f1_fedavg_list.append(res[1])
        f1_fedprox_list.append(res[2])
        f1_trimmed_list.append(res[3])
        f1_krum_list.append(res[4])
        print(f"Seed {seed+1}/{args.seeds} completed.")
    def summarise(name: str, data: List[float]) -> str:
        arr = np.array(data)
        return f"{name}: {arr.mean():.4f} ± {arr.std():.4f}"
    print("\nF1‑score summary across seeds:\n")
    print(summarise("NSI", f1_nsi_list))
    print(summarise("FedAvg", f1_fedavg_list))
    print(summarise("FedProx", f1_fedprox_list))
    print(summarise("Trimmed", f1_trimmed_list))
    print(summarise("Krum", f1_krum_list))


if __name__ == "__main__":
    main()