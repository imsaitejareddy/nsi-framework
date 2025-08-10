"""Standalone simulation script for the Neuro‑Symbiotic Intelligence (NSI) framework.

This module demonstrates how to run a small NSI experiment outside of
the package API.  It includes definitions for several helper
functions—``quantize_tensor``, ``dequantize_tensor`` and
``sparsify_top_k``—that are used to compress model updates.  The
simulation itself instantiates micro and macro models from the
``nsi_framework`` package, generates a synthetic dataset using a
Gaussian mixture model, partitions the data in a non‑IID fashion and
executes a few communication rounds.  During training a simple
Byzantine adversary flips the sign of one micro node’s gradient and
scales it to simulate an attack.  At the end of each round the
script reports accuracy, F1‑score and approximate bandwidth usage for
both NSI and a naive federated learning baseline.

The purpose of this script is to mirror the experiments described in
the NSI paper using readily available Python libraries.  It can be
run directly with ``python nsi_simulation.py`` or imported as a
module to expose the ``main`` function.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from nsi_framework.micro import MicroLLM
from nsi_framework.macro import MacroLLM


def quantize_tensor(tensor: np.ndarray, bit_width: int = 8) -> Tuple[np.ndarray, float]:
    """Quantise a floating‑point vector to signed integers.

    Parameters
    ----------
    tensor : numpy.ndarray
        Array of floats to quantise.
    bit_width : int, optional
        Number of bits for quantisation.  Only 8‑bit quantisation is
        supported in this prototype.  Default is 8.

    Returns
    -------
    q : numpy.ndarray
        Quantised values with dtype ``np.int8``.
    scale : float
        Maximum absolute value of the original tensor used to
        dequantise.  If the maximum is zero the scale will be 1.0.

    Notes
    -----
    The quantisation rescales the input to the range [‑1, 1] and then
    multiplies by ``2^(bit_width‑1)‑1`` before rounding.  To
    reconstruct an approximate float the inverse transform divides by
    the same factor and multiplies by the saved ``scale``.
    """
    if bit_width != 8:
        raise ValueError("Only 8‑bit quantisation is supported in this prototype")
    max_abs = float(np.max(np.abs(tensor)))
    if max_abs == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    normalised = tensor / max_abs
    scaled = normalised * 127
    q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
    return q, max_abs


def dequantize_tensor(q: np.ndarray, scale: float, bit_width: int = 8) -> np.ndarray:
    """Reconstruct an approximate floating‑point vector from a quantised one.

    Parameters
    ----------
    q : numpy.ndarray
        Array of quantised values (dtype ``np.int8``).
    scale : float
        Scale factor returned by ``quantize_tensor``.
    bit_width : int, optional
        Number of bits used for quantisation.  Only 8‑bit is
        implemented.  Default is 8.

    Returns
    -------
    x : numpy.ndarray
        Approximate reconstruction of the original floating‑point
        vector.
    """
    if bit_width != 8:
        raise ValueError("Only 8‑bit dequantisation is supported in this prototype")
    if scale == 0:
        return np.zeros_like(q, dtype=float)
    return (q.astype(float) / 127.0) * scale


def sparsify_top_k(vector: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top‑k magnitude entries of a vector and their indices.

    Parameters
    ----------
    vector : numpy.ndarray
        Input vector from which to select large entries.
    ratio : float
        Fraction of entries to keep.  Must be in (0, 1].

    Returns
    -------
    values : numpy.ndarray
        Array of the selected values.
    indices : numpy.ndarray
        Indices of the selected values in the original vector.
    """
    d = len(vector)
    k = max(1, int(d * ratio))
    idx = np.argpartition(np.abs(vector), -k)[-k:]
    return vector[idx], idx


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Partition labels into non‑IID subsets using a Dirichlet distribution."""
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    per_client: List[List[int]] = [[] for _ in range(num_clients)]
    for idxs in class_indices:
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * len(idxs)).astype(int)
        counts[-1] = len(idxs) - np.sum(counts[:-1])
        start = 0
        for client_id, count in enumerate(counts):
            per_client[client_id].extend(idxs[start:start + count].tolist())
            start += count
    return [np.array(sorted(id_list)) for id_list in per_client]


def generate_gaussian_data(samples: int, input_dim: int, random_state: int | None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset using Gaussian mixtures.

    The positive class is drawn from a mixture of two Gaussians and the
    negative class from a single Gaussian.  The means and covariances
    are chosen randomly based on the provided seed.
    """
    rng = np.random.default_rng(random_state)
    # positive class: mixture of two clusters
    means_pos = rng.uniform(-2.0, 2.0, size=(2, input_dim))
    covs_pos = np.array([np.eye(input_dim) * rng.uniform(0.5, 1.5) for _ in range(2)])
    comp_weights = [0.6, 0.4]
    gm_pos = GaussianMixture(n_components=2, weights_init=comp_weights,
                             means_init=means_pos, precisions_init=np.linalg.inv(covs_pos))
    gm_pos.fit(np.vstack([rng.multivariate_normal(means_pos[i], covs_pos[i], size=100) for i in range(2)]))
    X_pos, _ = gm_pos.sample(int(samples * 0.1))  # minority class
    # negative class: single cluster
    mean_neg = rng.uniform(-2.0, 2.0, size=input_dim)
    cov_neg = np.eye(input_dim) * rng.uniform(0.5, 1.5)
    X_neg = rng.multivariate_normal(mean_neg, cov_neg, size=int(samples * 0.9))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos), dtype=int), np.zeros(len(X_neg), dtype=int)])
    # shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def simulate(num_nodes: int = 10,
             rounds: int = 5,
             samples: int = 6000,
             input_dim: int = 10,
             alpha: float = 0.7,
             top_k_ratio: float = 0.1,
             prune_threshold: float = 0.5,
             lr: float = 0.01,
             dt_depth: int = 3,
             random_state: int | None = None) -> None:
    """Run the NSI simulation using Gaussian mixture data and report metrics.

    A simple Byzantine adversary is simulated by selecting one micro
    node and flipping the sign of its gradient and scaling it by 10x.
    This tests the macro’s resilience to poisoned updates.
    """
    rng = np.random.default_rng(random_state)
    # generate data
    X, y = generate_gaussian_data(samples, input_dim, random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=random_state)
    # partition labels into non‑IID shards
    client_indices = dirichlet_partition(y_train, num_nodes, alpha, rng)
    # instantiate models
    classes = list(np.unique(y_train))
    micros: List[MicroLLM] = []
    for i in range(num_nodes):
        micros.append(MicroLLM(node_id=i, input_dim=input_dim, classes=classes,
                               lr=lr, top_k_ratio=top_k_ratio, dt_depth=dt_depth,
                               random_state=(None if random_state is None else random_state + i)))
    macro = MacroLLM(num_nodes=num_nodes, input_dim=input_dim, prune_threshold=prune_threshold)
    # baseline FL model
    fl_global = np.zeros(input_dim + 1)
    # results storage
    acc_nsi: List[float] = []
    f1_nsi: List[float] = []
    acc_fl: List[float] = []
    f1_fl: List[float] = []
    nsi_bw: List[int] = []
    fl_bw: List[int] = []
    # simulation
    for r in range(rounds):
        capsules = []
        full_grads = []
        # choose a random adversarial node for this round
        adv_idx = rng.integers(0, num_nodes)
        for i, node in enumerate(micros):
            idx = client_indices[i]
            X_local = X_train[idx]
            y_local = y_train[idx]
            # local training
            node.train_local(X_local, y_local, epochs=1)
            # produce capsule
            cap = node.generate_capsule(X_local, y_local, top_k_ratio=top_k_ratio)
            # apply adversarial manipulation
            if i == adv_idx:
                # flip sign and scale gradient by 10x
                cap.G_q = (-cap.G_q.astype(int) * 10).astype(np.int8)
            capsules.append(cap)
            # compute full gradient for FL baseline
            w = node._get_weights()
            grad = w - fl_global
            full_grads.append(grad)
        # macro operations
        R = macro.heuristic_resonance(capsules)
        macro.dagp(R)
        macro.aggregate(capsules)
        # update FL baseline
        fl_grad = np.mean(np.stack(full_grads, axis=0), axis=0)
        fl_global += fl_grad
        # evaluate on test
        X_bias = np.hstack([X_test, np.ones((len(X_test), 1))])
        preds_nsi = (X_bias @ macro.global_weights >= 0).astype(int)
        preds_fl = (X_bias @ fl_global >= 0).astype(int)
        acc_nsi.append(accuracy_score(y_test, preds_nsi))
        f1_nsi.append(f1_score(y_test, preds_nsi))
        acc_fl.append(accuracy_score(y_test, preds_fl))
        f1_fl.append(f1_score(y_test, preds_fl))
        # bandwidth calculation
        d = input_dim + 1
        k = max(1, int(top_k_ratio * d))
        bw_capsule = k * 4 + d * 1 + d * 1
        nsi_bw.append(bw_capsule * num_nodes)
        fl_bw.append((d * 4) * num_nodes)
        print(f"Round {r+1}/{rounds}: acc_nsi={acc_nsi[-1]:.4f}, f1_nsi={f1_nsi[-1]:.4f}, acc_fl={acc_fl[-1]:.4f}, f1_fl={f1_fl[-1]:.4f}")
    # summary
    def summarise(name: str, vals: List[float]) -> None:
        arr = np.array(vals)
        print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
    print("\nSimulation summary:\n")
    summarise("NSI Accuracy", acc_nsi)
    summarise("NSI F1", f1_nsi)
    summarise("FL Accuracy", acc_fl)
    summarise("FL F1", f1_fl)
    print(f"Bandwidth NSI per round: {nsi_bw[0]/1e3:.2f} kB per node")
    print(f"Bandwidth FL per round: {fl_bw[0]/1e3:.2f} kB per node")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NSI simulation with helper functions.")
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top-k-ratio", type=float, default=0.1)
    parser.add_argument("--prune-threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dt-depth", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()
    simulate(num_nodes=args.num_nodes,
             rounds=args.rounds,
             samples=args.samples,
             input_dim=args.input_dim,
             alpha=args.alpha,
             top_k_ratio=args.top_k_ratio,
             prune_threshold=args.prune_threshold,
             lr=args.lr,
             dt_depth=args.dt_depth,
             random_state=args.random_state)


if __name__ == "__main__":
    main()
