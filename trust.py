"""Continuum Trust module for behavioural security in NSI.

This module implements a simple form of behavioural security check
based on Kullback–Leibler (KL) divergence between the current
gradient distribution and a baseline.  The intuition is borrowed
from the Continuum Trust component described in the paper: if the
distribution of gradient updates deviates significantly from
historical norms (e.g. due to poisoned updates), the macro can flag
the corresponding micro node as suspicious.

In a real system the baseline distribution could be learned over
time, and more sophisticated statistical tests (e.g. adaptive
thresholds, multivariate distributions) could be employed.  This
prototype uses univariate Gaussian approximations for simplicity.
"""

from __future__ import annotations

import numpy as np



def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the KL divergence KL(p || q) between two discrete
    probability distributions.

    Both ``p`` and ``q`` are expected to be one‑dimensional arrays
    summing to 1.  Zeros are handled by adding a small epsilon.
    """
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def continuum_trust(grad: np.ndarray, baseline: np.ndarray, threshold: float = 0.1) -> bool:
    """Determine whether a gradient update deviates significantly from baseline.

    The gradient is binned into a histogram with a fixed number of
    bins (e.g. 20), and the KL divergence between this histogram and
    a baseline histogram is computed.  If the divergence exceeds the
    specified threshold, the update is flagged as suspicious.

    Parameters
    ----------
    grad : numpy.ndarray
        One‑dimensional gradient update from a micro model (before
        quantisation).
    baseline : numpy.ndarray
        Baseline gradient distribution (same shape as ``grad``).  This
        baseline can be the average gradient from previous rounds.
    threshold : float, optional
        KL divergence threshold above which an update is considered
        anomalous.  Default is 0.1, following the paper.

    Returns
    -------
    suspicious : bool
        True if the KL divergence exceeds the threshold, indicating a
        potential poisoning attack; False otherwise.
    """
    # compute histograms with shared bins
    bins = np.linspace(-1.0, 1.0, 21)
    # normalise gradients to [-1,1] for histogramming
    max_abs = np.max(np.abs(np.concatenate([grad, baseline])))
    if max_abs == 0:
        return False
    g_norm = grad / max_abs
    b_norm = baseline / max_abs
    p_hist, _ = np.histogram(g_norm, bins=bins, density=True)
    q_hist, _ = np.histogram(b_norm, bins=bins, density=True)
    div = kl_divergence(p_hist, q_hist)
    return div > threshold