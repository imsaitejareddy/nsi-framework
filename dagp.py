"""Standalone implementation of Dynamic Aptitude Graph Pruning (DAGP).

While the ``MacroLLM.dagp`` method in ``macro.py`` applies DAGP to a
macro instance’s adjacency matrix, this module provides a pure
function that can operate on arbitrary graphs.  It is useful for
unit testing or for applying DAGP outside of the macro context.

The algorithm follows Algorithm 1 in the paper:

```
Input: graph G = (V, E), resonance matrix R
For each edge e ∈ E:
    if R(e) < threshold:
        remove e from E
Output: pruned graph
```

In this implementation the graph is represented by an adjacency
matrix ``adj`` where ``adj[i,j] = 1`` indicates an edge between nodes
``i`` and ``j``.  ``R`` is a square matrix of the same size with
pairwise resonance scores.  ``threshold`` is a scalar.  The function
returns a new adjacency matrix with low‑resonance edges removed.
"""

from __future__ import annotations

import numpy as np


def dagp_prune(adj: np.ndarray, R: np.ndarray, threshold: float) -> np.ndarray:
    """Prune edges from an adjacency matrix based on resonance.

    Parameters
    ----------
    adj : numpy.ndarray
        Square adjacency matrix representing the graph.  Assumed to be
        symmetric and with zeros on the diagonal.
    R : numpy.ndarray
        Pairwise resonance matrix.  Must be the same shape as ``adj``.
    threshold : float
        Minimum resonance required to keep an edge.

    Returns
    -------
    pruned_adj : numpy.ndarray
        New adjacency matrix with the same shape as ``adj`` where
        edges with resonance below ``threshold`` are removed (set to 0).
    """
    assert adj.shape == R.shape, "adjacency and resonance matrices must have the same shape"
    pruned = adj.copy().astype(np.int8)
    n = pruned.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if R[i, j] < threshold:
                pruned[i, j] = 0
                pruned[j, i] = 0
            else:
                pruned[i, j] = 1
                pruned[j, i] = 1
    return pruned