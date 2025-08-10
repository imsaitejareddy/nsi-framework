"""Implementation of the macro‑scale coordinator for NSI.

The ``MacroLLM`` orchestrates communication among micro nodes,
performs heuristic resonance analysis, prunes the aptitude graph via
DAGP, and optionally aggregates model updates.  In a real NSI
deployment the macro might be a large language model hosted in the
cloud.  Here we represent it with simple numpy operations on
vectors contained in ``CognitionCapsule`` objects.

Primary responsibilities:

* Maintain an adjacency matrix representing connections between micro
  nodes.  Initially the graph is fully connected.  DAGP will prune
  low‑resonance edges by setting their adjacency entry to zero.  For
  simplicity each node communicates directly with the macro; edges
  between micro nodes are conceptual, representing the macro’s
  willingness to share information between those nodes in future
  rounds.
* Compute heuristic resonance scores based on pairwise cosine
  similarity of the ``H_p`` vectors in received capsules.  These
  similarity values are used by DAGP to decide which edges to prune.
* Aggregate quantised gradients and (optionally) sparse embeddings
  from micro nodes.  The default aggregation is a simple average of
  dequantised gradients.  More sophisticated methods (e.g. weighted
  by resonance) could be implemented.
* Assign resonance scores ``R`` back to the capsules for logging and
  potential use by micro nodes.

The macro does not train its own model in this prototype; instead it
updates a shared weight vector using aggregated gradients.  This
weight vector could be used to compute global predictions or to
initialise micro models between communication rounds.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
from numpy.linalg import norm

from .capsule import CognitionCapsule


class MacroLLM:
    """Coordinator for the NSI ecosystem.

    Parameters
    ----------
    num_nodes : int
        Number of micro nodes in the system.
    input_dim : int
        Dimensionality of the weight vectors (coef + bias).
    prune_threshold : float, optional
        Threshold on resonance for pruning edges in DAGP.  A value of
        0.5 corresponds to Algorithm 1 in the paper.  Default is 0.5.
    use_resonance_weights : bool, optional
        If True, weight gradient aggregation by each capsule’s
        resonance; otherwise use a simple average.  Default is False.

    Notes
    -----
    The macro maintains a weight vector ``global_weights`` whose
    dimension equals ``input_dim + 1`` (to match the micro weights).  At
    each communication round it aggregates dequantised gradient
    estimates from capsules and applies them to ``global_weights``.
    """

    def __init__(self,
                 num_nodes: int,
                 input_dim: int,
                 prune_threshold: float = 0.5,
                 use_resonance_weights: bool = False) -> None:
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        # adjacency matrix for DAGP; 1 indicates active edge
        self.adj = np.ones((num_nodes, num_nodes), dtype=np.int8)
        np.fill_diagonal(self.adj, 0)  # no self‑edges
        self.prune_threshold = prune_threshold
        self.use_resonance_weights = use_resonance_weights
        # global weight vector (including bias) initialised to zeros
        self.global_weights = np.zeros(input_dim + 1)

    def _dequantise_gradient(self, q: np.ndarray) -> np.ndarray:
        """Reconstruct an approximate gradient from an int8 vector.

        Since micro nodes normalise their gradient by its maximum
        absolute value and multiply by 127, we cannot recover the
        original scaling factor.  We therefore scale by the maximum of
        ``|q|`` to approximate the gradient magnitude.  This is
        sufficient for demonstration but not identical to the true
        gradient.  In practice one can send the scale factor as an
        additional scalar in the capsule.
        """
        max_abs = np.max(np.abs(q))
        if max_abs == 0:
            return np.zeros_like(q, dtype=float)
        # invert the normalisation: q = round(grad/max_grad*127)
        # approximate grad ≈ q / 127 * max_abs
        return (q.astype(float) / 127.0) * max_abs

    def heuristic_resonance(self, capsules: List[CognitionCapsule]) -> np.ndarray:
        """Compute heuristic resonance matrix from capsules.

        Resonance between two capsules is defined as the cosine
        similarity of their heuristic vectors ``H_p``.  The result is
        a symmetric matrix ``R`` of shape (n, n).  Self‑similarities
        are set to zero.  If a capsule’s ``H_p`` vector is all zeros
        (no features used), its row and column will be all zeros.

        The resonance scores are also assigned back into the
        ``CognitionCapsule.R`` field as the average resonance of that
        capsule with all others.  This value may be used later for
        weighted aggregation.

        Parameters
        ----------
        capsules : list of CognitionCapsule
            Capsules received from micro nodes.

        Returns
        -------
        R : numpy.ndarray
            Pairwise resonance matrix with entries in [0, 1].
        """
        n = len(capsules)
        R = np.zeros((n, n))
        Hp = [c.H_p.astype(float) for c in capsules]
        norms = [norm(h) if norm(h) > 0 else 1.0 for h in Hp]
        for i in range(n):
            for j in range(i + 1, n):
                if norms[i] == 0 or norms[j] == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(Hp[i], Hp[j]) / (norms[i] * norms[j]))
                R[i, j] = sim
                R[j, i] = sim
        # assign average resonance to capsules
        for i, c in enumerate(capsules):
            c.R = float(np.mean(R[i, :])) if n > 1 else 0.0
        return R

    def dagp(self, R: np.ndarray) -> None:
        """Prune low‑resonance edges in the adjacency matrix.

        Implements Algorithm 1 from the paper: for each edge (i, j)
        where i < j, compute the resonance score R[i,j].  If it is
        below the threshold ``self.prune_threshold``, remove the edge by
        setting ``adj[i,j]`` and ``adj[j,i]`` to zero.  Otherwise the
        edge remains.  The pruning applies in place and updates
        ``self.adj``.

        Parameters
        ----------
        R : numpy.ndarray
            Pairwise resonance matrix computed by ``heuristic_resonance``.
        """
        n = R.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if R[i, j] < self.prune_threshold:
                    self.adj[i, j] = 0
                    self.adj[j, i] = 0
                else:
                    self.adj[i, j] = 1
                    self.adj[j, i] = 1

    def aggregate(self, capsules: List[CognitionCapsule]) -> None:
        """Aggregate quantised gradients from capsules and update global weights.

        The aggregation uses a weighted average if ``use_resonance_weights``
        is True, in which case each gradient is weighted by the
        capsule’s resonance ``R``.  Otherwise, a simple arithmetic mean
        is used.  Note that dequantisation yields an approximate
        gradient; in a full implementation the scale factor would be
        transmitted alongside the quantised values.

        Parameters
        ----------
        capsules : list of CognitionCapsule
            Capsules received from micro nodes.  Must contain the
            ``G_q`` field.
        """
        grads = [self._dequantise_gradient(c.G_q) for c in capsules]
        grads = np.stack(grads, axis=0)
        if self.use_resonance_weights:
            weights = np.array([c.R for c in capsules], dtype=float)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(capsules)) / len(capsules)
            agg_grad = np.sum(grads * weights[:, None], axis=0)
        else:
            agg_grad = np.mean(grads, axis=0)
        # apply update to global weights
        self.global_weights += agg_grad

    def get_pruned_adjacency(self) -> np.ndarray:
        """Return a copy of the current adjacency matrix."""
        return self.adj.copy()