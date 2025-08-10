"""Definition of cognition capsules used for micro–macro communication.

In the NSI framework, micro models at the edge send compressed
representations of their state and heuristics to the macro model in
the form of ``CognitionCapsule`` objects.  Each capsule contains
several components:

* ``H_a``: a sparse embedding vector capturing the most salient
  directions in the micro model’s parameter space.  Following the
  paper, this is obtained by taking the top‑k magnitude elements of
  the model’s weight vector; the rest are omitted to reduce
  bandwidth.
* ``G_q``: an 8‑bit quantized version of the model’s gradient update
  (difference between the current and previous weights).  Quantisation
  is used to minimise communication cost without significantly
  harming downstream performance.
* ``H_p``: a binary heuristic vector extracted from a shallow decision
  tree trained on the micro node’s data.  The vector has length
  ``input_dim`` and indicates which features are used in any of the
  tree’s decision rules.
* ``R``: the resonance score assigned by the macro model after
  performing heuristic resonance analysis.  Initially this is zero
  and is updated by the macro.
* ``id``: identifier of the micro node that produced the capsule.

This class is a simple dataclass with no behaviour beyond storing the
fields.  The micro and macro classes are responsible for populating
and consuming these capsules.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class CognitionCapsule:
    """Container for compressed micro model state and heuristics.

    Parameters
    ----------
    id : int
        Unique identifier of the micro node.
    H_a : numpy.ndarray
        Sparse embedding (top‑k elements of the weight vector).  Shape
        is (k,), where k << d (d is the full parameter dimension).
    G_q : numpy.ndarray
        Quantised gradient update.  This is an int8 vector of the same
        length as the full weight vector, but stored as a numpy array
        with dtype ``np.int8`` to reduce communication cost.
    H_p : numpy.ndarray
        Binary heuristic vector extracted from a decision tree.  Shape
        is (d,) and entries are 0 or 1 indicating whether each input
        dimension participates in any split in the tree.
    R : float
        Resonance value assigned by the macro model.  Initially zero
        and updated after heuristic resonance analysis.

    Notes
    -----
    The capsule intentionally stores compressed information rather
    than the entire model or gradient.  The paper reports that such
    compression yields >99% bandwidth savings relative to standard
    federated learning updates.  For simplicity, this prototype does
    not implement entropy coding; rather, quantisation and sparsity
    provide approximate savings.
    """

    id: int
    H_a: np.ndarray
    G_q: np.ndarray
    H_p: np.ndarray
    R: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation of the capsule.

        This can be useful when logging or serialising the capsule for
        network transmission.  Large arrays are converted to lists of
        Python floats/ints.
        """
        return {
            "id": int(self.id),
            "H_a": self.H_a.tolist(),
            "G_q": self.G_q.astype(int).tolist(),
            "H_p": self.H_p.astype(int).tolist(),
            "R": float(self.R),
        }