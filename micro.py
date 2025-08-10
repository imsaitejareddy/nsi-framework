"""Implementation of micro‑scale models for the NSI framework.

Each micro node in the NSI ecosystem hosts a small language model or
classifier tailored to local data and resource constraints.  In this
research prototype we use an ``SGDClassifier`` from scikit‑learn with
logistic loss, which approximates a quantised transformer in the
sense that it learns a linear decision boundary and supports
incremental updates.  The ``MicroLLM`` class encapsulates local
training, model state management and extraction of cognition capsules.

Key responsibilities:

* Maintain a linear classifier that can be updated via mini‑batch
  gradient descent (``partial_fit`` in scikit‑learn).  For real
  LLMs you would substitute a quantised transformer here.
* Track previous weights to compute gradient deltas on each
  communication round.  This is necessary for building the quantised
  gradient component of the capsule.
* Train a shallow decision tree on local data to extract heuristic
  information (``H_p``).  A binary vector of length ``input_dim``
  records which features were used at least once in the tree.
* Produce a ``CognitionCapsule`` with compressed weight embedding
  (``H_a``), quantised gradient (``G_q``) and heuristics (``H_p``).

The micro model does not perform resonance analysis; that is handled
by the ``MacroLLM``.  Instead, capsules are passed to the macro and
returned with updated ``R`` values.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from .capsule import CognitionCapsule


class MicroLLM:
    """Lightweight micro model for edge training.

    Parameters
    ----------
    node_id : int
        Unique identifier for this micro node.
    input_dim : int
        Dimensionality of the input features.  Determines the length of
        the weight vector and heuristic vector.
    classes : List[int]
        List of class labels expected by the classifier.  Needed for
        ``partial_fit`` in scikit‑learn.
    lr : float, optional
        Learning rate for the SGDClassifier.  Default is 0.01.
    top_k_ratio : float, optional
        Fraction of largest weight magnitudes to include in ``H_a``.
        For example, 0.1 means 10% of the dimensions will be
        transmitted.  Default is 0.1.
    dt_depth : int, optional
        Maximum depth of the decision tree used to extract heuristics.
        Default is 3.
    random_state : int, optional
        Seed for reproducibility.  Default is None.

    Notes
    -----
    The classifier is initialised with ``max_iter=1`` and ``warm_start=True``
    to ensure that each call to ``partial_fit`` performs a single pass over
    the data, emulating one local epoch.  A ``StandardScaler`` is used
    to normalise features because scikit‑learn’s SGDClassifier does not
    include internal normalisation.  Without scaling, gradient updates
    can become unstable.
    """

    def __init__(self,
                 node_id: int,
                 input_dim: int,
                 classes: List[int],
                 lr: float = 0.01,
                 top_k_ratio: float = 0.1,
                 dt_depth: int = 3,
                 random_state: int | None = None) -> None:
        self.id = node_id
        self.input_dim = input_dim
        self.classes = np.array(classes)
        self.top_k_ratio = top_k_ratio
        self.random_state = random_state
        self.dt_depth = dt_depth
        # scaler to normalise features locally
        self.scaler = StandardScaler()
        # initialise classifier; logistic loss approximates binary cross‑entropy
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant',
                                   eta0=lr, max_iter=1, warm_start=True,
                                   random_state=random_state)
        self.model_initialized = False
        # track previous weights to compute gradient deltas
        self.prev_weights: np.ndarray | None = None
        self.prev_intercept: np.ndarray | None = None

    def _get_weights(self) -> np.ndarray:
        """Concatenate coefficients and intercept to form a flat vector.

        Returns
        -------
        w : numpy.ndarray
            Weight vector of shape (d,) where d = input_dim + 1 (bias).
        """
        coef = self.model.coef_.reshape(-1)
        intercept = self.model.intercept_.reshape(-1)
        return np.concatenate([coef, intercept])

    def train_local(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        """Perform local training on a dataset for a number of epochs.

        Data is shuffled and scaled before each epoch.  After training,
        the internal scaler retains the mean and variance to scale
        subsequent data passed to ``predict``.  ``prev_weights`` and
        ``prev_intercept`` are updated after training to track the
        previous state for gradient delta computation.

        Parameters
        ----------
        X : numpy.ndarray
            Local features, shape (n_samples, input_dim).
        y : numpy.ndarray
            Corresponding labels, shape (n_samples,).
        epochs : int, optional
            Number of passes over the local data.  Default is 1.
        """
        # Fit scaler on first call
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(X)
        for _ in range(epochs):
            X_shuf, y_shuf = shuffle(X, y, random_state=self.random_state)
            X_scaled = self.scaler.transform(X_shuf)
            if not self.model_initialized:
                # first call needs classes for partial_fit
                self.model.partial_fit(X_scaled, y_shuf, classes=self.classes)
                self.model_initialized = True
            else:
                self.model.partial_fit(X_scaled, y_shuf)
        # update previous weights after training
        w = self._get_weights()
        if self.prev_weights is None:
            self.prev_weights = w.copy()
        else:
            # leave prev_weights unchanged; updated after capsule extraction
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given samples."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def _quantise_gradient(self, grad: np.ndarray) -> np.ndarray:
        """Quantise a gradient vector to 8‑bit integers.

        The gradient is first normalised to the range [-1, 1] and then
        scaled to the int8 range [-128, 127].  Values outside this range
        are clipped.  On the receiving side the macro model can
        reconstruct an approximate floating‑point gradient by applying
        the inverse transform.

        Parameters
        ----------
        grad : numpy.ndarray
            Gradient vector to quantise.

        Returns
        -------
        q : numpy.ndarray
            Quantised gradient with dtype ``np.int8``.
        """
        if np.allclose(grad, 0):
            return np.zeros_like(grad, dtype=np.int8)
        max_abs = np.max(np.abs(grad))
        # avoid division by zero
        if max_abs == 0:
            return np.zeros_like(grad, dtype=np.int8)
        normalised = grad / max_abs
        scaled = normalised * 127
        q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        return q

    def _extract_heuristics(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train a shallow decision tree to extract feature usage heuristics.

        The tree is trained on scaled data with maximum depth ``dt_depth``.
        A binary vector of length ``input_dim`` is returned where a 1
        indicates that a feature was used in at least one split of the
        tree.  If no tree is built (e.g., all labels the same), the
        vector will be all zeros.

        Parameters
        ----------
        X : numpy.ndarray
            Local features (scaled).
        y : numpy.ndarray
            Corresponding labels.

        Returns
        -------
        h_p : numpy.ndarray
            Binary heuristic vector of length ``input_dim``.
        """
        X_scaled = self.scaler.transform(X)
        # guard against trivial data where tree cannot split
        try:
            dt = DecisionTreeClassifier(max_depth=self.dt_depth, random_state=self.random_state)
            dt.fit(X_scaled, y)
            # dt.tree_.feature stores indices of features used in splits;
            # -2 indicates a leaf
            used_features = dt.tree_.feature
            h_p = np.zeros(self.input_dim, dtype=np.int8)
            for feat in used_features:
                if feat >= 0 and feat < self.input_dim:
                    h_p[feat] = 1
            return h_p
        except Exception:
            return np.zeros(self.input_dim, dtype=np.int8)

    def generate_capsule(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         top_k_ratio: float | None = None) -> CognitionCapsule:
        """Construct a cognition capsule to send to the macro model.

        Parameters
        ----------
        X : numpy.ndarray
            Local features used to extract heuristics.  Should be the
            same data used for the most recent local training round.
        y : numpy.ndarray
            Corresponding labels.
        top_k_ratio : float, optional
            Override the instance’s ``top_k_ratio`` for this capsule.

        Returns
        -------
        capsule : CognitionCapsule
            Capsule containing compressed weight embedding, quantised
            gradient, heuristics and node identifier.
        """
        if top_k_ratio is None:
            top_k_ratio = self.top_k_ratio
        w = self._get_weights()
        # compute gradient update relative to previous weights
        if self.prev_weights is None:
            grad = np.zeros_like(w)
        else:
            grad = w - self.prev_weights
        q_grad = self._quantise_gradient(grad)
        # update prev_weights for next round
        self.prev_weights = w.copy()
        # build sparse embedding H_a
        d = len(w)
        k = max(1, int(top_k_ratio * d))
        # indices of top‑k magnitude weights
        idx = np.argpartition(np.abs(w), -k)[-k:]
        H_a = w[idx]
        # heuristics vector H_p
        H_p = self._extract_heuristics(X, y)
        return CognitionCapsule(id=self.id, H_a=H_a, G_q=q_grad, H_p=H_p, R=0.0)