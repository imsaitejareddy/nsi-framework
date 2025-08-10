"""NSI Framework package.

This package provides a lightweight research prototype of the
Neuro‑Symbiotic Intelligence (NSI) framework described in the paper
"Neuro‑Symbiotic Intelligence: A Convergent Learning Ecosystem via
Co‑Evolution of Micro‑ and Macro‑Scale Language Models".  The goal of
this code is to capture the key concepts of NSI – cognition capsules,
heuristic resonance analysis, dynamic aptitude graph pruning (DAGP),
continuum trust, and a simple co‑evolutionary training loop – in a
form that can be executed on modest hardware.  The implementation
relies solely on numpy and scikit‑learn to avoid heavy dependencies.

The simulation is not intended to reproduce the exact scale of the
experiments in the paper (which use 100M–1B parameter language
models); instead it uses small linear models and decision trees to
illustrate how the micro and macro components interact.  When run
with the provided ``simulate.py`` script, the framework generates
synthetic non‑IID data, trains multiple micro models, exchanges
cognition capsules with a macro coordinator, prunes low‑resonance
connections via DAGP, and evaluates accuracy, F1 and bandwidth/latency
savings relative to a naive federated learning baseline.
"""

from .capsule import CognitionCapsule
from .micro import MicroLLM
from .macro import MacroLLM
from .dagp import dagp_prune
from .trust import continuum_trust

__all__ = [
    "CognitionCapsule",
    "MicroLLM",
    "MacroLLM",
    "dagp_prune",
    "continuum_trust",
]