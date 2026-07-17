"""DiReCT: Directionally-Restrained Constrained Training.

A reference implementation of the sample-selection algorithm proposed in
"Towards Efficient LLMs Annealing with Principled Sample Selection"
(ICML 2026).

The public entry point is :func:`direct.select_samples`; :class:`DiReCTConfig`
holds every hyper-parameter documented in the paper's Appendix A.
"""

from .config import DiReCTConfig
from .selector import select_samples
from .sketch import GaussianSketcher
from .spectral import spectral_elbow
from .projection import dykstra_projection
from .sca import sca_solver

__all__ = [
    "DiReCTConfig",
    "select_samples",
    "GaussianSketcher",
    "spectral_elbow",
    "dykstra_projection",
    "sca_solver",
]

__version__ = "0.1.0"
