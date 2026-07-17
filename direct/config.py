"""Hyper-parameter container for the DiReCT sample selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


ParamFilter = Callable[[str, nn.Parameter], bool]


@dataclass
class DiReCTConfig:
    """Configuration for :func:`direct.select_samples`.

    Attributes:
        K: Number of training samples to select.
        k_sketch: Dimension ``k`` of the Gaussian sketching matrix
            ``R`` with entries drawn from ``N(0, 1/k)``.
        elbow_energy: Cumulative-energy threshold used by
            :func:`direct.spectral_elbow` to partition the sketched spectrum
            into a stiff subspace ``I_perp`` and a flat subspace
            ``I_parallel`` (Section 5.3, default ``0.945``).
        tau_perp: *Relative* budget on the stiff-subspace energy per sample;
            the absolute budget passed to the solver is
            ``tau_perp * mean(a) * K`` where ``a_i`` is the stiff cost of
            sample ``i`` (Appendix A.2, default ``0.1``).
        seed: Seed for the Gaussian sketch. Fixing this makes selection
            reproducible.
        sca_max_iter: Maximum number of outer SCA iterations ``T_max``
            (Alg. 2).
        sca_tol: Convergence tolerance ``delta`` on ``||w^{t+1} - w^t||``.
        inner_iters: Number of inner projected-gradient-ascent steps used
            to approximately solve each SCA sub-problem.
        inner_lr: Step size for the inner PGA loop. The gradient direction
            is normalised to unit length so ``1.0`` is a reasonable default.
        device: PyTorch device on which sketching and eigendecomposition
            run.  Defaults to CUDA if available.
        dtype: Compute dtype for the sketched vectors.
        param_filter: Optional callable ``(name, param) -> bool``.  Only
            parameters for which it returns ``True`` contribute to the
            per-sample gradients.  ``None`` means all trainable parameters.
        micro_batch_size: How many val/train samples to *materialise* at a
            time.  Per-sample gradients themselves are still computed one
            at a time; this only controls printing frequency.
        verbose: If ``True``, print progress on stdout.
    """

    K: int
    k_sketch: int = 2048
    elbow_energy: float = 0.945
    tau_perp: float = 0.1
    seed: int = 0
    sca_max_iter: int = 20
    sca_tol: float = 1e-4
    inner_iters: int = 50
    inner_lr: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    param_filter: Optional[ParamFilter] = None
    micro_batch_size: int = 1
    verbose: bool = False
