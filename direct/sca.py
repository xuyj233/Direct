"""Successive Convex Approximation solver (Algorithm 2 of the paper)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .projection import dykstra_projection


def sca_solver(
    G: np.ndarray,
    a: np.ndarray,
    K: int,
    tau_perp: float,
    w0: Optional[np.ndarray] = None,
    max_iter: int = 20,
    tol: float = 1e-4,
    inner_lr: float = 1.0,
    inner_iters: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """Solve the relaxed DiReCT selection problem.

    Approximately maximises

    .. math::
        J(w) = \\|G w\\|_2^2
        \\quad \\text{s.t.} \\quad
        w \\in [0, 1]^N, \\; \\mathbf{1}^\\top w = K, \\;
        a^\\top w \\le \\tau_\\perp

    using Successive Convex Approximation (Alg. 2 of the paper).  At each
    outer iteration ``t`` a linear surrogate

    .. math::
        c^{(t)} = 2 G^\\top (G w^{(t)})

    is maximised over the same feasible set via a projected-gradient-ascent
    inner loop with Dykstra projection.

    Args:
        G: ``|I_parallel| x N`` matrix whose column ``i`` is
            ``[g_{i, j}]_{j \\in I_parallel}``.
        a: Length-``N`` vector of stiff-subspace costs
            :math:`a_i = \\sum_{j \\in I_\\perp} \\lambda_j g_{i,j}^2`.
        K: Target cardinality.
        tau_perp: Absolute budget on the stiff subspace.
        w0: Warm-start selection weights.  Defaults to the uniform vector
            ``K/N``.
        max_iter: Maximum number of outer SCA iterations.
        tol: Stop when ``||w^{t+1} - w^t|| < tol``.
        inner_lr: Step size for the inner PGA loop (gradient is normalised).
        inner_iters: Number of inner PGA steps per outer iteration.
        verbose: If ``True``, print the objective at every outer step.

    Returns:
        A continuous vector ``w* in [0, 1]^N`` approximately maximising ``J``.
    """
    N = G.shape[1]
    w = np.full(N, K / N, dtype=np.float64) if w0 is None else np.asarray(w0, dtype=np.float64).copy()

    for t in range(max_iter):
        p_t = G @ w
        c_t = 2.0 * (G.T @ p_t)
        norm_c = float(np.linalg.norm(c_t)) + 1e-12

        w_new = w.copy()
        for _ in range(inner_iters):
            w_new = w_new + inner_lr * c_t / norm_c
            w_new = dykstra_projection(w_new, a, tau_perp, K)

        diff = float(np.linalg.norm(w_new - w))
        if verbose:
            obj = float(np.linalg.norm(G @ w_new) ** 2)
            print(f"[SCA] iter={t:02d}  obj={obj:.4e}  |dw|={diff:.4e}")
        w = w_new
        if diff < tol:
            break
    return w
