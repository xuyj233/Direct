"""Dykstra alternating projection onto the DiReCT feasible set.

The feasible set is the intersection of three convex sets

    - the unit box :math:`[0, 1]^N`;
    - the cardinality hyperplane :math:`\\{w : \\mathbf{1}^\\top w = K\\}`;
    - the stiff-budget half-space :math:`\\{w : a^\\top w \\le \\tau\\}`.

Since the projection onto each single set has a closed form, Dykstra's
algorithm converges to the exact Euclidean projection onto the
intersection.
"""

from __future__ import annotations

import numpy as np


def _proj_box(w: np.ndarray) -> np.ndarray:
    """Project ``w`` onto :math:`[0, 1]^N`."""
    return np.clip(w, 0.0, 1.0)


def _proj_sum(w: np.ndarray, K: float) -> np.ndarray:
    """Project ``w`` onto :math:`\\{w : \\mathbf{1}^\\top w = K\\}`."""
    return w + (K - w.sum()) / len(w)


def _proj_halfspace(w: np.ndarray, a: np.ndarray, tau: float) -> np.ndarray:
    """Project ``w`` onto :math:`\\{w : a^\\top w \\le \\tau\\}`."""
    v = float(a @ w) - tau
    if v <= 0.0:
        return w
    denom = float(a @ a)
    if denom <= 0.0:
        return w
    return w - (v / denom) * a


def dykstra_projection(
    w: np.ndarray,
    a: np.ndarray,
    tau: float,
    K: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Project ``w`` onto ``[0, 1]^N ∩ {1ᵀw = K} ∩ {aᵀw ≤ tau}``.

    Args:
        w: Point to project (1-D array of length ``N``).
        a: Coefficients of the half-space (1-D array of length ``N``).
        tau: Right-hand side of the half-space constraint.
        K: Target cardinality; the returned vector satisfies
            ``sum(w) = K`` up to numerical tolerance.
        max_iter: Maximum outer sweeps over the three constraints.
        tol: Stopping tolerance on relative change of ``w``.

    Returns:
        A vector in the intersection, of the same shape as ``w``.
    """
    x = np.asarray(w, dtype=np.float64).copy()
    a = np.asarray(a, dtype=np.float64)
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    r = np.zeros_like(x)

    for _ in range(max_iter):
        # Box projection
        y = _proj_box(x + p)
        p = x + p - y
        # Cardinality hyperplane
        z = _proj_sum(y + q, K)
        q = y + q - z
        # Stiff-budget half-space
        x_new = _proj_halfspace(z + r, a, tau)
        r = z + r - x_new

        if np.linalg.norm(x_new - x) < tol * (1.0 + np.linalg.norm(x)):
            x = x_new
            break
        x = x_new
    return x
