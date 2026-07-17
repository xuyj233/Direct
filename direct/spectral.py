"""Spectral elbow used to split ``I_perp`` (stiff) from ``I_parallel`` (flat)."""

from __future__ import annotations

import numpy as np


def spectral_elbow(eigvals: np.ndarray, energy: float = 0.945) -> int:
    """Locate the elbow index of a descending spectrum.

    Given eigenvalues :math:`\\lambda_1 \\ge \\lambda_2 \\ge \\dots \\ge \\lambda_k \\ge 0`,
    return the smallest ``k*`` such that

    .. math::
        \\frac{\\sum_{i=1}^{k^{*}} \\lambda_i}{\\sum_{i=1}^{k} \\lambda_i}
            \\; \\ge \\; \\text{energy}.

    The paper uses ``energy=0.945`` (Section 5.3).

    Args:
        eigvals: 1-D array of eigenvalues in *descending* order.  Negative
            entries (which may occur from numerical noise) are clipped to
            zero before the cumulative sum is formed.
        energy: Cumulative-energy threshold in ``(0, 1]``.

    Returns:
        The 1-based cut-off ``k*`` guaranteed to lie in ``[1, len(eigvals)]``.

    Raises:
        ValueError: If ``eigvals`` is empty or ``energy`` is not in ``(0, 1]``.
    """
    if len(eigvals) == 0:
        raise ValueError("eigvals must be non-empty")
    if not (0.0 < energy <= 1.0):
        raise ValueError(f"energy must lie in (0, 1], got {energy}")

    eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    total = eigvals.sum()
    if total <= 0.0:
        return 1
    cum = np.cumsum(eigvals) / total
    k_star = int(np.searchsorted(cum, energy) + 1)
    return max(1, min(k_star, len(eigvals)))
