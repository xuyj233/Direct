"""Memory-efficient Gaussian random projection for gradient sketching."""

from __future__ import annotations

import math

import torch


class GaussianSketcher:
    """Applies a ``k x d`` Gaussian random projection without materialising ``R``.

    The sketching matrix ``R`` has i.i.d. entries drawn from
    :math:`\\mathcal{N}(0, 1/k)`, so that :math:`\\mathbb{E}\\|R g\\|_2^2 = \\|g\\|_2^2`.

    Because ``d`` (the number of model parameters) can be in the billions,
    we never build ``R`` explicitly.  Instead the columns of ``R`` are
    regenerated on the fly, block by block, from a deterministic seed.

    Example:
        >>> sketcher = GaussianSketcher(k=1024, d=1_000_000, seed=0)
        >>> z = sketcher.project(torch.randn(1_000_000))
        >>> z.shape
        torch.Size([1024])

    Args:
        k: Output (sketched) dimensionality.
        d: Input dimensionality (typically ``sum(p.numel() for p in params)``).
        seed: Master seed. Two sketchers with the same seed and ``d`` apply
            *exactly* the same projection.
        device: Torch device used to draw the projection blocks and store
            the result.
        chunk_size: Number of input coordinates processed at a time.  A
            larger value is faster but uses more peak memory.
    """

    def __init__(
        self,
        k: int,
        d: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
        chunk_size: int = 1 << 20,
    ) -> None:
        self.k = int(k)
        self.d = int(d)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.scale = 1.0 / math.sqrt(self.k)

    def project(self, g: torch.Tensor) -> torch.Tensor:
        """Return :math:`R g \\in \\mathbb{R}^{k}`.

        Args:
            g: 1-D tensor of length ``d`` (or any tensor whose ``.view(-1)``
                has that length).

        Returns:
            A 1-D ``float32`` tensor of length ``k`` on ``self.device``.
        """
        assert g.numel() == self.d, (
            f"GaussianSketcher expected input of size {self.d}, "
            f"got {g.numel()}"
        )
        g = g.to(self.device, dtype=torch.float32).reshape(-1)
        out = torch.zeros(self.k, device=self.device, dtype=torch.float32)
        gen = torch.Generator(device=self.device)
        offset = 0
        chunk_idx = 0
        while offset < self.d:
            cs = min(self.chunk_size, self.d - offset)
            gen.manual_seed(self.seed * 1_000_003 + chunk_idx)
            block = torch.randn(
                self.k,
                cs,
                generator=gen,
                device=self.device,
                dtype=torch.float32,
            ) * self.scale
            out += block @ g[offset : offset + cs]
            offset += cs
            chunk_idx += 1
        return out
