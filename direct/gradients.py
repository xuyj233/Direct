"""Per-sample gradient utilities used by the selector."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn


LossFn = Callable[[nn.Module, object], torch.Tensor]


def select_trainable_params(
    model: nn.Module,
    param_filter: Callable[[str, nn.Parameter], bool] | None = None,
) -> list[nn.Parameter]:
    """Return the parameters that will contribute to per-sample gradients.

    Args:
        model: The PyTorch model.
        param_filter: Optional callable ``(name, param) -> bool`` used to
            further restrict the parameter list. ``None`` selects every
            trainable parameter.

    Returns:
        A list of :class:`torch.nn.Parameter` in a deterministic order
        (the order returned by :meth:`nn.Module.named_parameters`).
    """
    params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if param_filter is not None and not param_filter(name, p):
            continue
        params.append(p)
    return params


def flat_grad(
    model: nn.Module,
    loss_fn: LossFn,
    sample: object,
    params: Sequence[nn.Parameter],
) -> torch.Tensor:
    """Compute a *flat* gradient vector of ``loss_fn(model, sample)``.

    Args:
        model: PyTorch model providing the parameters to differentiate.
        loss_fn: Callable returning a scalar loss tensor with autograd
            enabled.
        sample: Any object accepted by ``loss_fn``.
        params: The subset of parameters wrt. which we compute the
            gradient.  Parameters not touched by ``loss_fn`` receive a
            zero contribution.

    Returns:
        A 1-D tensor of length ``sum(p.numel() for p in params)`` on the
        parameters' device.
    """
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, sample)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    flats = []
    for p, g in zip(params, grads):
        if g is None:
            flats.append(torch.zeros_like(p).reshape(-1))
        else:
            flats.append(g.detach().reshape(-1))
    return torch.cat(flats, dim=0)
