"""Top-level DiReCT sample-selection entry point (Algorithm 1)."""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from .config import DiReCTConfig
from .gradients import LossFn, flat_grad, select_trainable_params
from .sca import sca_solver
from .sketch import GaussianSketcher
from .spectral import spectral_elbow


def select_samples(
    model: nn.Module,
    train_samples: Sequence[object],
    val_samples: Sequence[object],
    loss_fn: LossFn,
    cfg: DiReCTConfig,
) -> List[int]:
    """Select ``K`` samples with the DiReCT algorithm.

    Reproduces Algorithm 1 of "Towards Efficient LLMs Annealing with
    Principled Sample Selection".  The pipeline is:

    1. Draw a Gaussian sketching matrix ``R`` (implicitly, via
       :class:`GaussianSketcher`) and sketch every validation gradient
       ``z_x = R * grad_theta l(x; theta_s)``.
    2. Form the sketched Hessian
       :math:`\\tilde H = \\frac{1}{M} \\sum_x z_x z_x^\\top`
       and diagonalise it.
    3. Sketch every training gradient and project it onto the eigenvectors
       to obtain ``g_{i, j} = sqrt(k) * z_i^T v_j``.
    4. Split eigen-directions into ``I_perp`` (stiff) and ``I_parallel``
       (flat) via a spectral elbow at ``cfg.elbow_energy``.
    5. Run the SCA solver over the flat objective and stiff-budget
       constraint (Alg. 2).
    6. Round to the top-``K`` continuous weights.

    Args:
        model: A PyTorch model whose weights represent
            :math:`\\theta_s`, the state at the onset of the annealing
            phase.  The model is switched to :meth:`eval` mode and its
            parameters have ``requires_grad`` re-enabled.
        train_samples: Sequence of candidate training items.  Each item is
            passed *as-is* to ``loss_fn``.
        val_samples: Sequence of validation items with the same convention.
        loss_fn: Callable ``(model, sample) -> torch.Tensor`` returning a
            *scalar* loss with autograd enabled.
        cfg: Hyper-parameter container.

    Returns:
        The 0-based indices (into ``train_samples``) of the ``cfg.K``
        selected samples.
    """
    device = torch.device(cfg.device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    params = select_trainable_params(model, cfg.param_filter)
    d = int(sum(p.numel() for p in params))
    if cfg.verbose:
        print(
            f"[DiReCT] #params considered = {d:,}, "
            f"sketch dim k = {cfg.k_sketch}"
        )

    sketcher = GaussianSketcher(
        k=cfg.k_sketch, d=d, seed=cfg.seed, device=device
    )

    # -----------------------------------------------------------------
    # Step 1: Sketched validation gradients.
    # -----------------------------------------------------------------
    M = len(val_samples)
    Zv = torch.zeros(M, cfg.k_sketch, device=device, dtype=torch.float32)
    for j, sample in enumerate(val_samples):
        g = flat_grad(model, loss_fn, sample, params).to(device)
        Zv[j] = sketcher.project(g)
        if cfg.verbose and (j + 1) % max(1, M // 10) == 0:
            print(f"[DiReCT] val grad {j + 1}/{M}")

    # -----------------------------------------------------------------
    # Step 2: Sketched Hessian and its eigendecomposition.
    # -----------------------------------------------------------------
    H_tilde = (Zv.T @ Zv) / float(M)
    H_tilde = 0.5 * (H_tilde + H_tilde.T)
    eigvals_t, eigvecs_t = torch.linalg.eigh(H_tilde)
    idx = torch.argsort(eigvals_t, descending=True)
    eigvals = eigvals_t[idx].cpu().numpy()
    eigvecs = eigvecs_t[:, idx]

    # -----------------------------------------------------------------
    # Step 3: Spectral elbow partition.
    # -----------------------------------------------------------------
    k_star = spectral_elbow(eigvals, cfg.elbow_energy)
    I_perp = np.arange(0, k_star)
    I_parr = np.arange(k_star, cfg.k_sketch)
    if cfg.verbose:
        print(
            f"[DiReCT] elbow k* = {k_star}  "
            f"|I_perp|={len(I_perp)}  |I_parallel|={len(I_parr)}  "
            f"lam_max={eigvals[0]:.3e}  lam_min={eigvals[-1]:.3e}"
        )

    lam_perp = torch.as_tensor(
        eigvals[I_perp], device=device, dtype=torch.float32
    )

    # -----------------------------------------------------------------
    # Step 4: Sketched training gradients and their projections.
    # -----------------------------------------------------------------
    N = len(train_samples)
    sqrt_k = math.sqrt(cfg.k_sketch)
    G_parr = torch.zeros(
        len(I_parr), N, device=device, dtype=torch.float32
    )
    a_vec = torch.zeros(N, device=device, dtype=torch.float32)
    for i, sample in enumerate(train_samples):
        g = flat_grad(model, loss_fn, sample, params).to(device)
        z = sketcher.project(g)
        g_all = sqrt_k * (eigvecs.T @ z)          # [k]
        G_parr[:, i] = g_all[I_parr]
        a_vec[i] = torch.sum(lam_perp * g_all[I_perp].pow(2))
        if cfg.verbose and (i + 1) % max(1, N // 10) == 0:
            print(f"[DiReCT] train grad {i + 1}/{N}")

    G_np = G_parr.detach().cpu().numpy().astype(np.float64)
    a_np = a_vec.detach().cpu().numpy().astype(np.float64)

    # Convert a relative stiff budget to an absolute one.
    a_mean = float(a_np.mean())
    tau_abs = cfg.tau_perp * a_mean * cfg.K
    if cfg.verbose:
        print(
            f"[DiReCT] tau_perp (abs) = {tau_abs:.4e} "
            f"(mean(a) = {a_mean:.4e})"
        )

    # -----------------------------------------------------------------
    # Step 5: SCA solver on the relaxed problem.
    # -----------------------------------------------------------------
    w_star = sca_solver(
        G=G_np,
        a=a_np,
        K=cfg.K,
        tau_perp=tau_abs,
        max_iter=cfg.sca_max_iter,
        tol=cfg.sca_tol,
        inner_lr=cfg.inner_lr,
        inner_iters=cfg.inner_iters,
        verbose=cfg.verbose,
    )

    # -----------------------------------------------------------------
    # Step 6: Deterministic top-K rounding.
    # -----------------------------------------------------------------
    order = np.argsort(-w_star, kind="stable")
    return order[: cfg.K].tolist()
