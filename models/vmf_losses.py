# -*- coding: utf-8 -*-
"""
Loss functions for vMF Hypersphere Music Representation.

Supported losses:
- cosine direction loss for mu
- approximate vMF negative log-likelihood
- kappa regularization
- masked cross entropy
- masked BCE with logits
- masked regression losses
- multi-task weighted loss wrapper

The implementation is designed for tensors shaped like:
    [B, T, D] for continuous sequence outputs
    [B, T, C] for classification logits
    [B, T]    for class labels or binary targets
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS: float = 1.0e-8


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """
    Stable L2 normalization.
    """
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


def apply_mask_and_reduce(
    loss: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Apply optional mask and reduce.

    Args:
        loss:
            Elementwise loss.
        mask:
            Broadcastable mask. True/1 means valid.
        reduction:
            "mean", "sum", or "none".
    """
    if mask is not None:
        mask = mask.to(device=loss.device, dtype=loss.dtype)
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        loss = loss * mask

    if reduction == "none":
        return loss

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        if mask is None:
            return loss.mean()
        denom = mask.expand_as(loss).sum().clamp_min(eps)
        return loss.sum() / denom

    raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------
# vMF / direction losses
# ---------------------------------------------------------------------

def cosine_direction_loss(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Direction loss:

        L_mu = 1 - pred_mu^T target_mu

    Both pred_mu and target_mu are normalized internally.

    Args:
        pred_mu:
            Predicted direction, shape [..., D].
        target_mu:
            Target direction, same shape.
        mask:
            Optional mask for sequence positions, shape [...] without D.
    """
    pred = l2_normalize(pred_mu, dim=-1, eps=eps)
    target = l2_normalize(target_mu, dim=-1, eps=eps)

    cos = (pred * target).sum(dim=-1)
    loss = 1.0 - cos
    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


def angular_error(
    pred_mu: torch.Tensor,
    target_mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Angular error in radians.

        arccos(pred_mu^T target_mu)
    """
    pred = l2_normalize(pred_mu, dim=-1, eps=eps)
    target = l2_normalize(target_mu, dim=-1, eps=eps)
    cos = (pred * target).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    err = torch.acos(cos)
    return apply_mask_and_reduce(err, mask=mask, reduction=reduction, eps=eps)


def vmf_nll_approx(
    pred_mu: torch.Tensor,
    target_direction: torch.Tensor,
    pred_kappa: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kappa_min: float = 1.0e-4,
    kappa_max: float = 100.0,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Practical approximate vMF negative log-likelihood.

    Exact vMF NLL contains the normalizing constant C_D(kappa),
    which can be numerically troublesome for early prototypes.

    Here we use a stable surrogate:

        L = - kappa * cos(pred_mu, target) + log(1 + kappa)

    This encourages:
        - high directional agreement
        - nonzero but not exploding kappa

    Args:
        pred_mu:
            Predicted mean direction, shape [..., D].
        target_direction:
            Target direction, same shape.
        pred_kappa:
            Predicted concentration, shape [...] or [..., 1].
    """
    pred = l2_normalize(pred_mu, dim=-1, eps=eps)
    target = l2_normalize(target_direction, dim=-1, eps=eps)

    cos = (pred * target).sum(dim=-1)

    kappa = F.softplus(pred_kappa)
    if kappa.dim() == cos.dim() + 1:
        kappa = kappa.squeeze(-1)

    kappa = kappa.clamp(min=kappa_min, max=kappa_max)

    loss = -kappa * cos + torch.log1p(kappa)
    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


def kappa_regularization(
    pred_kappa: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    mode: str = "l2",
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Regularize concentration kappa.

    mode:
        "l2"     : penalize kappa^2
        "l1"     : penalize |kappa|
        "target" : penalize (kappa - target)^2

    pred_kappa is passed through softplus.
    """
    kappa = F.softplus(pred_kappa)

    if mode == "l2":
        loss = kappa.pow(2)
    elif mode == "l1":
        loss = kappa.abs()
    elif mode == "target":
        if target is None:
            raise ValueError("target must be provided when mode='target'")
        target = target.to(device=kappa.device, dtype=kappa.dtype)
        loss = (kappa - target).pow(2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


# ---------------------------------------------------------------------
# Classification / regression losses
# ---------------------------------------------------------------------

def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross entropy for sequence logits.

    Args:
        logits:
            Shape [B, T, C] or [N, C].
        target:
            Shape [B, T] or [N].
        mask:
            Shape [B, T] or [N].
    """
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)
    target_flat = target.reshape(-1).long()

    loss_flat = F.cross_entropy(
        logits_flat,
        target_flat,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    valid = target_flat.ne(ignore_index)

    if mask is not None:
        mask_flat = mask.reshape(-1).to(device=logits.device).bool()
        valid = valid & mask_flat

    if reduction == "none":
        return loss_flat.reshape(target.shape)

    loss_flat = loss_flat * valid.to(loss_flat.dtype)

    if reduction == "sum":
        return loss_flat.sum()

    if reduction == "mean":
        denom = valid.sum().clamp_min(1)
        return loss_flat.sum() / denom

    raise ValueError(f"Unknown reduction: {reduction}")


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    BCE with logits for binary or multi-label sequence outputs.

    Args:
        logits:
            Shape [B, T], [B, T, 1], or [B, T, C].
        target:
            Same shape or broadcastable.
        mask:
            Shape [B, T].
    """
    target = target.to(device=logits.device, dtype=logits.dtype)

    if target.shape != logits.shape:
        # common case: logits [B,T,1], target [B,T]
        if logits.dim() == target.dim() + 1 and logits.shape[-1] == 1:
            target = target.unsqueeze(-1)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "smooth_l1",
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Masked regression loss.

    loss_type:
        "mae"
        "mse"
        "smooth_l1"
    """
    target = target.to(device=pred.device, dtype=pred.dtype)

    if target.shape != pred.shape:
        if pred.dim() == target.dim() + 1 and pred.shape[-1] == 1:
            target = target.unsqueeze(-1)

    if loss_type == "mae":
        loss = (pred - target).abs()
    elif loss_type == "mse":
        loss = (pred - target).pow(2)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


def entropy_regularization(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sign: str = "minimize_entropy",
    reduction: str = "mean",
    eps: float = EPS,
) -> torch.Tensor:
    """
    Entropy regularization for prototype probabilities gamma.

    Args:
        probs:
            Probability tensor [..., K].
        sign:
            "minimize_entropy": returns +H(p)
            "maximize_entropy": returns -H(p)
    """
    p = probs.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)

    if sign == "minimize_entropy":
        loss = entropy
    elif sign == "maximize_entropy":
        loss = -entropy
    else:
        raise ValueError(f"Unknown sign: {sign}")

    return apply_mask_and_reduce(loss, mask=mask, reduction=reduction, eps=eps)


# ---------------------------------------------------------------------
# Multi-task wrapper
# ---------------------------------------------------------------------

class MultiTaskVMFLoss(nn.Module):
    """
    Multi-task loss for Conformer-vMF style models.

    Expected output keys:
        mu
        kappa
        root_logits
        chord_like_logits
        template_logits
        triad_logits
        seventh_logits
        beat_logits
        bar_logits
        onset_logits
        velocity
        timing
        duration
        gamma

    Expected target keys:
        mu
        kappa
        root
        chord_like
        template
        triad
        seventh
        beat
        bar
        onset
        velocity
        timing
        duration
        gamma_label

    Only keys that exist in both outputs and targets are used.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -100,
        regression_loss: str = "smooth_l1",
        use_vmf_nll: bool = False,
        kappa_reg_mode: str = "l2",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        default_weights = {
            "mu": 1.0,
            "vmf_nll": 0.0,
            "kappa": 0.01,
            "root": 1.0,
            "chord_like": 0.5,
            "template": 1.0,
            "triad": 0.5,
            "seventh": 0.5,
            "beat": 0.3,
            "bar": 0.3,
            "onset": 0.5,
            "velocity": 0.3,
            "timing": 0.2,
            "duration": 0.2,
            "gamma_entropy": 0.0,
            "gamma_ce": 0.0,
        }

        if weights is not None:
            default_weights.update(weights)

        self.weights = default_weights
        self.ignore_index = ignore_index
        self.regression_loss = regression_loss
        self.use_vmf_nll = use_vmf_nll
        self.kappa_reg_mode = kappa_reg_mode
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_loss, loss_dict
        """
        device = next(iter(outputs.values())).device
        total = torch.zeros((), device=device)
        losses: Dict[str, torch.Tensor] = {}

        def add_loss(name: str, value: torch.Tensor) -> None:
            nonlocal total
            w = float(self.weights.get(name, 0.0))
            if w != 0.0:
                total = total + w * value
            losses[name] = value.detach()

        # Direction loss
        if "mu" in outputs and "mu" in targets and self.weights.get("mu", 0.0) != 0.0:
            loss_mu = cosine_direction_loss(outputs["mu"], targets["mu"], mask=mask)
            add_loss("mu", loss_mu)

        # vMF approximate NLL
        if (
            self.use_vmf_nll
            and "mu" in outputs
            and "mu" in targets
            and "kappa" in outputs
            and self.weights.get("vmf_nll", 0.0) != 0.0
        ):
            loss_vmf = vmf_nll_approx(
                pred_mu=outputs["mu"],
                target_direction=targets["mu"],
                pred_kappa=outputs["kappa"],
                mask=mask,
            )
            add_loss("vmf_nll", loss_vmf)

        # Kappa regularization or supervised kappa
        if "kappa" in outputs and self.weights.get("kappa", 0.0) != 0.0:
            if "kappa" in targets:
                loss_kappa = kappa_regularization(
                    outputs["kappa"],
                    mask=mask,
                    target=targets["kappa"],
                    mode="target",
                )
            else:
                loss_kappa = kappa_regularization(
                    outputs["kappa"],
                    mask=mask,
                    mode=self.kappa_reg_mode,
                )
            add_loss("kappa", loss_kappa)

        # Multi-class heads
        ce_specs = [
            ("root", "root_logits"),
            ("template", "template_logits"),
            ("triad", "triad_logits"),
            ("seventh", "seventh_logits"),
            ("beat", "beat_logits"),
            ("bar", "bar_logits"),
        ]

        for name, logit_key in ce_specs:
            if (
                logit_key in outputs
                and name in targets
                and self.weights.get(name, 0.0) != 0.0
            ):
                loss = masked_cross_entropy(
                    outputs[logit_key],
                    targets[name],
                    mask=mask,
                    ignore_index=self.ignore_index,
                    label_smoothing=self.label_smoothing,
                )
                add_loss(name, loss)

        # Binary / multi-label heads
        bce_specs = [
            ("chord_like", "chord_like_logits"),
            ("onset", "onset_logits"),
        ]

        for name, logit_key in bce_specs:
            if (
                logit_key in outputs
                and name in targets
                and self.weights.get(name, 0.0) != 0.0
            ):
                loss = masked_bce_with_logits(
                    outputs[logit_key],
                    targets[name],
                    mask=mask,
                )
                add_loss(name, loss)

        # Regression heads
        reg_specs = [
            ("velocity", "velocity"),
            ("timing", "timing"),
            ("duration", "duration"),
        ]

        for name, out_key in reg_specs:
            if (
                out_key in outputs
                and name in targets
                and self.weights.get(name, 0.0) != 0.0
            ):
                loss = masked_regression_loss(
                    outputs[out_key],
                    targets[name],
                    mask=mask,
                    loss_type=self.regression_loss,
                )
                add_loss(name, loss)

        # Prototype entropy regularization
        if "gamma" in outputs and self.weights.get("gamma_entropy", 0.0) != 0.0:
            loss_gamma_entropy = entropy_regularization(
                outputs["gamma"],
                mask=mask,
                sign="minimize_entropy",
            )
            add_loss("gamma_entropy", loss_gamma_entropy)

        # Optional prototype CE
        if (
            "gamma_logits" in outputs
            and "gamma_label" in targets
            and self.weights.get("gamma_ce", 0.0) != 0.0
        ):
            loss_gamma_ce = masked_cross_entropy(
                outputs["gamma_logits"],
                targets["gamma_label"],
                mask=mask,
                ignore_index=self.ignore_index,
            )
            add_loss("gamma_ce", loss_gamma_ce)

        losses["total"] = total.detach()
        return total, losses
