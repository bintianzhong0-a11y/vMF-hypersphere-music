# -*- coding: utf-8 -*-
"""
vMF Hypersphere Music Representation
====================================

Conformer / Transformer style sequence model for hyperspherical
music representation learning.

The model predicts:
- vMF mean direction mu
- concentration kappa
- root class
- chord-like probability
- chord template
- triad quality
- seventh quality
- beat / bar context
- onset
- velocity
- timing residual
- duration

Output keys are compatible with models/vmf_losses.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1.0e-8


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Input:
        x: [B, T, D]

    Output:
        x + PE: [B, T, D]
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pe[:, :t, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    """
    Conformer-style feed-forward module.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    """
    Lightweight Conformer convolution module.

    Input:
        x: [B, T, D]

    Output:
        [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same padding.")

        self.layer_norm = nn.LayerNorm(d_model)

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=1,
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, T]

        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)

        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # [B, T, D]

        return self.dropout(x)


class ConformerBlock(nn.Module):
    """
    Simplified Conformer block.

    Structure:
        FFN half step
        Multi-head self-attention
        Convolution module
        FFN half step
        Final LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(
            d_model=d_model,
            expansion=ff_expansion,
            dropout=dropout,
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.conv = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = FeedForwardModule(
            d_model=d_model,
            expansion=ff_expansion,
            dropout=dropout,
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, T, D]
            padding_mask:
                [B, T], True means padded position.
                This is PyTorch MultiheadAttention convention.
        """
        x = x + 0.5 * self.ffn1(x)

        y = self.attn_norm(x)
        y, _ = self.self_attn(
            y,
            y,
            y,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(y)

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


class MLPHead(nn.Module):
    """
    Small MLP prediction head.
    """

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------

class VMFConformer(nn.Module):
    """
    vMF Hypersphere Music Representation model.

    Args:
        input_dim:
            Input feature dimension.
            For vmf_coordinates.make_note_direction with register and onset,
            this is usually 10.
        d_model:
            Internal hidden dimension.
        mu_dim:
            Dimension of predicted hyperspherical direction.
            Usually same as input_dim, but can be different.
        num_layers:
            Number of Conformer blocks.
        n_heads:
            Attention heads.
        num_root:
            Root classes. Usually 12.
        num_template:
            Chord template classes.
        num_triad:
            Triad classes.
        num_seventh:
            Seventh classes.
        num_beat:
            Beat classes.
        num_bar:
            Bar context classes.
    """

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 128,
        mu_dim: Optional[int] = None,
        num_layers: int = 4,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        num_root: int = 12,
        num_template: int = 8,
        num_triad: int = 6,
        num_seventh: int = 6,
        num_beat: int = 4,
        num_bar: int = 4,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        if mu_dim is None:
            mu_dim = input_dim

        self.input_dim = input_dim
        self.d_model = d_model
        self.mu_dim = mu_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoding = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=4096,
                dropout=dropout,
            )
            if use_positional_encoding
            else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_expansion=ff_expansion,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(d_model)

        # vMF heads
        self.mu_head = MLPHead(d_model, mu_dim, dropout=dropout)
        self.kappa_head = MLPHead(d_model, 1, dropout=dropout)

        # classification heads
        self.root_head = MLPHead(d_model, num_root, dropout=dropout)
        self.chord_like_head = MLPHead(d_model, 1, dropout=dropout)
        self.template_head = MLPHead(d_model, num_template, dropout=dropout)
        self.triad_head = MLPHead(d_model, num_triad, dropout=dropout)
        self.seventh_head = MLPHead(d_model, num_seventh, dropout=dropout)
        self.beat_head = MLPHead(d_model, num_beat, dropout=dropout)
        self.bar_head = MLPHead(d_model, num_bar, dropout=dropout)
        self.onset_head = MLPHead(d_model, 1, dropout=dropout)

        # regression heads
        self.velocity_head = MLPHead(d_model, 1, dropout=dropout)
        self.timing_head = MLPHead(d_model, 1, dropout=dropout)
        self.duration_head = MLPHead(d_model, 1, dropout=dropout)

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input sequence.

        Args:
            x:
                [B, T, input_dim]
            padding_mask:
                [B, T], True means padded.

        Returns:
            h:
                [B, T, d_model]
        """
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        for block in self.blocks:
            h = block(h, padding_mask=padding_mask)

        return self.output_norm(h)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:
                [B, T, input_dim]
            padding_mask:
                [B, T], True means padded.
            return_hidden:
                Whether to include hidden state h.

        Returns:
            Dict compatible with MultiTaskVMFLoss.
        """
        h = self.encode(x, padding_mask=padding_mask)

        mu_raw = self.mu_head(h)
        mu = l2_normalize(mu_raw, dim=-1)

        kappa_raw = self.kappa_head(h)

        outputs: Dict[str, torch.Tensor] = {
            "mu": mu,
            "kappa": kappa_raw,
            "kappa_positive": F.softplus(kappa_raw),

            "root_logits": self.root_head(h),
            "chord_like_logits": self.chord_like_head(h).squeeze(-1),
            "template_logits": self.template_head(h),
            "triad_logits": self.triad_head(h),
            "seventh_logits": self.seventh_head(h),
            "beat_logits": self.beat_head(h),
            "bar_logits": self.bar_head(h),
            "onset_logits": self.onset_head(h).squeeze(-1),

            "velocity": self.velocity_head(h),
            "timing": self.timing_head(h),
            "duration": self.duration_head(h),
        }

        if return_hidden:
            outputs["hidden"] = h

        return outputs

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience prediction method.
        """
        self.eval()
        outputs = self.forward(x, padding_mask=padding_mask, return_hidden=False)

        pred = dict(outputs)
        pred["root_pred"] = outputs["root_logits"].argmax(dim=-1)
        pred["template_pred"] = outputs["template_logits"].argmax(dim=-1)
        pred["triad_pred"] = outputs["triad_logits"].argmax(dim=-1)
        pred["seventh_pred"] = outputs["seventh_logits"].argmax(dim=-1)
        pred["beat_pred"] = outputs["beat_logits"].argmax(dim=-1)
        pred["bar_pred"] = outputs["bar_logits"].argmax(dim=-1)

        pred["chord_like_prob"] = torch.sigmoid(outputs["chord_like_logits"])
        pred["onset_prob"] = torch.sigmoid(outputs["onset_logits"])

        return pred


def build_vmf_conformer_from_config(config: Dict[str, Any]) -> VMFConformer:
    """
    Build VMFConformer from a config dict.
    Unknown keys are ignored.
    """
    allowed_keys = {
        "input_dim",
        "d_model",
        "mu_dim",
        "num_layers",
        "n_heads",
        "ff_expansion",
        "conv_kernel_size",
        "dropout",
        "num_root",
        "num_template",
        "num_triad",
        "num_seventh",
        "num_beat",
        "num_bar",
        "use_positional_encoding",
    }

    kwargs = {k: v for k, v in config.items() if k in allowed_keys}
    return VMFConformer(**kwargs)
