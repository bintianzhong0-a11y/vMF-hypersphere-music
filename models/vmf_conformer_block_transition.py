# -*- coding: utf-8 -*-
"""
Conformer-mediated block-level function transition model.

This model extends VMFConformerFunction with a block-level transition head.

Prediction target:
    f_b

Condition:
    H_{b-1} from Conformer block pooling
    f_{b-1}
    phrase_pos_b
    root_{b-1}
    template_{b-1}

So it learns:
    P(f_b | H_{b-1}, f_{b-1}, phrase_pos_b, root_{b-1}, template_{b-1})
"""

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn

from models.vmf_conformer_function import VMFConformerFunction


class BlockTransitionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_function: int = 4,
        phrase_period: int = 4,
        num_root: int = 12,
        num_template: int = 16,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_function = num_function
        self.phrase_period = phrase_period
        self.num_root = num_root
        self.num_template = num_template

        self.emb_function = nn.Embedding(num_function, emb_dim)
        self.emb_phrase = nn.Embedding(phrase_period, emb_dim)
        self.emb_root = nn.Embedding(num_root, emb_dim)
        self.emb_template = nn.Embedding(num_template, emb_dim)

        in_dim = d_model + emb_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_function),
        )

    def forward(
        self,
        block_hidden_prev: torch.Tensor,
        prev_function: torch.Tensor,
        phrase_pos: torch.Tensor,
        prev_root: torch.Tensor,
        prev_template: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat(
            [
                block_hidden_prev,
                self.emb_function(prev_function.clamp(0, self.num_function - 1)),
                self.emb_phrase(phrase_pos.clamp(0, self.phrase_period - 1)),
                self.emb_root(prev_root.clamp(0, self.num_root - 1)),
                self.emb_template(prev_template.clamp(0, self.num_template - 1)),
            ],
            dim=-1,
        )

        return self.net(z)


class VMFConformerBlockTransition(VMFConformerFunction):
    def __init__(
        self,
        *args,
        phrase_period: int = 4,
        block_emb_dim: int = 32,
        block_hidden_dim: int = 128,
        num_block_template: int = 16,
        d_model: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            d_model=d_model,
            dropout=dropout,
            **kwargs,
        )

        self.phrase_period = phrase_period

        self.block_transition_head = BlockTransitionHead(
            d_model=d_model,
            num_function=4,
            phrase_period=phrase_period,
            num_root=12,
            num_template=num_block_template,
            emb_dim=block_emb_dim,
            hidden_dim=block_hidden_dim,
            dropout=dropout,
        )

    def pool_blocks(
        self,
        hidden: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        events_per_block: int,
    ) -> torch.Tensor:
        """
        hidden:
            [B, T, H]
        padding_mask:
            [B, T], True means padded.
        returns:
            [B, n_blocks, H]
        """
        B, T, H = hidden.shape
        n_blocks = T // events_per_block
        T_use = n_blocks * events_per_block

        hidden = hidden[:, :T_use, :]
        hidden = hidden.reshape(B, n_blocks, events_per_block, H)

        if padding_mask is None:
            return hidden.mean(dim=2)

        mask = padding_mask[:, :T_use]
        valid = (~mask).float().reshape(B, n_blocks, events_per_block, 1)
        denom = valid.sum(dim=2).clamp_min(1.0)
        pooled = (hidden * valid).sum(dim=2) / denom

        return pooled

    def forward_block_transition(
        self,
        x: torch.Tensor,
        prev_function: torch.Tensor,
        phrase_pos: torch.Tensor,
        prev_root: torch.Tensor,
        prev_template: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        events_per_block: int = 8,
        return_base_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x:
            [B, T, D]
        prev_function:
            [B, n_transitions]
        phrase_pos:
            [B, n_transitions]
        prev_root:
            [B, n_transitions]
        prev_template:
            [B, n_transitions]

        The model predicts target function for next block.
        If x has n_blocks, then n_transitions should be n_blocks - 1.
        """
        base_outputs = super().forward(
            x,
            padding_mask=padding_mask,
            return_hidden=True,
        )

        hidden = base_outputs["hidden"]
        block_hidden = self.pool_blocks(
            hidden=hidden,
            padding_mask=padding_mask,
            events_per_block=events_per_block,
        )

        # Use H_{b-1} to predict f_b.
        block_hidden_prev = block_hidden[:, :-1, :]

        n_trans = min(
            block_hidden_prev.shape[1],
            prev_function.shape[1],
            phrase_pos.shape[1],
            prev_root.shape[1],
            prev_template.shape[1],
        )

        block_hidden_prev = block_hidden_prev[:, :n_trans, :]
        prev_function = prev_function[:, :n_trans]
        phrase_pos = phrase_pos[:, :n_trans]
        prev_root = prev_root[:, :n_trans]
        prev_template = prev_template[:, :n_trans]

        logits = self.block_transition_head(
            block_hidden_prev=block_hidden_prev,
            prev_function=prev_function,
            phrase_pos=phrase_pos,
            prev_root=prev_root,
            prev_template=prev_template,
        )

        out = {
            "block_transition_logits": logits,
            "block_hidden": block_hidden,
        }

        if return_base_outputs:
            out["base_outputs"] = base_outputs

        return out
