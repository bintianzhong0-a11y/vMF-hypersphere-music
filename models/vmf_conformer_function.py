# -*- coding: utf-8 -*-
"""
Function-aware VMFConformer.

Adds:
- function_logits: T / D / SD / OTHER
- function_transition_logits: previous function -> current function pattern

Function labels:
    0: T
    1: D
    2: SD
    3: OTHER

Transition labels:
    prev_function * 4 + current_function
    e.g. T->SD = 0*4 + 2 = 2
         SD->D = 2*4 + 1 = 9
         D->T  = 1*4 + 0 = 4
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from models.vmf_conformer import VMFConformer, MLPHead


class VMFConformerFunction(VMFConformer):
    def __init__(
        self,
        *args,
        num_function: int = 4,
        num_function_transition: int = 16,
        dropout: float = 0.1,
        d_model: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            d_model=d_model,
            dropout=dropout,
            **kwargs,
        )

        self.num_function = num_function
        self.num_function_transition = num_function_transition

        self.function_head = MLPHead(
            d_model=d_model,
            out_dim=num_function,
            dropout=dropout,
        )

        self.function_transition_head = MLPHead(
            d_model=d_model,
            out_dim=num_function_transition,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = super().forward(
            x,
            padding_mask=padding_mask,
            return_hidden=True,
        )

        h = outputs["hidden"]

        outputs["function_logits"] = self.function_head(h)
        outputs["function_transition_logits"] = self.function_transition_head(h)

        if not return_hidden:
            outputs.pop("hidden", None)

        return outputs
