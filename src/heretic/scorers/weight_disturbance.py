# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from typing import cast

import torch
from peft.tuners.lora.layer import Linear
from torch import Tensor
from torch.nn import Module

from heretic.model import Model, get_base_weight
from heretic.scorer import Context, Score, Scorer


def get_modules(model: Model) -> list[Module]:
    """
    The abliterable modules in a fixed order (layer, component, module), which identifies
    each module across the LoRA wrapping and reloading that replace the module objects.
    """
    return [
        module
        for layer_index in range(len(model.get_layers()))
        for modules in model.get_layer_modules(layer_index).values()
        for module in modules
    ]


def get_squared_norm(module: Module) -> float:
    """Squared Frobenius norm of the module's original weight."""
    return torch.linalg.vector_norm(get_base_weight(module)).item() ** 2


def get_squared_delta_norm(module: Module, row_scale: Tensor | None) -> float:
    """
    Squared Frobenius norm of the change the module's adapter makes to its weight W,
    which is D = s * (W + c B A) - W with row scales s, or c B A without them, or 0 if
    the module has no adapter. The norm is computed from the low-rank factors without
    forming the full matrix: with M = c B A, ||M||^2 is the trace of (B^T B)(A A^T) times
    c^2, and each row of D contributes (s - 1)^2 ||w||^2 + 2 s (s - 1) w . m + s^2 ||m||^2.
    """
    if "default" not in getattr(module, "lora_A", {}):
        return 0.0
    # This cast is valid because only LoRA layers have adapters.
    layer = cast(Linear, module)
    A = cast(Tensor, layer.lora_A["default"].weight).float()
    B = cast(Tensor, layer.lora_B["default"].weight).float()
    scaling = layer.scaling["default"]
    if row_scale is None:
        return scaling**2 * torch.sum((B.T @ B) * (A @ A.T)).item()

    W = get_base_weight(module)
    s = row_scale.float().to(W.device)
    A, B = A.to(W.device), B.to(W.device)
    w_squared = torch.sum(W * W, dim=1)
    w_dot_m = scaling * torch.sum(B * (W @ A.T), dim=1)
    m_squared = scaling**2 * torch.sum((B @ (A @ A.T)) * B, dim=1)
    rows = (s - 1) ** 2 * w_squared + 2 * s * (s - 1) * w_dot_m + s**2 * m_squared
    return torch.sum(rows).item()


class WeightDisturbance(Scorer):
    """
    How much the trial changed the abliterable weights: the Frobenius norm of all changes
    the modifier's LoRA adapters make, relative to the norm of the original weights of all
    abliterable modules, in percent. It depends on the trial's parameters alone, and
    measures change in every direction, whatever its effect on any task.
    """

    @property
    def score_name(self) -> str:
        return "Weight disturbance"

    @property
    def reproducible(self) -> bool:
        return True

    def init(self, ctx: Context) -> None:
        self.squared_norms = [
            get_squared_norm(module) for module in get_modules(ctx.get_model())
        ]

    def get_score(self, ctx: Context) -> Score:
        model = ctx.get_model()
        squared_deltas = [
            get_squared_delta_norm(module, model.get_row_scale(module))
            for module in get_modules(model)
        ]
        total = 100 * math.sqrt(sum(squared_deltas) / sum(self.squared_norms))
        largest = 100 * max(
            math.sqrt(delta / norm)
            for delta, norm in zip(squared_deltas, self.squared_norms, strict=True)
        )

        return Score(
            value=total,
            rich_display=f"[bold]{total:.4f}%[/] (largest module change {largest:.4f}%)",
            md_display=f"{total:.4f}% (largest module change {largest:.4f}%)",
            components={"largest_module_change": largest},
        )
