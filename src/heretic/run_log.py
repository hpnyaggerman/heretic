# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
from datetime import datetime, timezone
from typing import Any, cast

import torch
from torch import Tensor

from .config import Settings
from .evaluator import Evaluator
from .model import Model
from .modifier import ModifierEntry
from .utils import get_environment_dict, get_system_dict


def get_module_inventory(model: Model) -> list[dict[str, list[dict[str, Any]]]]:
    """
    For each layer, the abliterable modules of each component,
    with their names and the shapes and dtypes of their weights.
    """

    names = {id(module): name for name, module in model.model.named_modules()}
    inventory = []

    for layer_index in range(len(model.get_layers())):
        layer = {}
        for component, modules in model.get_layer_modules(layer_index).items():
            layer[component] = []
            for module in modules:
                weight = cast(Tensor, module.weight)
                # 4-bit weights are stored packed; the quantization state has the real shape.
                quant_state = getattr(weight, "quant_state", None)
                layer[component].append(
                    {
                        "name": names.get(id(module)),
                        "shape": list(
                            weight.shape if quant_state is None else quant_state.shape
                        ),
                        "dtype": str(weight.dtype),
                    }
                )
        inventory.append(layer)

    return inventory


def write_run_information(
    directory: str,
    settings: Settings,
    model: Model,
    modifier_entry: ModifierEntry,
    evaluator: Evaluator,
):
    os.makedirs(directory, exist_ok=True)

    modifier = modifier_entry.modifier

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_dict(),
        "system": get_system_dict(),
        "settings": settings.model_dump(mode="json"),
        "model_config": model.model.config.to_dict(),
        "modules": get_module_inventory(model),
        "modifier": {
            "name": modifier_entry.name,
            "settings": None
            if modifier.settings is None
            else modifier.settings.model_dump(mode="json"),
        },
        "objectives": [
            {"name": name, "direction": direction.name}
            for name, direction in zip(
                evaluator.get_objective_names(),
                evaluator.get_objective_directions(),
            )
        ],
        "baseline_scores": [
            {"name": name, "score": score.summary()}
            for name, score in evaluator.baseline_scores
        ],
    }

    with open(os.path.join(directory, "run.json"), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)

    priors = modifier.get_priors()
    if priors:
        torch.save(
            {name: tensor.cpu() for name, tensor in priors.items()},
            os.path.join(directory, "priors.pt"),
        )


def append_trial_record(directory: str, record: dict[str, Any]):
    with open(os.path.join(directory, "trials.jsonl"), "a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def append_outputs(
    directory: str, trial: int | str, scorer: str, records: list[dict[str, Any]]
):
    """
    Appends what a scorer graded in a trial, or for the baseline, one line per request:
    the texts, their verdicts or grades, and the degeneration judge's reviews.
    """
    with open(os.path.join(directory, "outputs.jsonl"), "a", encoding="utf-8") as file:
        for record in records:
            line = {"trial": trial, "scorer": scorer, **record}
            file.write(json.dumps(line, default=str) + "\n")
