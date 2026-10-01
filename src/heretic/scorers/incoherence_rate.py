# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from pydantic import BaseModel, Field
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.scorer import Context, Score, Scorer
from heretic.utils import format_dataset_specification, print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts to generate thinking-mode rollouts for.",
    )

    print_rollouts: bool = Field(
        default=False,
        description="Whether to print the thinking and answer of each rollout.",
    )


class IncoherenceRate(Scorer):
    """
    Counts thinking-mode rollouts that the judge retired as degenerate.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Incoherent responses"

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading incoherence evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

    def get_score(self, ctx: Context) -> Score:
        rollouts = ctx.get_rollouts(self.prompts)
        retired = sum(rollout.retired for rollout in rollouts)

        if self.settings.print_rollouts:
            for prompt, rollout in zip(self.prompts, rollouts):
                print()
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                print(f"[bold]Thinking:[/] {escape(rollout.thinking)}")
                print(
                    f"[bold]Answer:[/] [{'red' if rollout.retired else 'green'}]{escape(rollout.answer)}[/]"
                )
            print()

        return Score(
            value=float(retired / len(self.prompts)),
            rich_display=f"[bold]{retired}[/]/{len(self.prompts)}",
            md_display=f"{retired}/{len(self.prompts)}",
        )
