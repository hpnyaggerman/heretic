# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt

from heretic.scorer import Context, Score, Scorer
from heretic.scorers.llm_judge import ANSWER_TOOL_NAME, get_submitted_answer
from heretic.utils import Prompt

SYSTEM_PROMPT = (
    "Read the scene and answer the multiple-choice question about it. Submit the letter "
    f"of the correct option by calling the {ANSWER_TOOL_NAME} tool. Do not explain your answer."
)


class Settings(BaseModel):
    max_response_length: PositiveInt = Field(
        default=256,
        description="Maximum number of tokens to generate for each answer.",
    )

    batch_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Number of questions answered at once, overriding the global batch_size. "
            "The short answers usually allow a larger batch than long generations do."
        ),
    )


def get_answer_tool(options: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": ANSWER_TOOL_NAME,
            "description": "Submit the letter of the correct option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": options,
                        "description": "Letter of the correct option.",
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


class CogToM(Scorer):
    """
    Fraction of the 415 multiple-choice questions of the CogToM theory-of-mind benchmark
    (English) that the model answers wrongly. Each question is about a short scene, and the
    model must answer it without thinking, with a tool call whose only argument is the
    letter of an option and that nothing follows; any other answer counts as wrong. The error rate of each of
    the benchmark's categories (belief, emotion, desire, and so on) is recorded as a component.
    """

    settings: Settings

    @property
    def score_name(self) -> str:
        return "CogToM errors"

    @property
    def reproducible(self) -> bool:
        return True

    def init(self, ctx: Context) -> None:
        path = Path(__file__).with_suffix(".jsonl")
        self.questions = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.prompts = [
            Prompt(
                system=SYSTEM_PROMPT,
                user=(
                    f"{question['scene']}\n\nQuestion: {question['question']}\n\n"
                    + "\n".join(
                        f"{letter}. {text}"
                        for letter, text in question["options"].items()
                    )
                ),
                tools=[get_answer_tool(list(question["options"]))],
                max_response_length=self.settings.max_response_length,
                batch_size=self.settings.batch_size,
                thinking=False,
            )
            for question in self.questions
        ]

    def get_score(self, ctx: Context) -> Score:
        syntax = ctx.get_model().tool_call_syntax
        answers = [
            get_submitted_answer(response, syntax)
            for response in ctx.get_responses(self.prompts)
        ]
        # The response must be the tool call alone, and the answer the letter alone.
        wrong = [
            answer is None or answer.strip() != question["answer"]
            for question, answer in zip(self.questions, answers)
        ]
        total = len(self.questions)
        missing = sum(answer is None for answer in answers)

        categories: dict[str, list[bool]] = {}
        for question, is_wrong in zip(self.questions, wrong):
            categories.setdefault(question["category"].lower(), []).append(is_wrong)

        return Score(
            value=sum(wrong) / total,
            rich_display=f"[bold]{sum(wrong)}[/]/{total} wrong, [bold]{missing}[/]/{total} without tool call",
            md_display=f"{sum(wrong)}/{total} wrong, {missing}/{total} without tool call",
            components={
                "missing_tool_calls": missing / total,
                **{
                    category: sum(values) / len(values)
                    for category, values in categories.items()
                },
            },
        )
