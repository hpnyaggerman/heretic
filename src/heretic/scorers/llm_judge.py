# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from openrouter import OpenRouter, components
from openrouter.types import UNSET
from pydantic import BaseModel, Field, field_validator, model_validator
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.judge import get_provider_preferences, raise_if_permanent
from heretic.scorer import Context, Score, Scorer
from heretic.utils import (
    Prompt,
    ends_with_call,
    find_xml_calls,
    format_dataset_specification,
    print,
)

TOOL_NAME = "submit_category"
ARGUMENT_NAME = "category"

# The judged model must submit its answer with this tool, so that the score also
# reflects whether abliteration has damaged its instruction following.
ANSWER_TOOL_NAME = "submit_answer"
ANSWER_INSTRUCTION = (
    f"Submit your complete answer by calling the {ANSWER_TOOL_NAME} tool."
)
ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": ANSWER_TOOL_NAME,
        "description": "Submit your answer to the user's request.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your complete answer.",
                },
            },
            "required": ["answer"],
        },
    },
}
# The answer argument up to its closing quote or, if the call was cut off, the end of the text.
ANSWER_PATTERN = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def get_submitted_answer(text: str, syntax: str = "json") -> str | None:
    """
    Returns the answer argument of the last submit_answer call in the text, or None if
    there is none, anything but the template's call syntax follows it, or it has
    arguments besides the answer. Reasoning may come before the call. In the "xml" syntax,
    the call is a <function=...> block; in the "json" syntax, it is located loosely, so that
    it is found in any chat template's JSON call syntax. Either way, a call cut off by the
    response length limit counts.
    """
    if syntax == "xml":
        calls = [call for call in find_xml_calls(text) if call[2] == ANSWER_TOOL_NAME]
        if not calls:
            return None
        _, end, _, arguments = calls[-1]
        if set(arguments) != {"answer"} or not ends_with_call(text, end):
            return None
        return arguments["answer"]

    # The first mention, as the answer itself may mention the tool; a mention in reasoning
    # before the call is harmless, since the call's arguments are found after it.
    start = text.find(ANSWER_TOOL_NAME)
    match = None if start < 0 else ANSWER_PATTERN.search(text, start)
    if match is None:
        return None
    brace = text.rfind("{", start, match.start())
    if brace < 0:
        return None
    # The call ends with its arguments object, or with the text if it was cut off.
    try:
        arguments, end = json.JSONDecoder(strict=False).raw_decode(text, brace)
    except ValueError:
        arguments, end = {"answer": None}, len(text)
    if set(arguments) != {"answer"} or not ends_with_call(text, end):
        return None
    try:
        return json.loads(f'"{match.group(1)}"', strict=False)
    except ValueError:
        return match.group(1)


def normalize_identifier(identifier: str) -> str:
    """
    Judge models follow different naming conventions, so identifiers they echo back
    are compared ignoring case, whitespace, "-", and "_". For example, "Complies-With-Caveats",
    "complies with caveats", and "compliesWithCaveats" all match "complies_with_caveats".
    """
    return re.sub(r"[\s_-]", "", identifier).casefold()


class Category(BaseModel):
    label: str = Field(
        description=(
            "Name of the category, which the judge submits to select it. "
            'The submitted name is matched ignoring case, whitespace, "-", and "_".'
        ),
    )

    description: str = Field(
        description="What a judged text must show to belong to the category.",
    )

    details: str | None = Field(
        default=None,
        description=(
            "Criteria, examples, or edge cases that clarify when the category applies. "
            "The judge receives them along with the description."
        ),
    )

    value: float = Field(
        ge=0,
        le=1,
        description=(
            "Value of the category. The score is the mean value over the requests, where the "
            "value of an answer outside the tool is moved halfway to the highest value and a "
            "retired rollout counts as the highest."
        ),
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if not normalize_identifier(label):
            raise ValueError(
                'must contain characters other than whitespace, "-", and "_"'
            )
        return label

    @field_validator("description", "details")
    @classmethod
    def validate_text(cls, text: str | None) -> str | None:
        if text is not None and not text.strip():
            raise ValueError("cannot be empty or whitespace")
        return text


class Settings(BaseModel):
    score_name: str = Field(
        default="LLM judge",
        description="Name that describes what the configured judge measures.",
    )

    model: str = Field(
        description="OpenRouter model ID of the judge. The model must support tool calling.",
    )

    provider: list[str] | None = Field(
        default=None,
        description=(
            "OpenRouter providers that serve the judge's requests, in order of preference and "
            'with no fallback to others (e.g. ["deepinfra"]). If not set, OpenRouter chooses '
            "among the endpoints known to be at fp8 or higher precision."
        ),
    )

    thinking: bool = Field(
        default=True,
        description=(
            "Whether the judged model thinks before answering in thinking mode. If false, it "
            "answers without thinking, and only its response is judged, as outside thinking mode."
        ),
    )

    response_prompt: str = Field(
        description=(
            "Instructions for judging the response (e.g. how it handles the request). "
            "The judge sees the user request, the thinking (if the judged model thinks), and the "
            "response, and submits one of the response categories."
        ),
    )

    response_categories: list[Category] = Field(
        description="Categories the judge chooses from when judging the response.",
    )

    thinking_prompt: str | None = Field(
        default=None,
        description=(
            "Instructions for judging the thinking (e.g. whether it decides to refuse). "
            "The judge sees the user request, the thinking, and the response, "
            "and submits one of the thinking categories. Required in thinking mode, "
            "unless thinking is false."
        ),
    )

    thinking_categories: list[Category] | None = Field(
        default=None,
        description=(
            "Categories the judge chooses from when judging the thinking. "
            "Required in thinking mode, unless thinking is false."
        ),
    )

    response_prompt_file: str | None = Field(
        default=None,
        description=(
            "File to read response_prompt from, instead of setting it directly. "
            "A relative path is relative to the working directory."
        ),
    )

    thinking_prompt_file: str | None = Field(
        default=None,
        description=(
            "File to read thinking_prompt from, instead of setting it directly. "
            "A relative path is relative to the working directory."
        ),
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description=(
            "Dataset of prompts whose responses are judged. The default (harmful requests) "
            "suits measuring refusals; for another target, use prompts that elicit it."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def read_prompt_files(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for name in ("response_prompt", "thinking_prompt"):
            path = data.get(f"{name}_file")
            if path is None:
                continue
            if data.get(name) is not None:
                raise ValueError(f"{name} and {name}_file cannot both be set")
            data = {**data, name: Path(path).read_text(encoding="utf-8")}

        return data

    @field_validator("response_categories", "thinking_categories")
    @classmethod
    def validate_categories(
        cls, categories: list[Category] | None
    ) -> list[Category] | None:
        if categories is None:
            return categories

        if len(categories) < 2:
            raise ValueError("at least two categories are required")

        # Unique after normalization, so that a submitted label matches at most one category.
        labels = [normalize_identifier(category.label) for category in categories]
        if len(set(labels)) < len(labels):
            raise ValueError(
                'category labels must be unique when compared ignoring case, whitespace, "-", and "_"'
            )

        return categories


def get_category_tool(
    categories: list[Category],
) -> components.ChatFunctionToolFunction:
    lines = []
    for category in categories:
        lines.append(f"- {category.label}: {category.description}")
        if category.details is not None:
            lines.append(textwrap.indent(category.details, "  "))
    definitions = "\n".join(lines)

    return components.ChatFunctionToolFunction(
        type="function",
        function=components.ChatFunctionToolFunctionFunction(
            name=TOOL_NAME,
            description="Submit the category that best fits.",
            parameters={
                "type": "object",
                "properties": {
                    ARGUMENT_NAME: {
                        "type": "string",
                        "enum": [category.label for category in categories],
                        "description": f"Category that best fits, by these definitions:\n{definitions}",
                    },
                },
                "required": [ARGUMENT_NAME],
                "additionalProperties": False,
            },
        ),
    )


class LLMJudge(Scorer):
    """
    Mean value of the categories that an LLM judge, called through the OpenRouter API,
    assigns to the judged texts.

    The judge selects a category through a tool whose argument is restricted to the
    configured labels and documents their descriptions. Each category maps to its
    configured value. The configured prompts and categories alone define what the judge
    measures, so any behavior can be targeted. Only the default dataset is geared
    toward refusals.

    In thinking mode, the thinking and the response of each rollout are judged in
    separate requests, each with its own prompt and categories, and the score is
    the mean of the two mean values, unless the thinking setting is false: then
    the model answers without thinking, and only the response is judged.

    The judged model is instructed to submit its answer with a submit_answer tool call.
    An answer submitted any other way, or with anything written after the call, is judged
    as it was written, and its response value is then moved halfway to the highest category
    value, which counts as the worst, as for a score that is minimized. A rollout retired by
    the degeneration judge gets the highest category values, without being judged.
    """

    settings: Settings

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    @property
    def value_range(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def init(self, ctx: Context) -> None:
        if (
            self.heretic_settings.thinking
            and self.settings.thinking
            and (
                self.settings.thinking_prompt is None
                or self.settings.thinking_categories is None
            )
        ):
            raise ValueError(
                "thinking_prompt and thinking_categories must be set in thinking mode, "
                "unless thinking is false."
            )

        # The client reads the key from the environment; it is never a setting.
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("The OPENROUTER_API_KEY environment variable is not set.")

        # Failed requests are retried by _judge(), which warns about each failure.
        self.client = OpenRouter(
            client=httpx.Client(timeout=None, follow_redirects=True),
            retry_config=None,
        )

        author, _, slug = self.settings.model.partition("/")
        model = self.client.models.get(author=author, slug=slug)
        supported = model.data.supported_parameters
        if "tools" not in supported:
            raise ValueError(
                f"The judge model {self.settings.model} does not support tool calling."
            )
        self.temperature = 0.0 if "temperature" in supported else UNSET
        self.provider = get_provider_preferences(
            self.client, self.settings.model, self.settings.provider
        )

        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = [
            Prompt(
                system=f"{prompt.system} {ANSWER_INSTRUCTION}".strip(),
                user=prompt.user,
                tools=[ANSWER_TOOL],
                thinking=self.settings.thinking,
            )
            for prompt in ctx.load_prompts(self.settings.prompts)
        ]
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        tokenizer = ctx.get_model().tokenizer
        chat = [{"role": "user", "content": self.prompts[0].user}]
        if tokenizer.apply_chat_template(
            chat, tools=[ANSWER_TOOL], tokenize=False
        ) == tokenizer.apply_chat_template(chat, tokenize=False):
            raise ValueError(
                f"The chat template of {self.heretic_settings.model} ignores tools, "
                f"so the model cannot submit its answers with the {ANSWER_TOOL_NAME} tool."
            )

    def get_score(self, ctx: Context) -> Score:
        total = len(self.prompts)
        syntax = ctx.get_model().tool_call_syntax
        judges_thinking = self.heretic_settings.thinking and self.settings.thinking
        thinking_prompt = self.settings.thinking_prompt
        thinking_categories = self.settings.thinking_categories
        if judges_thinking:
            assert thinking_prompt is not None and thinking_categories is not None

        outputs = ctx.get_outputs(self.prompts)
        answers = [get_submitted_answer(output.answer, syntax) for output in outputs]
        # A retired rollout gets the worst values without being judged.
        judged = [index for index, output in enumerate(outputs) if not output.retired]
        missing = sum(answers[index] is None for index in judged)
        transcripts = [
            f"[User request]\n{self.prompts[index].user}\n\n"
            + (f"[Thinking]\n{outputs[index].thinking}\n\n" if judges_thinking else "")
            # An answer not submitted with the tool is judged as it was written.
            + f"[Response]\n{outputs[index].answer if answers[index] is None else answers[index]}"
            for index in judged
        ]
        instructions = [self.settings.response_prompt] * len(judged)
        categories = [self.settings.response_categories] * len(judged)
        if judges_thinking:
            assert thinking_prompt is not None and thinking_categories is not None
            instructions += [thinking_prompt] * len(judged)
            categories += [thinking_categories] * len(judged)
        judgments = self._judge(
            instructions,
            categories,
            transcripts * (2 if judges_thinking else 1),
        )
        response_judgments = dict(zip(judged, judgments[: len(judged)]))
        thinking_judgments = dict(zip(judged, judgments[len(judged) :]))

        worst_response = max(
            category.value for category in self.settings.response_categories
        )
        worst_thinking = max(
            (category.value for category in thinking_categories or []), default=1.0
        )
        response_values = []
        thinking_values = []
        records = []
        for index, (prompt, output, answer) in enumerate(
            zip(self.prompts, outputs, answers)
        ):
            response_judgment = response_judgments.get(index)
            thinking_judgment = thinking_judgments.get(index)
            if response_judgment is None:
                response_value = worst_response
            else:
                response_value = response_judgment["value"]
                if answer is None:
                    # Judged as written, but writing outside the tool still costs half
                    # of the way to the worst value.
                    response_value += (worst_response - response_value) / 2
            response_values.append(response_value)
            thinking_value = (
                worst_thinking
                if thinking_judgment is None
                else thinking_judgment["value"]
            )
            thinking_values.append(thinking_value)
            records.append(
                {
                    "request": prompt.user,
                    "thinking": output.thinking,
                    "answer": output.written_answer,
                    "submitted": answer is not None,
                    "retired": output.retired,
                    "checks": output.checks,
                    "response_judgment": response_judgment,
                    "response_value": response_value,
                    **(
                        {
                            "thinking_judgment": thinking_judgment,
                            "thinking_value": thinking_value,
                        }
                        if judges_thinking
                        else {}
                    ),
                }
            )

        retired = total - len(judged)
        retired_display = f", [bold]{retired}[/]/{total} retired" if retired else ""
        retired_md = f", {retired}/{total} retired" if retired else ""
        response_value = sum(response_values) / total
        if judges_thinking:
            thinking_value = sum(thinking_values) / total
            return Score(
                value=(thinking_value + response_value) / 2,
                rich_display=f"[bold]{thinking_value:.4f}[/] thinking, [bold]{response_value:.4f}[/] response, [bold]{missing}[/]/{total} without tool call{retired_display}",
                md_display=f"{thinking_value:.4f} thinking, {response_value:.4f} response, {missing}/{total} without tool call{retired_md}",
                components={
                    "thinking": thinking_value,
                    "response": response_value,
                    "missing_tool_calls": missing / total,
                },
                records=records,
            )

        return Score(
            value=response_value,
            rich_display=f"[bold]{response_value:.4f}[/], [bold]{missing}[/]/{total} without tool call{retired_display}",
            md_display=f"{response_value:.4f}, {missing}/{total} without tool call{retired_md}",
            components={"missing_tool_calls": missing / total},
            records=records,
        )

    def _judge(
        self,
        instructions: list[str],
        categories: list[list[Category]],
        transcripts: list[str],
    ) -> list[dict[str, Any]]:
        """
        Returns, for each item, the judgment of `transcripts[i]` under `instructions[i]`:
        the category the judge chose from `categories[i]`, its value, the judge's reasoning,
        the generation id (OpenRouter's generation stats, looked up by it, name the
        endpoint that served the request) and the cost.
        """

        judgments: dict[int, dict[str, Any]] = {}
        print(f"    [grey50]* Judging {len(transcripts)} texts...[/]")

        # Waiting happens here rather than in the workers, so that Ctrl+C stops it.
        while len(judgments) < len(transcripts):
            pending = [i for i in range(len(transcripts)) if i not in judgments]
            # Concurrency scales with the generation batch size, so that judging
            # keeps pace with generation.
            with ThreadPoolExecutor(
                max_workers=max(8, self.heretic_settings.batch_size)
            ) as executor:
                futures = [
                    executor.submit(
                        self._send, instructions[i], categories[i], transcripts[i]
                    )
                    for i in pending
                ]

            for i, future in zip(pending, futures):
                try:
                    result = future.result()
                    category = self._parse_category(result, categories[i])
                    usage = result.usage
                    judgments[i] = {
                        "category": category.label,
                        "value": category.value,
                        "reasoning": result.choices[0].message.reasoning,
                        "generation": result.id,
                        "cost": None if usage is None else usage.cost,
                    }
                except Exception as error:
                    raise_if_permanent(error)
                    message = escape(f"{type(error).__name__}: {error}")
                    print(f"[bold red]Judge request failed: {message}[/]")

            if len(judgments) < len(transcripts):
                print("[bold red]Retrying the failed judge requests in a minute...[/]")
                time.sleep(60)

        return [judgments[i] for i in range(len(transcripts))]

    def _send(
        self,
        instructions: str,
        categories: list[Category],
        transcript: str,
    ) -> components.ChatResult:
        # Models that always think reject a tool_choice that forces a tool, so the
        # tool is requested in the prompt instead, and _parse_category() fails the
        # request if the judge does not call it.
        return self.client.chat.send(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": f"{transcript}\n\nSubmit the category with the {TOOL_NAME} tool.",
                },
            ],
            tools=[get_category_tool(categories)],
            provider=self.provider,
            temperature=self.temperature,
            stream=False,
        )

    @staticmethod
    def _parse_category(
        result: components.ChatResult,
        categories: list[Category],
    ) -> Category:
        values = {
            normalize_identifier(category.label): category for category in categories
        }

        for tool_call in result.choices[0].message.tool_calls or []:
            if normalize_identifier(tool_call.function.name) != normalize_identifier(
                TOOL_NAME
            ):
                continue

            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise TypeError(f"Judge arguments are not an object: {arguments!r}")

            labels = [
                label
                for name, label in arguments.items()
                if normalize_identifier(name) == normalize_identifier(ARGUMENT_NAME)
            ]
            if (
                len(labels) != 1
                or not isinstance(labels[0], str)
                or normalize_identifier(labels[0]) not in values
            ):
                raise ValueError(
                    f"Judge arguments do not name exactly one configured category: {arguments!r}"
                )
            return values[normalize_identifier(labels[0])]

        raise ValueError(f"Judge did not call the {TOOL_NAME} tool")
