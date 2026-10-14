# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt

from heretic.scorer import Context, Score, Scorer
from heretic.utils import Prompt, ends_with_call, find_xml_calls

SUBMIT = "submit_answer"
ABSTAIN = "report_unanswerable"

SYSTEM_PROMPT = (
    "You are solving technical problems. Work each problem out carefully before answering. "
    "Submit your final answer by calling exactly one of the provided tools, exactly once: "
    f"{SUBMIT} with every field filled in, or {ABSTAIN} if the problem cannot be solved as stated "
    "(for example, because required information is missing or something it refers to does not exist). "
    "Use the units and formats given in the field descriptions. Do not ask the user questions."
)

ABSTAIN_TOOL = {
    "type": "function",
    "function": {
        "name": ABSTAIN,
        "description": (
            f"Call this instead of {SUBMIT} when the problem cannot be solved as stated, for example "
            "because required information is missing or something it refers to does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "missing_information": {
                    "type": "string",
                    "description": "What is missing or does not exist, in a short phrase.",
                },
            },
            "required": ["missing_information"],
            "additionalProperties": False,
        },
    },
}


class Settings(BaseModel):
    max_response_length: PositiveInt = Field(
        default=8192,
        description=(
            "Maximum number of tokens to generate for each answer, including any reasoning "
            "the model writes before its tool call."
        ),
    )

    max_thinking_length: PositiveInt = Field(
        default=65536,
        description="Maximum number of thinking tokens for each answer in thinking mode.",
    )

    excluded_questions: list[str] = Field(
        default_factory=list,
        description='Ids of questions to leave out, such as "Q5a" (the ids are in quality_canary.json).',
    )


def get_submit_tool(question: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for field in question["fields"]:
        spec = {"type": field["type"], "description": field["description"]}
        if field["type"] == "array":
            spec["items"] = {"type": "string"}
        for name in ("enum", "pattern"):
            if name in field:
                spec[name] = field[name]
        properties[field["name"]] = spec

    return {
        "type": "function",
        "function": {
            "name": SUBMIT,
            "description": "Submit the final answer to the problem. Fill in every field.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def parse_value(value: str) -> Any:
    """An argument value from a <parameter> block: JSON arrays and objects are decoded."""
    if value.lstrip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def get_call(text: str, syntax: str = "json") -> tuple[str, Any] | None:
    """
    Returns the name and arguments of the last call to either tool in the text, or None
    if there is none or anything but the template's call syntax follows it. Reasoning may
    come before the call. In the "xml" syntax, calls are read as <function=...> blocks; in
    the "json" syntax, the arguments are the first JSON object after the tool's name,
    which locates them in the usual chat template call syntaxes.
    """
    if syntax == "xml":
        calls = [call for call in find_xml_calls(text) if call[2] in (SUBMIT, ABSTAIN)]
        if not calls:
            return None
        _, end, name, arguments = calls[-1]
        if not ends_with_call(text, end):
            return None
        return name, {key: parse_value(value) for key, value in arguments.items()}

    start, name = max((text.rfind(name), name) for name in (SUBMIT, ABSTAIN))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    try:
        arguments, end = json.JSONDecoder().raw_decode(text, brace)
    except ValueError:
        return None
    if not ends_with_call(text, end):
        return None
    return name, arguments


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Thousands separators, as in "215,837.01".
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", value.strip())
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
            return float(text)
    return None


def as_lines(value: Any) -> list[str] | None:
    if isinstance(value, str):
        value = value.split("\n")
    if not isinstance(value, list):
        return None
    lines = [str(line).rstrip() for line in value]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def as_datetime(value: Any) -> tuple[int, ...] | None:
    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})[ T]+(\d{1,2}):(\d{2})", str(value).strip()
    )
    return tuple(int(part) for part in match.groups()) if match else None


def is_correct(check: dict[str, Any], value: Any) -> bool:
    """
    Whether the value is the answer and nothing else: a number without units or words,
    a date and time in the requested format alone, a peptide without separators or stop
    symbol. Only case, spacing and punctuation of option names are not compared.
    """
    kind, key = check["kind"], check["key"]
    if kind in ("num", "int"):
        number = as_number(value)
        if number is None:
            return False
        if kind == "int":
            return number == key
        tolerance = check["tolerance"]
        limit = max(tolerance["absolute"], tolerance["relative"] * abs(key))
        # The margin absorbs floating-point error when an answer lies exactly on the limit.
        return abs(number - key) <= limit * (1 + 1e-9)
    if kind == "enum":
        return re.sub(r"[^a-z0-9]", "", str(value).lower()) == re.sub(
            r"[^a-z0-9]", "", key.lower()
        )
    if kind == "protein":
        return str(value).strip().upper() == key
    if kind == "datetime":
        return as_datetime(value) == as_datetime(key)
    if kind == "lines":
        return as_lines(value) == key
    if kind == "line":
        return str(value).strip().strip("`").strip() == key
    raise ValueError(f"Unknown check kind: {kind}")


def is_passed(question: dict[str, Any], call: tuple[str, Any] | None) -> bool:
    if call is None:
        return False
    name, arguments = call
    if not isinstance(arguments, dict):
        return False
    # The call carries the answer and nothing else, so arguments the tool lacks fail it.
    if question["expect"] == "abstain":
        return name == ABSTAIN and set(arguments) <= {"missing_information"}
    if name != SUBMIT or not set(arguments) <= {f["name"] for f in question["fields"]}:
        return False
    return all(
        field["name"] in arguments
        and is_correct(field["check"], arguments[field["name"]])
        for field in question["fields"]
        if "check" in field
    )


class QualityCanary(Scorer):
    """
    Fraction of 20 applied-reasoning questions, less any excluded in the settings, that
    the model fails. Each question needs domain knowledge the prompt does not supply, and
    the model must answer it by calling a tool with typed fields, or a second tool if the
    question cannot be answered as stated (three of the 20 cannot). A question is passed when the right tool is called
    and its graded fields match the answer keys.

    The questions come from the lora_canary suite; their keys are computed in that suite.
    The model may reason before calling the tool, in its thinking or its response, but
    nothing may follow the call, and the call must carry exactly the answer: no arguments
    the tool lacks, and values in the requested formats without anything added.
    """

    settings: Settings

    @property
    def score_name(self) -> str:
        return "Quality canary"

    @property
    def reproducible(self) -> bool:
        return True

    def init(self, ctx: Context) -> None:
        questions = json.loads(
            Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
        )
        excluded = set(self.settings.excluded_questions)
        unknown = excluded - {question["id"] for question in questions}
        if unknown:
            raise ValueError(
                f"Unknown quality canary question ids: {', '.join(sorted(unknown))}"
            )
        self.questions = [
            question for question in questions if question["id"] not in excluded
        ]
        self.prompts = [
            Prompt(
                system=SYSTEM_PROMPT,
                user=question["prompt"],
                tools=[get_submit_tool(question), ABSTAIN_TOOL],
                max_response_length=self.settings.max_response_length,
                max_thinking_length=self.settings.max_thinking_length,
            )
            for question in self.questions
        ]

    def get_score(self, ctx: Context) -> Score:
        if self.heretic_settings.thinking:
            responses = [rollout.answer for rollout in ctx.get_rollouts(self.prompts)]
        else:
            responses = ctx.get_responses(self.prompts)

        syntax = ctx.get_model().tool_call_syntax
        calls = [get_call(response, syntax) for response in responses]
        total = len(self.questions)
        failed = sum(
            not is_passed(question, call)
            for question, call in zip(self.questions, calls)
        )
        missing = sum(call is None for call in calls)

        return Score(
            value=failed / total,
            rich_display=f"[bold]{failed}[/]/{total} failed, [bold]{missing}[/]/{total} without tool call",
            md_display=f"{failed}/{total} failed, {missing}/{total} without tool call",
            components={"missing_tool_calls": missing / total},
        )
