# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import math
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import httpx
import huggingface_hub
import torch
from openrouter import OpenRouter, components
from openrouter.types import UNSET
from rich.markup import escape
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

from .config import Settings
from .utils import print

SHIELDSTRAL_MODEL = "mistralai/Shieldstral-1.0-3B"
QUESTION = "Is the model output degenerate?"


class Judge(ABC):
    calls = 0
    cost = 0.0

    @abstractmethod
    def score(self, requests: list[str], texts: list[str]) -> list[float]:
        """
        Returns, for each item, the probability that `texts[i]` is degenerate
        output for the user request `requests[i]`.
        """

    def usage(self) -> str:
        line = f"[grey50]Judge: [bold]{self.calls}[/] calls"
        if self.cost:
            line += f", [bold]{self.cost:.4f}[/] credits"
        self.calls, self.cost = 0, 0.0
        return line + "[/]"


class ShieldstralJudge(Judge):
    # Fixed by the model card; the model was trained on this exact prompt.
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        'and the Instruction provided. Note that the answer can only be "yes" or "no".'
    )
    YES = ("yes", "yes.", '"yes"', "'yes'")
    NO = ("no", "no.", '"no"', "'no'")

    def __init__(self, settings: Settings):
        self.settings = settings
        commit = (
            settings.judge_model_commit
            or huggingface_hub.model_info(SHIELDSTRAL_MODEL).sha
        )
        if commit is None:
            raise RuntimeError(f"No commit hash for {SHIELDSTRAL_MODEL}")
        settings.judge_model_commit = commit
        self.tokenizer = AutoTokenizer.from_pretrained(
            SHIELDSTRAL_MODEL, revision=commit
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            SHIELDSTRAL_MODEL,
            revision=commit,
            device_map="auto",
            dtype=torch.bfloat16,
        ).eval()

    def score(self, requests: list[str], texts: list[str]) -> list[float]:
        scores = []

        for request, text in zip(requests, texts):
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "<Instruct>: You are reviewing the partial output of a language "
                        f"model. {self.settings.judge_criteria} Apply a strict standard.\n\n"
                        f"<Query>: {QUESTION}\n\n"
                        f"<Document>: [User]\n{request}\n\n[Assistant]\n{text}"
                    ),
                },
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)

            logits = self.model(**inputs).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)

            # Values from the model card; if neither token is a candidate, the
            # division below fails, which is the intended loud failure.
            yes, no = -1e9, -1e9
            values, indices = logprobs.topk(20)
            for logprob, token_id in zip(values.tolist(), indices.tolist()):
                token = self.tokenizer.decode([token_id]).strip().lower()
                if token in self.YES:
                    yes = max(yes, logprob)
                elif token in self.NO:
                    no = max(no, logprob)

            scores.append(math.exp(yes) / (math.exp(yes) + math.exp(no)))

        self.calls += len(requests)
        return scores


class OpenRouterJudge(Judge):
    VERDICT = components.ChatFormatJSONSchemaConfig(
        type="json_schema",
        json_schema=components.ChatJSONSchemaConfig(
            name="verdict",
            strict=True,
            schema_={
                "type": "object",
                "properties": {"degenerate": {"type": "boolean"}},
                "required": ["degenerate"],
                "additionalProperties": False,
            },
        ),
    )

    def __init__(self, settings: Settings):
        if settings.openrouter_model is None:
            raise ValueError(
                'openrouter_model must be set when thinking_judge is "openrouter".'
            )

        # The client reads the key from the environment; it is never a setting.
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("The OPENROUTER_API_KEY environment variable is not set.")

        self.settings = settings
        # Failed requests are retried by score(), which warns about each failure.
        self.client = OpenRouter(
            client=httpx.Client(timeout=None, follow_redirects=True),
            retry_config=None,
        )

        author, _, slug = settings.openrouter_model.partition("/")
        model = self.client.models.get(author=author, slug=slug)
        # Unsupported parameters are not sent, because require_parameters would
        # exclude every provider of the model.
        supported = model.data.supported_parameters
        self.temperature = 0.0 if "temperature" in supported else UNSET
        self.reasoning = (
            components.ChatRequestReasoning(effort="low")
            if model.data.reasoning is not None and "reasoning" in supported
            else None
        )

    def _send(self, request: str, text: str) -> components.ChatResult:
        return self.client.chat.send(
            model=self.settings.openrouter_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{self.settings.judge_criteria} {QUESTION}\n\n"
                        f"[User request]\n{request}\n\n"
                        f"[Model output so far]\n{text}"
                    ),
                }
            ],
            response_format=self.VERDICT,
            provider=components.ProviderPreferences(require_parameters=True),
            reasoning=self.reasoning,
            temperature=self.temperature,
            stream=False,
        )

    @staticmethod
    def _verdict(result: components.ChatResult) -> float:
        verdict = json.loads(str(result.choices[0].message.content))["degenerate"]
        if not isinstance(verdict, bool):
            raise ValueError(f"Judge verdict is not a boolean: {verdict!r}")
        return float(verdict)

    def score(self, requests: list[str], texts: list[str]) -> list[float]:
        verdicts: dict[int, float] = {}

        # Waiting happens here rather than in the workers, so that Ctrl+C stops it.
        while len(verdicts) < len(requests):
            pending = [i for i in range(len(requests)) if i not in verdicts]
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(self._send, requests[i], texts[i]) for i in pending
                ]

            for i, future in zip(pending, futures):
                try:
                    result = future.result()
                    # A reply is billed even when its verdict is unusable.
                    usage = result.usage
                    if usage is not None and isinstance(usage.cost, float):
                        self.cost += usage.cost
                    verdicts[i] = self._verdict(result)
                except Exception as error:
                    message = escape(f"{type(error).__name__}: {error}")
                    print(f"[bold red]Judge request failed: {message}[/]")

            if len(verdicts) < len(requests):
                print("[bold red]Retrying the failed judge requests in a minute...[/]")
                time.sleep(60)

        self.calls += len(requests)
        return [verdicts[i] for i in range(len(requests))]


def load_judge(settings: Settings) -> Judge | None:
    if settings.thinking_judge == "shieldstral":
        print()
        print(f"Loading judge [bold]{SHIELDSTRAL_MODEL}[/]...")
        return ShieldstralJudge(settings)

    if settings.thinking_judge == "openrouter":
        print()
        print(f"Using OpenRouter judge [bold]{settings.openrouter_model}[/]")
        return OpenRouterJudge(settings)

    return None
