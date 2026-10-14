# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import math
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import httpx
import huggingface_hub
import torch
from openrouter import OpenRouter, components
from openrouter.errors import OpenRouterError
from openrouter.types import UNSET
from rich.markup import escape
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    Mistral3ForConditionalGeneration,
    PreTrainedTokenizerBase,
)

from .config import Settings
from .utils import print

SHIELDSTRAL_MODEL = "mistralai/Shieldstral-1.0-3B"
QUESTION = "Is the model output degenerate?"


# HTTP statuses of OpenRouter errors that retrying cannot fix. Out of credits (402) is
# not among them, because credits can be added while the failed requests wait.
PERMANENT_ERROR_STATUSES = {400, 401, 403, 404, 413, 422}


def raise_if_permanent(error: Exception) -> None:
    """
    Raises if a failed judge request cannot succeed on retry, so that the run stops
    instead of retrying it forever. Rate limits, overloaded providers, server and
    network errors, and unusable replies are left to be retried.
    """
    if (
        isinstance(error, OpenRouterError)
        and error.status_code in PERMANENT_ERROR_STATUSES
    ):
        raise RuntimeError(
            f"Judge request failed with HTTP {error.status_code}, which retrying "
            f"cannot fix: {error.body}"
        ) from error


FP8_OR_HIGHER: list[components.Quantization] = ["fp8", "mxfp8", "fp16", "bf16", "fp32"]


def get_provider_preferences(
    client: OpenRouter,
    model: str,
    providers: list[str] | None,
) -> components.ProviderPreferences:
    """
    Routes OpenRouter requests to the given providers, in order and with no fallback
    to others. If none are given, any of the model's endpoints known to be at fp8 or
    higher precision may serve them. Either way, a provider must support every
    parameter the request uses.
    """
    if providers is not None:
        return components.ProviderPreferences(
            order=providers,
            allow_fallbacks=False,
            require_parameters=True,
        )

    author, _, slug = model.partition("/")
    endpoints = client.endpoints.list(author=author, slug=slug).data.endpoints
    quantizations = {endpoint.quantization for endpoint in endpoints}
    if quantizations <= set(FP8_OR_HIGHER):
        return components.ProviderPreferences(require_parameters=True)
    if not quantizations & set(FP8_OR_HIGHER):
        raise ValueError(
            f"No OpenRouter endpoint of {model} is known to be at fp8 or higher "
            "precision, so one can only be used by setting its provider."
        )
    print(f"* Endpoints of {model} below fp8 or of unknown precision are excluded")
    return components.ProviderPreferences(
        quantizations=FP8_OR_HIGHER,
        require_parameters=True,
    )


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
        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(SHIELDSTRAL_MODEL, revision=commit),
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
            # This cast is valid because BatchEncoding is the return type
            # with tokenize=True (the default) and return_dict=True.
            inputs = cast(
                BatchEncoding,
                self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=True,
                ),
            ).to(self.model.device)

            logits = self.model(**inputs).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)

            # Values from the model card; if neither token is a candidate, the
            # division below fails, which is the intended loud failure.
            yes, no = -1e9, -1e9
            values, indices = logprobs.topk(20)
            for logprob, token_id in zip(values.tolist(), indices.tolist()):
                # decode returns a string for a flat list of ids.
                token = cast(str, self.tokenizer.decode([token_id])).strip().lower()
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
        reasons = model.data.reasoning is not None and "reasoning" in supported
        self.reasoning = (
            components.ChatRequestReasoning(effort="low") if reasons else None
        )
        self.confirmation_reasoning = (
            components.ChatRequestReasoning(effort="high") if reasons else None
        )
        self.provider = get_provider_preferences(
            self.client, settings.openrouter_model, settings.openrouter_provider
        )

    def _send(
        self,
        request: str,
        text: str,
        reasoning: components.ChatRequestReasoning | None,
    ) -> components.ChatResult:
        system_prompt = self.settings.openrouter_system_prompt
        return self.client.chat.send(
            model=self.settings.openrouter_model,
            messages=[
                *(
                    []
                    if system_prompt is None
                    else [{"role": "system", "content": system_prompt}]
                ),
                {
                    "role": "user",
                    "content": (
                        f"{self.settings.judge_criteria} {QUESTION}\n\n"
                        f"[User request]\n{request}\n\n"
                        f"[Model output so far]\n{text}"
                    ),
                },
            ],
            response_format=self.VERDICT,
            provider=self.provider,
            reasoning=reasoning,
            temperature=self.temperature,
            stream=False,
        )

    @staticmethod
    def _verdict(result: components.ChatResult) -> float:
        verdict = json.loads(str(result.choices[0].message.content))["degenerate"]
        if not isinstance(verdict, bool):
            raise TypeError(f"Judge verdict is not a boolean: {verdict!r}")
        return float(verdict)

    def score(self, requests: list[str], texts: list[str]) -> list[float]:
        verdicts = self._verdicts(requests, texts, self.reasoning)
        # A degenerate verdict retires the rollout and can prune the whole trial, so a judge
        # that can reason confirms it at a higher effort first.
        flagged = [
            i
            for i, verdict in enumerate(verdicts)
            if verdict > self.settings.judge_threshold
        ]
        if flagged and self.confirmation_reasoning is not None:
            confirmed = self._verdicts(
                [requests[i] for i in flagged],
                [texts[i] for i in flagged],
                self.confirmation_reasoning,
            )
            kept = sum(verdict > self.settings.judge_threshold for verdict in confirmed)
            print(
                f"    [grey50]* Judge confirmed {kept} of {len(flagged)} degenerate verdicts[/]"
            )
            for i, verdict in zip(flagged, confirmed):
                verdicts[i] = verdict
        return verdicts

    def _verdicts(
        self,
        requests: list[str],
        texts: list[str],
        reasoning: components.ChatRequestReasoning | None,
    ) -> list[float]:
        verdicts: dict[int, float] = {}

        # Waiting happens here rather than in the workers, so that Ctrl+C stops it.
        while len(verdicts) < len(requests):
            pending = [i for i in range(len(requests)) if i not in verdicts]
            # The requests are the rollouts of one generation batch, all sent at once so
            # that generation waits for a single round of verdicts.
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = [
                    executor.submit(self._send, requests[i], texts[i], reasoning)
                    for i in pending
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
                    raise_if_permanent(error)
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
