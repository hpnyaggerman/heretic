# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from rich.markup import escape
from safetensors.torch import load_file, save_file
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.hooks import RemovableHandle
from transformers import (
    AttentionInterface,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)
from transformers.cache_utils import DynamicLayer
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .config import QuantizationMethod, Settings
from .judge import Judge
from .system import empty_cache
from .utils import Prompt, batchify, format_duration, format_exception, print


def get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any(("vision_config" in config) for config in configs):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


def detect_tool_call_syntax(tokenizer: PreTrainedTokenizerBase) -> str:
    """
    "xml" if the chat template writes a tool call's arguments as <parameter=name> blocks,
    as Qwen3.5's does, and "json" otherwise, including for templates without tool calls.
    """
    call = {
        "type": "function",
        "function": {"name": "probe", "arguments": {"value": "1"}},
    }
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
    ]
    try:
        # The declared message type has no room for tool calls, which templates do read.
        text = tokenizer.apply_chat_template(cast(Any, messages), tokenize=False)
    except Exception:
        return "json"
    return "xml" if "<parameter=value>" in str(text) else "json"


# Some chat templates (e.g. gpt-oss) insert the current date into the prompt,
# which makes the residuals, and therefore the modified model, depend on the day
# Heretic is run. Rendering templates with a fixed date keeps results reproducible.
def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(format)


def grouped_sdpa_attention_forward(
    module: Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    """
    transformers' SDPA attention, except that a single decoded token with an attention mask
    reads shared KV heads directly instead of a copy of them for every query head.
    """
    batch, heads, length, dim = query.shape
    groups = getattr(module, "num_key_value_groups", 1)
    if (
        length != 1
        or groups == 1
        or attention_mask is None
        or attention_mask.shape[1] != 1
        or kwargs.get("position_bias") is not None
    ):
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )
    # Query head h reads KV head h // groups, so the query heads sharing a KV head
    # take the place of query positions.
    output = F.scaled_dot_product_attention(
        query.reshape(batch, key.shape[1], groups, dim),
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
    )
    return output.reshape(batch, 1, heads, dim), None


# A left-padded batch always has an attention mask, and with a mask, transformers' SDPA attention
# copies the whole KV cache from the KV heads to every query head on each decoded token. With
# Qwen3.5's 4 KV heads and 24 query heads, every token writes and then reads six times the cache.
AttentionInterface.register("sdpa", grouped_sdpa_attention_forward)


# Tokens of room a KV cache layer gains whenever it runs out.
KV_CACHE_GROWTH = 4096


def update_kv_cache_layer(
    self: DynamicLayer,
    key_states: Tensor,
    value_states: Tensor,
    *args: Any,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    transformers' DynamicLayer.update, except that the states are written into buffers with
    room to spare, of which the cache holds views, instead of being concatenated to copies of
    the whole cache.
    """
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    old_keys, old_values = cast(Tensor, self.keys), cast(Tensor, self.values)

    length = self.get_seq_length()
    end = length + key_states.shape[-2]
    buffers = self.__dict__.get("_buffers")
    # Anything that replaced the states, such as selecting rows, left them outside the buffers.
    if (
        buffers is None
        or buffers[0].shape[-2] < end
        or buffers[0].shape[:-2] != key_states.shape[:-2]
        or (length > 0 and old_keys.data_ptr() != buffers[0].data_ptr())
    ):
        buffers = tuple(
            states.new_empty(
                (*states.shape[:-2], end + KV_CACHE_GROWTH, states.shape[-1])
            )
            for states in (key_states, value_states)
        )
        if length > 0:
            buffers[0][..., :length, :] = old_keys
            buffers[1][..., :length, :] = old_values
        self.__dict__["_buffers"] = buffers

    keys, values = buffers
    keys[..., length:end, :] = key_states
    values[..., length:end, :] = value_states
    self.keys = keys[..., :end, :]
    self.values = values[..., :end, :]
    return self.keys, self.values


# Concatenating copies the whole cache on every decoded token, which at long contexts costs
# about as much memory traffic as reading the cache for attention.
DynamicLayer.update = update_kv_cache_layer


# gpt-oss (harmony) answers on the final channel but makes tool calls on the commentary
# channel. For prompts with tools, the response prefix and the thinking closer therefore
# stop before this header, leaving the channel to the model, and the header is removed
# from answers on the final channel.
HARMONY_FINAL_HEADER = "<|channel|>final<|message|>"


def get_base_weight(module: Module) -> Tensor:
    """The module's original weight in float32, dequantized if stored in 4 bits."""
    weight = cast(Tensor, getattr(module, "base_layer", module).weight)
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        weight = BNB_F.dequantize_4bit(weight.data, quant_state)
    return weight.float()


def scale_rows_hook(
    scale: Tensor,
) -> Callable[[Module, tuple[Tensor, ...], Tensor], Tensor]:
    """A forward hook that scales each row of the module's effective weight by `scale`."""

    def hook(module: Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        # The scale is part of the adapter's change, so disabling the adapter, as for
        # benchmarking the original model, disables it too.
        if getattr(module, "disable_adapters", False):
            return output
        bias = cast(Tensor | None, getattr(module, "base_layer", module).bias)
        if bias is None:
            return (output.float() * scale).to(output.dtype)
        return ((output - bias).float() * scale + bias.float()).to(output.dtype)

    return hook


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


@dataclass
class Rollout:
    thinking: str
    # Empty for a retired rollout, which counts as no answer.
    answer: str
    retired: bool
    # The degeneration judge's reviews of the rollout, in order, for the run log.
    checks: list[dict[str, Any]] = field(default_factory=list)
    # What a retired rollout had answered before it was retired.
    retired_answer: str = ""

    @property
    def written_answer(self) -> str:
        """The answer as the model wrote it, a retired rollout's included, for the run log."""
        return self.retired_answer if self.retired else self.answer


class RolloutRetired(Exception):
    """
    Raised when the judge retires a second rollout of the same scorer while
    abort_on_retirement is set.
    """

    def __init__(self, request: str, part: str, outputs: list[dict[str, Any]]):
        super().__init__(f"The {part} for a request degenerated.")
        self.request = request
        # "thinking" or "answer"
        self.part = part
        # The scorer's rollouts as far as they got, each with its request.
        self.outputs = outputs
        # The scores of the scorers that finished before the abort, and the name of the
        # scorer whose rollouts were being generated, set by the evaluator.
        self.scores: list[tuple[str, Any]] = []
        self.scorer = ""


@dataclass
class _Run:
    """
    The padded sequences, attention mask and cache after a generate call. The next call
    on the rollouts that kept every token of it continues from them instead of reading
    their whole sequences again.
    """

    sequences: Tensor
    attention_mask: Tensor
    cache: Any
    rollouts: int = 0


@dataclass
class _RolloutState:
    prompt: Prompt
    prompt_ids: list[int]
    # None while the thinking block is open; otherwise where the answer starts.
    closer_end: int | None
    generated: list[int] = field(default_factory=list)
    # Number of tokens appended to close the thinking block.
    closer_length: int = 0
    judged_length: int = 0
    done: bool = False
    retired: bool = False
    # The run this rollout continues from, and its row there.
    run: _Run | None = None
    row: int = 0
    # The part, thinking or answer, that the flags belong to; the lengths, in characters,
    # of the text that each review flagging it saw; and whether the latest review did.
    flagged_part: str = "thinking"
    flags: list[int] = field(default_factory=list)
    flagged_last: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)

    def remaining(self, settings: Settings) -> int:
        if self.closer_end is None:
            limit = self.prompt.max_thinking_length or settings.max_thinking_length
            return limit - len(self.generated)
        limit = self.prompt.max_response_length or settings.max_response_length
        return self.closer_end + limit - len(self.generated)


GENERATION_PROGRESS_INTERVAL = 1024


class _GenerationProgress(StoppingCriteria):
    """Never stops generation; prints every `interval` tokens how far a generate call has got."""

    def __init__(self, prompt_length: int, end_ids: list[int], interval: int):
        self.prompt_length = prompt_length
        self.end_ids = end_ids
        self.interval = interval
        self.start_time = time.perf_counter()

    def __call__(
        self,
        input_ids: LongTensor,
        scores: tuple[FloatTensor] | None,
        **kwargs: Any,
    ) -> BoolTensor:
        length = input_ids.shape[1] - self.prompt_length
        if length % self.interval == 0:
            # A finished sequence ends in its stop token and, after that, in padding.
            ended = torch.isin(
                input_ids[:, -1],
                torch.tensor(self.end_ids, device=input_ids.device),
            )
            elapsed = time.perf_counter() - self.start_time
            print(
                f"      [grey50]* {length:,} tokens, {int((~ended).sum())}/{len(ended)} still generating, "
                f"{length / elapsed:.1f} tokens/s per sequence, {format_duration(elapsed)}[/]"
            )
        return cast(
            BoolTensor,
            torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device),
        )


class Model:
    # Whether a rollout retired by the judge raises RolloutRetired.
    abort_on_retirement = False

    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self._row_scales: dict[Module, tuple[Tensor, RemovableHandle]] = {}

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            ),
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.tool_call_syntax = (
            detect_tool_call_syntax(self.tokenizer)
            if settings.tool_call_syntax == "auto"
            else settings.tool_call_syntax
        )
        print(f"* Tool call syntax: [bold]{self.tool_call_syntax}[/]")

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        self.trusted_models = set()

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                )
            except Exception as error:
                self.model = None
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Abliterable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def apply_lora(self, lora_rank: int):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # PEFT initializes A randomly from the global RNG. B is zero, so this doesn't
        # change the model, but zeroing A too keeps saved adapters reproducible.
        self._zero_lora_weights()

    def _zero_lora_weights(self):
        for name, module in self.model.named_modules():
            if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
                torch.nn.init.zeros_(module.weight)

    def set_row_scale(self, module: Module, scale: Tensor):
        """
        Scales each row of the module's effective weight, its base weight plus its LoRA
        change, by the matching element of `scale`, which a LoRA alone cannot do.
        The scale lasts until the model is reset.
        """
        if module in self._row_scales:
            self._row_scales[module][1].remove()
        handle = module.register_forward_hook(scale_rows_hook(scale))
        self._row_scales[module] = (scale, handle)

    def get_row_scale(self, module: Module) -> Tensor | None:
        entry = self._row_scales.get(module)
        return None if entry is None else entry[0]

    def _clear_row_scales(self):
        for _, handle in self._row_scales.values():
            handle.remove()
        self._row_scales.clear()

    def _get_row_scales_by_name(self) -> dict[str, Tensor]:
        """The row scales by the names of their modules in the model without its adapters."""
        names = {id(module): name for name, module in self.model.named_modules()}
        return {
            names[id(module)].removeprefix("base_model.model."): scale
            for module, (scale, _) in self._row_scales.items()
        }

    @staticmethod
    def _apply_row_scales(model: Module, scales: dict[str, Tensor]):
        for name, scale in scales.items():
            weight = cast(Tensor, model.get_submodule(name).weight)
            scaled = weight.data.float() * scale.to(weight.device)[:, None]
            weight.data = scaled.to(weight.dtype)

    def save_adapter(self, directory: str, max_shard_size: int | str):
        """
        Saves the LoRA adapter. Row scales cannot be expressed by a LoRA, so with row scales
        the adapter is saved as a DoRA adapter: DoRA rescales each row of W + BA to its
        magnitude, so magnitudes equal to the original row norms ||W|| apply exactly the
        row scales, which are ||W|| / ||W + BA||, and leave unscaled modules unchanged.
        """
        assert isinstance(self.model, PeftModel)
        self.model.save_pretrained(directory, max_shard_size=max_shard_size)
        if not self._row_scales:
            return

        weights_path = os.path.join(directory, "adapter_model.safetensors")
        tensors = load_file(weights_path)
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) and "default" in module.lora_A:
                lora_A_name = f"{name}.lora_A.weight"
                tensors[f"{name}.lora_magnitude_vector"] = (
                    torch.linalg.vector_norm(get_base_weight(module), dim=1)
                    .to(tensors[lora_A_name].dtype)
                    .cpu()
                )
        save_file(tensors, weights_path, metadata={"format": "pt"})

        config_path = os.path.join(directory, "adapter_config.json")
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        config["use_dora"] = True
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            self._apply_row_scales(merged_model, self._get_row_scales_by_name())
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            row_scales = self._get_row_scales_by_name()
            merged_model = self.model.merge_and_unload()
            self._apply_row_scales(merged_model, row_scales)
            # The merge replaced the modules that the row scales were hooked to.
            self._clear_row_scales()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.

        Returns True if the fast path was taken.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            # Zeroing A as well ensures that adapter weights of modules not modified
            # by the next trial don't depend on what previous trials did.
            self._zero_lora_weights()
            self._clear_row_scales()
            return True

        # Purge existing model object from memory to make space.
        self._clear_row_scales()
        self.model = None
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self.needs_reload = False

        return False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

        # Text-only models.
        return model.model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty: ignore[unresolved-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty: ignore[unresolved-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty: ignore[unresolved-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty: ignore[unresolved-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.output_linear)  # ty: ignore[unresolved-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def template_kwargs(self, thinking: bool = True) -> dict[str, Any]:
        # Passed only in thinking mode so that prompts outside it stay byte-identical.
        return {"enable_thinking": thinking} if self.settings.thinking else {}

    def _apply_template(self, prompt: Prompt, thinking: bool) -> str:
        # This cast is valid because str is the return type
        # for a single chat operation with tokenize=False.
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                # This cast is valid because the template only reads the list.
                tools=cast(list[dict | Callable] | None, prompt.tools),
                add_generation_prompt=True,
                tokenize=False,
                # Overrides the function that templates call to get the current date.
                strftime_now=strftime_fixed_date,
                **self.template_kwargs(thinking),
            ),
        )

    def _chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        chat_prompts = []
        for prompt in prompts:
            chat_prompt = self._apply_template(prompt, prompt.thinking)
            # A template's own non-thinking mode, if it has one, is how the model
            # learned to answer directly, so its rendering is used as it is. Closing
            # the block that thinking mode opened instead would keep the rest of
            # thinking mode, such as Qwen3.8's instruction to reason carefully, which
            # the model then follows in its answer.
            if prompt.thinking or chat_prompt == self._apply_template(prompt, True):
                # Append the common response prefix to the prompts so that evaluation happens
                # at the point where responses start to differ for different prompts.
                chat_prompt += self._response_prefix(prompt)
            chat_prompts.append(chat_prompt)
        return chat_prompts

    def _response_prefix(self, prompt: Prompt) -> str:
        prefix = self.settings.response_prefix or ""
        if self.settings.thinking and not prompt.thinking:
            # Closing the thinking block at once makes the model answer without thinking.
            prefix += self.settings.thinking_closer or ""
        if prompt.tools:
            prefix = prefix.removesuffix(HARMONY_FINAL_HEADER)
        return prefix

    def _final_header_length(self, ids: list[int], prompt: Prompt) -> int:
        """
        Length of the harmony final channel header at the start of `ids`
        if the prompt has tools, otherwise 0.
        """
        header = self.tokenizer.encode(HARMONY_FINAL_HEADER, add_special_tokens=False)
        return len(header) if prompt.tools and ids[: len(header)] == header else 0

    def _closer_for(self, closer_ids: list[int], prompt: Prompt) -> list[int]:
        """The thinking closer without the harmony final channel header if the prompt has tools."""
        header = self.tokenizer.encode(HARMONY_FINAL_HEADER, add_special_tokens=False)
        if prompt.tools and closer_ids[-len(header) :] == header:
            return closer_ids[: -len(header)]
        return closer_ids

    def _batch_size(self, prompts: list[Prompt]) -> int:
        """The largest generation batch size that all of the prompts allow."""
        return min(
            (prompt.batch_size or self.settings.batch_size for prompt in prompts),
            default=self.settings.batch_size,
        )

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        inputs = self.tokenizer(
            self._chat_prompts(prompts),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def generate_ids(
        self,
        sequences: list[list[int]],
        max_new_tokens: int,
        stop_ids: list[int],
        stop_string: str | None,
        continued: tuple[_Run, list[int]] | None = None,
    ) -> tuple[list[tuple[list[int], int | None]], _Run]:
        """
        Generates continuations of the sequences or, given a run and the rows of the
        sequences in it, continues them from the run's cache.
        """
        kwargs = self.sampling_kwargs()
        if continued is None:
            inputs = self.tokenizer.pad(
                {"input_ids": sequences},
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            run, rows = continued
            index = torch.tensor(rows, device=run.sequences.device)
            input_ids = run.sequences[index]
            attention_mask = run.attention_mask[index]
            if rows != list(range(len(run.sequences))):
                run.cache.reorder_cache(index)
            kwargs["past_key_values"] = run.cache
            # Only the last token is new, and transformers' chunked prefill would
            # start over from the beginning of the sequences.
            kwargs.pop("prefill_chunk_size", None)
        prompt_length = input_ids.shape[1]
        # The positions transformers would derive from the attention mask. Passed explicitly,
        # they stay two-dimensional: transformers' chunked prefill slices the
        # three-dimensional positions of models such as Qwen3.5 along the wrong dimension,
        # and derives them for a continued call from offsets left by the latest call.
        position_ids = (attention_mask.long().cumsum(-1) - 1).masked_fill(
            attention_mask == 0, 0
        )

        pad_id = self.tokenizer.pad_token_id
        # A call no longer than one interval would only report at its end.
        progress = (
            [
                _GenerationProgress(
                    prompt_length,
                    stop_ids + ([] if pad_id is None else [pad_id]),
                    GENERATION_PROGRESS_INTERVAL,
                )
            ]
            if max_new_tokens > GENERATION_PROGRESS_INTERVAL
            else []
        )

        # Reseeding makes each rollout independent of what was generated before it.
        torch.manual_seed(self.settings.seed)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop_ids,
            pad_token_id=pad_id,
            stop_strings=[stop_string] if stop_string else None,
            tokenizer=self.tokenizer,
            stopping_criteria=StoppingCriteriaList(progress),
            return_dict_in_generate=True,
            **kwargs,
        )  # ty:ignore[call-non-callable]
        output_ids = cast(Tensor, outputs.sequences)

        results = []
        for row in output_ids[:, prompt_length:].tolist():
            # Padding only follows a stop token, so the first one is the true end.
            stop = next((token for token in row if token in stop_ids), None)
            results.append((row if stop is None else row[: row.index(stop)], stop))

        return results, _Run(
            sequences=output_ids,
            attention_mask=torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones(
                        (len(attention_mask), output_ids.shape[1] - prompt_length)
                    ),
                ],
                dim=1,
            ),
            cache=outputs.past_key_values,
        )

    def sampling_kwargs(self) -> dict[str, Any]:
        if self.settings.thinking_sampling == "greedy":
            return {"do_sample": False, **self.settings.thinking_generation_kwargs}
        return dict(self.settings.thinking_generation_kwargs)

    def thinking_closer_ids(self) -> tuple[list[int], int | None]:
        closer = self.settings.thinking_closer
        ids = self.tokenizer.encode(closer, add_special_tokens=False) if closer else []
        # Only a dedicated token can be a stop id; an ordinary token would stop
        # generation spuriously, so other closers are matched as text.
        if ids and ids[0] in self.tokenizer.get_added_vocab().values():
            return ids, ids[0]
        return ids, None

    def get_rollouts(
        self,
        prompts: list[Prompt],
        judge: Judge | None,
    ) -> list[Rollout]:
        settings = self.settings
        closer_ids, stop_id = self.thinking_closer_ids()
        stop_string = (
            settings.thinking_closer if closer_ids and stop_id is None else None
        )

        eos = self.model.generation_config.eos_token_id
        stop_ids = (
            list(eos) if isinstance(eos, list) else ([] if eos is None else [eos])
        )
        # A thinking block closed by a dedicated token, which the closer only follows with
        # whitespace, is left open to the model: a rollout that closes it goes on with its
        # answer in the same call, sharing the batch and its cache, instead of stopping
        # there and having its whole sequence read again for the answer.
        in_stream = (
            stop_id is not None
            # decode returns a string for a flat list of ids.
            and not cast(str, self.tokenizer.decode(closer_ids[1:])).strip()
        )
        if stop_id is not None and not in_stream:
            stop_ids.append(stop_id)

        states = [
            _RolloutState(
                prompt=prompt,
                # This cast is valid because a single unpadded string yields a list of ids.
                prompt_ids=cast(
                    list[int],
                    self.tokenizer(
                        chat_prompt,
                        return_token_type_ids=False,
                    )["input_ids"],
                ),
                # A prompt that does not think starts in its answer.
                closer_end=None if closer_ids and prompt.thinking else 0,
            )
            for prompt, chat_prompt in zip(prompts, self._chat_prompts(prompts))
        ]

        # A scorer's first retired rollout counts as failed; a second aborts the trial.
        retired_count = 0
        start_time = time.perf_counter()
        while active := [state for state in states if not state.done]:
            # Rollouts that closed their thinking within a run answer in it.
            running = [
                state
                for state in active
                if state.closer_end is None or state.run is not None
            ]
            answering = [
                state
                for state in active
                if state.closer_end is not None and state.run is None
            ]
            groups = [(running, True), (answering, True)]
            if settings.judge_first_chunk_only:
                # Rollouts that passed the first-chunk gate think to the end in one call;
                # one that the latest review flagged is judged on until a review passes it,
                # and answers are judged chunk by chunk.
                gated = [
                    state
                    for state in running
                    if state.closer_end is not None
                    or state.judged_length == 0
                    or state.flagged_last
                ]
                groups = [
                    (gated, True),
                    ([state for state in running if state not in gated], False),
                    (answering, True),
                ]
            for group, chunked in groups:
                # Thinking to the end in one call stops at the closer, so that the answer
                # is generated chunk by chunk instead of in the same call.
                call_stop_ids = (
                    [*stop_ids, stop_id]
                    if in_stream and not chunked and stop_id is not None
                    else stop_ids
                )
                batch_size = self._batch_size([state.prompt for state in group])
                for batch in batchify(group, batch_size):
                    retired = self._advance(
                        batch,
                        closer_ids,
                        stop_id,
                        call_stop_ids,
                        stop_string,
                        judge,
                        chunked,
                        # Keeping the caches of several batches between calls would
                        # need more memory than generating them one at a time.
                        len(group) <= batch_size,
                        retired_count,
                    )
                    retired_count += len(retired)
                    if self.abort_on_retirement and retired_count >= 2:
                        raise RolloutRetired(
                            retired[-1].prompt.user,
                            "thinking" if retired[-1].closer_end is None else "answer",
                            [self._output(state) for state in states],
                        )
                    elapsed = time.perf_counter() - start_time
                    tokens = sum(len(state.generated) for state in states)
                    print(
                        f"    [grey50]* Rollouts: {sum(state.done for state in states)}/{len(states)} finished "
                        f"({sum(state.retired for state in states)} retired), {tokens:,} tokens "
                        f"at {tokens / elapsed:,.0f} tokens/s, {format_duration(elapsed)}[/]"
                    )
                    longest = sorted(
                        (
                            (number, state)
                            for number, state in enumerate(states, 1)
                            if not state.done
                        ),
                        key=lambda item: len(item[1].generated),
                        reverse=True,
                    )[:3]
                    if longest:
                        print(
                            "      [grey50]* Longest running: "
                            + ", ".join(
                                f"#{number} ({len(state.generated):,} tokens, "
                                f"{escape(repr(state.prompt.user[:40]))})"
                                for number, state in longest
                            )
                            + "[/]"
                        )

        return [self._rollout(state) for state in states]

    def _output(self, state: _RolloutState) -> dict[str, Any]:
        """The rollout as far as it got, with its request, for the run log."""
        rollout = self._rollout(state)
        return {
            "request": state.prompt.user,
            "thinking": rollout.thinking,
            "answer": rollout.written_answer,
            "retired": rollout.retired,
            "checks": rollout.checks,
        }

    def _closer_start(self, ids: list[int], closer: str) -> int | None:
        if closer not in self.tokenizer.decode(ids, skip_special_tokens=True):
            return None
        # Generation stops right after the closer, so it is found near the end.
        return next(
            start
            for start in range(len(ids) - 1, -1, -1)
            if closer in self.tokenizer.decode(ids[start:], skip_special_tokens=True)
        )

    def _advance(
        self,
        batch: list[_RolloutState],
        closer_ids: list[int],
        stop_id: int | None,
        stop_ids: list[int],
        stop_string: str | None,
        judge: Judge | None,
        chunked: bool,
        continuable: bool,
        retired_count: int,
    ) -> list[_RolloutState]:
        """Generates the next chunk of the batch's rollouts and returns those it retired."""
        settings = self.settings
        max_new_tokens = max(state.remaining(settings) for state in batch)
        # Without a judge, chunks only serve to drop finished rollouts between calls, which
        # pays off only when the cache carries over.
        if chunked and (judge is not None or continuable):
            max_new_tokens = min(max_new_tokens, settings.thinking_chunk_size)

        # A run's cache can only be continued by a batch of all the rollouts that continue it.
        run = batch[0].run
        if run is not None and (
            run.rollouts != len(batch) or any(state.run is not run for state in batch)
        ):
            run = None
        results, next_run = self.generate_ids(
            [state.prompt_ids + state.generated for state in batch],
            max_new_tokens,
            stop_ids,
            stop_string,
            None if run is None else (run, [state.row for state in batch]),
        )
        for state in batch:
            state.run = None

        in_stream = stop_id is not None and stop_id not in stop_ids
        continuing = []
        for state, (new_ids, stop) in zip(batch, results):
            # Whether the rollout keeps every token of the call unchanged, so that its
            # row of the call's output remains its sequence.
            kept = stop is None and len(new_ids) == max_new_tokens

            if (
                in_stream
                and state.closer_end is None
                and stop_id in new_ids[: state.remaining(settings) + 1]
            ):
                end = new_ids.index(stop_id) + 1
                state.generated.extend(new_ids[:end])
                state.closer_length = 1
                state.closer_end = len(state.generated)
                new_ids = new_ids[end:]
            if in_stream and state.closer_end is not None and stop_id in new_ids:
                new_ids, stop = new_ids[: new_ids.index(stop_id)], stop_id

            budget = state.remaining(settings)
            if len(new_ids) > budget:
                new_ids, stop = new_ids[:budget], None
                kept = False
            state.generated.extend(new_ids)

            closed = stop is not None and stop == stop_id
            if stop_string is not None:
                offset = state.closer_end or 0
                start = self._closer_start(state.generated[offset:], stop_string)
                if start is not None:
                    del state.generated[offset + start :]
                    closed = True
                    kept = False

            if state.closer_end is None:
                if closed or state.remaining(settings) <= 0:
                    closer = self._closer_for(closer_ids, state.prompt)
                    state.generated.extend(closer)
                    state.closer_length = len(closer)
                    state.closer_end = len(state.generated)
                    kept = False
                elif stop is not None:
                    state.done = True
            elif closed or stop is not None or state.remaining(settings) <= 0:
                state.done = True
            continuing.append(kept and not state.done)

        if continuable:
            for row, (state, continues) in enumerate(zip(batch, continuing)):
                if continues:
                    state.run, state.row = next_run, row
                    next_run.rollouts += 1

        if judge is None:
            return []

        judged = []
        texts = []
        lengths = []
        for state in batch:
            if state.done and state.closer_end is None:
                continue
            if len(state.generated) == state.judged_length:
                continue
            if (
                settings.judge_first_chunk_only
                and state.closer_end is None
                and state.judged_length > 0
                and not state.flagged_last
            ):
                continue
            segment = (
                state.generated if state.closer_end is None else self._answer_ids(state)
            )
            # decode returns a string for a flat list of ids.
            text = cast(str, self.tokenizer.decode(segment, skip_special_tokens=True))
            if text.strip():
                state.judged_length = len(state.generated)
                judged.append(state)
                texts.append(text)
                lengths.append(len(segment))

        if not judged:
            return []

        for state in judged:
            part = "thinking" if state.closer_end is None else "answer"
            if part != state.flagged_part:
                # The answer is a new text, so the thinking's flags do not carry over.
                state.flagged_part, state.flags, state.flagged_last = part, [], False

        requests = [state.prompt.user for state in judged]
        verdicts = judge.score(requests, texts, [list(state.flags) for state in judged])
        escalated = []
        for index, (state, text, length, verdict) in enumerate(
            zip(judged, texts, lengths, verdicts)
        ):
            flagged = verdict.score > settings.judge_threshold
            state.checks.append(
                {
                    "part": state.flagged_part,
                    "tokens": length,
                    "characters": len(text),
                    "review": "quick",
                    "earlier_flags": list(state.flags),
                    "degenerate": flagged,
                    **verdict.record,
                }
            )
            if flagged:
                state.flags.append(len(text))
                # Only a second flag in a row escalates, so a lone flag never retires.
                if state.flagged_last:
                    escalated.append(index)
            state.flagged_last = flagged

        if not escalated:
            return []

        # Confirming cancels the trial if the scorer already lost a rollout; two confirmed
        # together cancel it as well, though each review is told it costs only its answer.
        confirmations = judge.confirm(
            [requests[index] for index in escalated],
            [texts[index] for index in escalated],
            [list(judged[index].flags) for index in escalated],
            self.abort_on_retirement and retired_count >= 1,
        )
        retired = []
        for index, verdict in zip(escalated, confirmations):
            state = judged[index]
            confirmed = verdict.score > settings.judge_threshold
            state.checks.append(
                {
                    "part": state.flagged_part,
                    "tokens": lengths[index],
                    "characters": len(texts[index]),
                    "review": "careful",
                    "earlier_flags": list(state.flags),
                    "degenerate": confirmed,
                    **verdict.record,
                }
            )
            if confirmed:
                state.done = True
                state.retired = True
                if state.run is not None:
                    state.run.rollouts -= 1
                    state.run = None
                retired.append(state)
        return retired

    def _answer_ids(self, state: _RolloutState) -> list[int]:
        assert state.closer_end is not None
        answer_ids = state.generated[state.closer_end :]
        return answer_ids[self._final_header_length(answer_ids, state.prompt) :]

    def _rollout(self, state: _RolloutState) -> Rollout:
        if state.closer_end is None:
            thinking_ids, answer_ids = state.generated, []
        else:
            thinking_ids = state.generated[: state.closer_end - state.closer_length]
            answer_ids = self._answer_ids(state)

        answer = cast(str, self.tokenizer.decode(answer_ids, skip_special_tokens=True))

        return Rollout(
            thinking=cast(
                str, self.tokenizer.decode(thinking_ids, skip_special_tokens=True)
            ),
            answer="" if state.retired else answer,
            retired=state.retired,
            checks=state.checks,
            retired_answer=answer if state.retired else "",
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=max(
                prompt.max_response_length or self.settings.max_response_length
                for prompt in prompts
            ),
        )

        # Extract the newly generated part.
        # This cast is valid because the input_ids property is a Tensor
        # if the tokenizer is invoked with return_tensors="pt", as above.
        rows = outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :].tolist()

        return self.tokenizer.batch_decode(
            [
                row[self._final_header_length(row, prompt) :]
                for prompt, row in zip(prompts, rows)
            ],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        start_time = time.perf_counter()
        for batch in batchify(prompts, self._batch_size(prompts)):
            responses.extend(
                self.get_responses(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                )
            )
            elapsed = time.perf_counter() - start_time
            # Batches take about equally long, so the elapsed time extrapolates.
            remaining = elapsed / len(responses) * (len(prompts) - len(responses))
            print(
                f"    [grey50]* Responses: {len(responses)}/{len(prompts)}, "
                f"{format_duration(elapsed)} elapsed, about {format_duration(remaining)} left[/]"
            )

        return responses

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(
                self.get_residuals(
                    batch,
                    winsorization_quantile=winsorization_quantile,
                )
            )

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_residuals = self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
            )

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # Each module should be invoked at most once per inference step.
                assert module_index not in module_io[layer_index][component]

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        self.generate(prompts, max_new_tokens=1)

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in batchify(prompts, self.settings.batch_size)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **self.template_kwargs(),
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
