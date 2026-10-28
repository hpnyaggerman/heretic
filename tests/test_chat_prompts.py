# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from heretic.config import Settings
from heretic.model import Model, chat_template_supports_system_role
from heretic.utils import Prompt

# A template that rejects the system role the way Gemma 2's does.
REJECTING_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}"
    "{% endif %}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

# A template that renders the system role.
ACCEPTING_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

PROMPT = Prompt(system="Answer briefly.", user="What is a rotor?")
FOLDED_USER_TURN = (
    "<start_of_turn>user\nAnswer briefly.\n\nWhat is a rotor?<end_of_turn>"
)


def make_model(template: str, thinking: bool, response_prefix: str = "") -> Model:
    backend = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=template,
    )
    model = Model.__new__(Model)
    model.tokenizer = tokenizer
    model.supports_system_role = chat_template_supports_system_role(tokenizer)
    model.settings = Settings.model_construct(
        thinking=thinking,
        response_prefix=response_prefix,
        thinking_closer="</think>",
    )
    return model


class ChatPromptsTests(unittest.TestCase):
    def test_folds_system_prompt_in_thinking_mode(self) -> None:
        chat_prompt = make_model(REJECTING_TEMPLATE, True)._chat_prompts([PROMPT])[0]
        self.assertIn(FOLDED_USER_TURN, chat_prompt)

    def test_folds_system_prompt_of_prompt_without_thinking(self) -> None:
        prompt = Prompt(system=PROMPT.system, user=PROMPT.user, thinking=False)
        chat_prompt = make_model(REJECTING_TEMPLATE, True)._chat_prompts([prompt])[0]
        self.assertIn(FOLDED_USER_TURN, chat_prompt)

    def test_folds_system_prompt_outside_thinking_mode(self) -> None:
        chat_prompt = make_model(REJECTING_TEMPLATE, False)._chat_prompts([PROMPT])[0]
        self.assertIn(FOLDED_USER_TURN, chat_prompt)

    def test_renders_unchanged_messages_for_accepting_template(self) -> None:
        model = make_model(ACCEPTING_TEMPLATE, False, response_prefix="Sure,")
        expected = model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": PROMPT.system},
                {"role": "user", "content": PROMPT.user},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        self.assertEqual(model._chat_prompts([PROMPT]), [f"{expected}Sure,"])


if __name__ == "__main__":
    unittest.main()
