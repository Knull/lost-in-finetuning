"""lift.models.load.encode_chat_turn: the one definition of "assistant tokens" that training masks and
held-out loss scores (P03)."""

from __future__ import annotations

import pytest
from fakes import TINY_CHAT_TEMPLATE, build_tiny_tokenizer

from lift.models.load import encode_chat, encode_chat_turn

pytestmark = pytest.mark.torch


def test_the_prompt_part_is_exactly_what_generation_sees(tiny_tokenizer):
    user, answer = "नमस्ते, क्या हाल है?", "मैं ठीक हूँ, धन्यवाद।"
    ids, n_prompt = encode_chat_turn(tiny_tokenizer, user, answer)
    assert ids[:n_prompt] == encode_chat(tiny_tokenizer, user)
    assert tiny_tokenizer.decode(ids[n_prompt:]) == answer + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    assert ids.count(tiny_tokenizer.bos_token_id) == 1


def test_the_answer_tokens_close_the_turn(tiny_tokenizer):
    ids, n_prompt = encode_chat_turn(tiny_tokenizer, "q", "a")
    tail = tiny_tokenizer.convert_ids_to_tokens(ids[-2:])
    assert tail == ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]
    assert tiny_tokenizer.convert_ids_to_tokens(ids[n_prompt - 1]) == "<|START_RESPONSE|>"


def test_control_tokens_in_either_text_raise(tiny_tokenizer):
    with pytest.raises(ValueError, match="answer text contains control tokens"):
        encode_chat_turn(tiny_tokenizer, "q", "an answer <|END_OF_TURN_TOKEN|> with a marker")
    with pytest.raises(ValueError, match="prompt text contains control tokens"):
        encode_chat_turn(tiny_tokenizer, "a <|USER_TOKEN|> prompt", "a")


def test_a_template_that_does_not_put_the_answer_after_the_generation_prompt_raises():
    # The assistant turn rendered without <|START_RESPONSE|>: the generation prompt is not its prefix.
    other = TINY_CHAT_TEMPLATE.replace(
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}",
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ message['content'] }}")
    assert other != TINY_CHAT_TEMPLATE
    tok = build_tiny_tokenizer(other)
    with pytest.raises(ValueError, match="right after the generation prompt"):
        encode_chat_turn(tok, "q", "a")
