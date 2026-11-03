"""Builders shared by the harness tests (imported directly: ``from fakes import ...``).

The tiny cohere2 model and tokenizer mimic Tiny Aya's structure: BOS, a system turn holding a
default preamble that is always injected, the user turn, then the chatbot turn as the generation
prompt. Nothing is downloaded.
"""

from __future__ import annotations

import pytest

from lift.schemas import PromptRecord

SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK_TOKEN>", "<BOS_TOKEN>", "<EOS_TOKEN>",
    "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
    "<|SYSTEM_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>",
]
TINY_PREAMBLE = "# System Preamble\nYou are in contextual safety mode. This tiny preamble is always injected."
# Like Tiny Aya's template (checked against the Hub's copy on Nov 3): the generation prompt ends in
# <|START_RESPONSE|>, and a system message never replaces the default preamble; it is appended to
# the same system turn. One line: transformers renders templates with trim_blocks and lstrip_blocks.
TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set user_preamble = messages[0]['content'] %}{% set loop_messages = messages[1:] %}"
    "{% else %}{% set user_preamble = none %}{% set loop_messages = messages %}{% endif %}"
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" + TINY_PREAMBLE.replace("\n", "{{ '\\n' }}")
    + "{% if user_preamble %}{{ '\\n# Developer Preamble\\n' + user_preamble }}{% endif %}"
    "<|END_OF_TURN_TOKEN|>"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}"
    "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% endif %}"
)
# Sampling defaults and logits processors a checkpoint might ship; greedy decoding must ignore them.
POISONED_GENERATION_CONFIG = {
    "do_sample": True, "temperature": 0.3, "top_k": 5, "top_p": 0.75, "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 2, "max_length": 20,
}


def make_prompt(pid: str, lang: str = "en", set_name: str = "harmful", split: str = "dev",
                text: str | None = None) -> PromptRecord:
    return PromptRecord(pid=pid, lang=lang, set=set_name, split=split,
                        text=text if text is not None else f"prompt {pid} {lang}", source="test")


def build_tiny_tokenizer(chat_template: str = TINY_CHAT_TEMPLATE):
    """Byte-level BPE with no merges: every UTF-8 byte is one token, so Hindi, Chinese or emoji
    encode losslessly. Like the Cohere tokenizer, a plain ``tokenizer(text)`` prepends BOS."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<UNK>", fuse_unk=False))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS])
    backend.post_processor = processors.TemplateProcessing(
        single="<BOS_TOKEN> $A:0", pair="<BOS_TOKEN> $A:0 <BOS_TOKEN> $B:1",
        special_tokens=[("<BOS_TOKEN>", vocab["<BOS_TOKEN>"])],
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<BOS_TOKEN>", eos_token="<|END_OF_TURN_TOKEN|>",
        unk_token="<UNK>", pad_token="<PAD>", padding_side="left",
        model_input_names=["input_ids", "attention_mask"], model_max_length=1024,
        chat_template=chat_template, clean_up_tokenization_spaces=False,
    )


def build_tiny_cohere2(tokenizer, *, seed: int = 0, sliding_window: int = 16, max_positions: int = 1024):
    """A randomly initialised 4-layer Cohere2ForCausalLM (hidden 64, tied embeddings, three
    sliding-window layers then one global layer, like Tiny Aya's 3:1 pattern), fp32, eval mode.

    initializer_range 0.2, not the default 0.02: at 0.02 a random tied-embedding model copies its
    last input token forever, so every row is identical and a broken mask would go unnoticed."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(seed)
    config = transformers.Cohere2Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=max_positions,
        sliding_window=sliding_window,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"), tie_word_embeddings=True,
        initializer_range=0.2,
    )
    return transformers.Cohere2ForCausalLM(config).eval()
