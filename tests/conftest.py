"""Shared fixtures. Every test runs with data/, outputs/ and validation/ moved into a tmp dir, so
nothing a test writes can land in the repo or be pushed by ``lift sync``.

Model tests use a tiny randomly initialised cohere2 model and an offline tokenizer whose chat
template mimics Tiny Aya's structure (BOS, a system turn holding a default preamble that is always
injected, the user turn, then the chatbot turn as the generation prompt). Nothing is downloaded:
huggingface_hub runs in offline mode unless the run selects the ``net`` marker. Tests that use these
fixtures are marked ``torch``; the fixtures skip when torch, transformers or peft are missing.

Tests never depend on the revisions pinned in configs/project.yaml: every test sees them as
"main" (``unpinned_revisions``), so pinning the shas Gate 0 prints changes no test's outcome.
"""

from __future__ import annotations

import copy
import os
import re
import sys

import pytest
from fakes import POISONED_GENERATION_CONFIG, build_tiny_cohere2, build_tiny_tokenizer

from lift import config as lift_config
from lift.schemas import PromptRecord

PINNED_ENTRIES = ("models.global", "models.base", "judges.safety_primary")


def pytest_configure(config):
    """Rule 7: no test reaches the Hub. huggingface_hub reads HF_HUB_OFFLINE at import, so the module
    flag is set too when something imported it already."""
    if re.search(r"(?<!not )\bnet\b", config.getoption("markexpr") or ""):
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        hub_constants.HF_HUB_OFFLINE = True


@pytest.fixture(autouse=True)
def unpinned_revisions(monkeypatch):
    """Serve configs/project.yaml with every pinnable revision set to "main". A test that needs a
    pin sets it itself: it mutates this view (one copy per test, like the real cached load) or
    monkeypatches lift.config.load or lift.config.get."""
    real = lift_config.load
    views: dict[str, dict] = {}

    def load(name: str = "project") -> dict:
        if name != "project":
            return real(name)
        if name not in views:
            data = copy.deepcopy(real(name))
            for dotted in PINNED_ENTRIES:
                node = data
                for part in dotted.split("."):
                    node = node.get(part) if isinstance(node, dict) else None
                if isinstance(node, dict) and "revision" in node:
                    node["revision"] = "main"
            views[name] = data
        return views[name]

    monkeypatch.setattr(lift_config, "load", load)


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Point LIFT_DATA_DIR, LIFT_OUTPUT_DIR and LIFT_VALIDATION_DIR at a fresh tmp tree."""
    root = tmp_path / "lift-tree"
    for name, var in (("data", "LIFT_DATA_DIR"), ("outputs", "LIFT_OUTPUT_DIR"),
                      ("validation", "LIFT_VALIDATION_DIR")):
        (root / name).mkdir(parents=True)
        monkeypatch.setenv(var, str(root / name))
    return root


@pytest.fixture
def fake_prompt_files(monkeypatch):
    """Install an in-memory prompt store behind lift.data.build.load_prompts.

    Returns a dict {(set, lang): [PromptRecord, ...]} that tests fill in; load_prompts filters it
    by split exactly like the real reader, and takes its ``allow_short`` (the store has no frozen split
    to be short of).
    """
    store: dict[tuple[str, str], list[PromptRecord]] = {}

    def load_prompts(set_name: str, lang: str, split: str | None = None, *,
                     allow_short: bool = False) -> list:
        if (set_name, lang) not in store:
            raise FileNotFoundError(f"no prompts for {set_name}_{lang}")
        return [r for r in store[(set_name, lang)] if split is None or r.split == split]

    monkeypatch.setattr("lift.data.build.load_prompts", load_prompts)
    return store


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return build_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_cohere2(tiny_tokenizer):
    """The in-memory tiny model (fp32, CPU). Do not mutate it; load ``tiny_model_dir`` instead."""
    return build_tiny_cohere2(tiny_tokenizer)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tiny_tokenizer):
    """The tiny model (same weights as ``tiny_cohere2``) and tokenizer saved like a checkpoint, with
    a generation_config.json full of sampling defaults (``POISONED_GENERATION_CONFIG``) that the
    harness must neutralise."""
    from transformers import GenerationConfig

    model = build_tiny_cohere2(tiny_tokenizer)
    model.generation_config = GenerationConfig(
        bos_token_id=tiny_tokenizer.bos_token_id, pad_token_id=tiny_tokenizer.pad_token_id,
        eos_token_id=[tiny_tokenizer.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")],
        **POISONED_GENERATION_CONFIG,
    )
    path = tmp_path_factory.mktemp("tiny-cohere2")
    model.save_pretrained(path)
    tiny_tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_lora_adapter(tmp_path_factory, tiny_model_dir):
    """A PEFT LoRA adapter on the tiny model with non-zero random weights (so it changes outputs),
    saved to tmp. Returns the adapter directory."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    torch.manual_seed(1)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, init_lora_weights=False,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                          "down_proj"], task_type="CAUSAL_LM")
    model = peft.get_peft_model(base, cfg)
    path = tmp_path_factory.mktemp("tiny-adapter")
    model.save_pretrained(path)
    return path
