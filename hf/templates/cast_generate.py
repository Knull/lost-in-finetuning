"""Generate with the CAST repair through the code the evaluation used.

The weights in this repo are the adapter-merged model, saved from the evaluated variant
(``lift.models.variants.build_variant``); on their own they are the unrepaired adapter. The repair is
applied at generation time by ``lift.repair.cast.make``, the function the evaluation's CAST builder
calls, with the same inputs: the artifact file ``repair/cast/cast.safetensors`` (checked against the
sha256 pinned in ``repair/params.json``), the frozen strength, the gate (on for "cast", always on for
"cast_always") and the tokenizer's pad id. Install the project at the frozen recipe tag first:

    pip install "lift[gpu] @ git+<github repo url>@<recipe-frozen tag>"

Usage:

    from cast_generate import load
    model, tokenizer = load(".")          # the downloaded repo folder
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "..."}],
                                        add_generation_prompt=True, return_tensors="pt")
    out = model.generate(ids.to("cuda"), max_new_tokens=256, do_sample=False)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

ARTIFACT = "cast.safetensors"  # lift.repair.cast.ARTIFACT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(
    repo_dir: str | Path = ".",
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    make: Callable[..., Any] | None = None,
):
    """(the CAST-wrapped model, its tokenizer). ``make`` defaults to ``lift.repair.cast.make``."""
    repo_dir = Path(repo_dir)
    params = json.loads((repo_dir / "repair" / "params.json").read_text(encoding="utf-8"))
    if params.get("method") not in ("cast", "cast_always"):
        raise ValueError(f"repair/params.json names a {params.get('method')!r} repair, not CAST")
    artifact = repo_dir / "repair" / "cast" / ARTIFACT
    pinned = (params.get("artifacts") or {}).get("cast")
    if not pinned or _sha256(artifact) != pinned:
        raise ValueError(
            f"{artifact} does not match the sha256 pinned in repair/params.json (artifacts.cast): "
            "download the repo again"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if make is None:
        from lift.repair.cast import make

    tokenizer = AutoTokenizer.from_pretrained(repo_dir)
    model = AutoModelForCausalLM.from_pretrained(repo_dir, torch_dtype=getattr(torch, dtype)).to(device)
    wrapped = make(
        model,
        artifact,
        float(params["params"]["strength"]),
        gate=params["method"] == "cast",
        pad_id=tokenizer.pad_token_id,
    )
    return wrapped, tokenizer
