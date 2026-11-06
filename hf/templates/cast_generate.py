"""Generate with the CAST repair, as the evaluation did.

The weights in this repo are the adapter-merged model; the repair is applied at generation time
from ``repair/cast/`` and ``repair/params.json``. This wrapper calls the project's own code
(``lift.repair.cast.make``), so install the project at the frozen recipe tag first:

    pip install "lift[gpu] @ git+<github repo url>@<recipe-frozen tag>"

Usage:

    from cast_generate import load
    model, tokenizer = load(".")          # the downloaded repo folder
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "..."}],
                                        add_generation_prompt=True, return_tensors="pt")
    out = model.generate(ids.to("cuda"), max_new_tokens=256, do_sample=False)

The gate decides per prompt; ``gate=False`` in params.json's method ("cast_always") steers every prompt.
"""

from __future__ import annotations

import json
from pathlib import Path


def load(repo_dir: str | Path = ".", *, device: str = "cuda:0", dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lift.repair.cast import make

    repo_dir = Path(repo_dir)
    params = json.loads((repo_dir / "repair" / "params.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(repo_dir)
    model = AutoModelForCausalLM.from_pretrained(repo_dir, torch_dtype=getattr(torch, dtype)).to(device)
    strength = float(params["params"]["strength"])
    wrapped = make(model, repo_dir / "repair" / "cast", strength, gate=params["method"] == "cast")
    return wrapped, tokenizer
