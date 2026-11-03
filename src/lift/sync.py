"""Move outputs between Kaggle sessions and the laptop through a PRIVATE HF dataset repo
(``LIFT_RUNS_REPO``). Implemented in P02.

``push``: upload outputs/ (generations, judgements, manifests, adapters, artifacts, metrics) with
``huggingface_hub.upload_folder``; ``pull``: ``snapshot_download`` into outputs/. The repo must be
private: it holds raw harmful completions and unrepaired adapters. Refuse to run if the repo is
public (check ``model_info``/``dataset_info`` ``private`` flag first).
"""

from __future__ import annotations


def push(subdirs: tuple[str, ...] = ()) -> None:
    raise NotImplementedError("P02")


def pull(subdirs: tuple[str, ...] = ()) -> None:
    raise NotImplementedError("P02")
