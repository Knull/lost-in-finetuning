"""Hugging Face artifacts, article, notebook helpers and the public-repo preflight (P08).

Every number a public artifact shows is read from a file under ``outputs/metrics`` (or a config, for
the pre-registered design numbers; or a ``lift train`` manifest under ``outputs/manifests``, for the rows
an adapter was trained on) through ``lift.publish.inputs.Facts``, which records where each one
came from. An input that does not exist yet renders as a ``[PENDING: ...]`` placeholder that names
the file and the stage that writes it, and an upload is refused while any staged text still holds one.

- ``settings``     configs/publish.yaml: namespace, repo ids, access, collections, team
- ``inputs``       the files P08 reads (selection.json from P05, final_report.json from P07), Facts
- ``render``       the template filler (``{{ name }}``) and markdown tables
- ``hub``          upload plans: dry run by default, ``--yes`` uploads (gating set before any file goes up)
- ``model``        the repaired model, the repair parameters and the gated adapters, with their cards
- ``datasets``     the public logs dataset, the gated raw completions, the translated prompt sets and the
                   training sets with CuratorKIT's files (docs/decisions.md item 118)
- ``collect``      the two HF collections
- ``article``      hf/article.md from hf/templates/article.md
- ``preflight``    secrets, tokens and raw completions in tracked files and git history
- ``acceptance``   helpers for notebooks/acceptance_test.ipynb

Nothing here imports huggingface_hub, torch or transformers at module import.
"""

from __future__ import annotations


class PublishError(RuntimeError):
    """A problem the person publishing has to fix. The CLI prints it without a traceback."""
