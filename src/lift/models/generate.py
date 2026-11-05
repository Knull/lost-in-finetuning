"""The frozen generation harness. Implemented in P02; frozen once the stock dev eval starts.

After the freeze it only takes bug fixes, and each fix means re-running anything it touched (bump
``HARNESS_VERSION``: it is part of every cache key).

Behaviour:
- Greedy, ``max_new_tokens`` 256, default chat template with the preamble, left padding. The model's
  own generation_config.json (sampling defaults, repetition penalty, ...) is replaced by an explicit
  greedy config, so nothing leaks in from the checkpoint.
- Stops on the model's end-of-turn tokens (``lift.models.load.stop_token_ids``). ``hit_cap`` means no
  end-of-turn token within ``max_new_tokens``. ``n_new_tokens`` counts the content tokens of the
  output (special tokens and the stop token excluded), the count the "fewer than 5 new tokens"
  unclear rule reads; ``meta.n_generated`` counts every generated token for throughput.
- A prompt that would not fit in the context with ``max_new_tokens`` more tokens raises; nothing is
  ever truncated.
- Batches sorted by prompt length (longest first); batch size chosen from free memory unless given (a
  given one must be at least 1). An out-of-memory error halves the batch and retries it (never skips a
  prompt).
- Decoding must stay finite. A row whose scores hold a NaN or +inf before its end-of-turn token (an
  fp16 overflow, typically) decodes garbage from then on (argmax over NaN picks id 0), which would be
  written as its output, often a cap hit cut short. So its batch raises NonFiniteOutputError before
  anything of it is written: a StaleOutputError, so lift generate and run-dev skip the cell and list
  it. A non-finite fp16 run is not reported (decisions item 27): the cell is made in bf16 (after its
  restart line, when its file already holds fp16 records). In bf16 or fp32 it is a fault to report.
- One GenerationRecord per prompt, appended to
  outputs/generations/{variant}/{set}_{lang}_{split}.jsonl as each batch finishes. On restart, keys
  already present are skipped (resumable across sessions); a record for the same pid under a
  different key (stale translation, other dtype, other template) raises instead of mixing.
- One file is one configuration and one stack. ``meta.run_config`` digests the inputs (everything in
  the cache key but the prompt) and never differs within a file. ``meta.stack`` digests torch,
  transformers and the GPU class: a finished file is reused under any stack (a session may resolve a
  newer patch release), but records are never appended under another stack than the file's. A file
  whose records name no ``meta.intervention`` (made before generations recorded it) is neither extended
  nor reused: every read refuses such a cell.
- Every such refusal is a StaleOutputError naming the ``lift restart`` line that starts the cell over
  (``restart_cell``): a job line right after ``lift sync pull``. Moving the file aside by hand does not
  work: sync never deletes, so the job's pull brings the runs repo's copy back. ``keys_changed`` says
  whether the cell made again may have other generation keys (not when only the stack moved, nor when
  the records only predate ``meta.intervention``), so a key list drawn from the cell (P04's) is drawn
  again. A file made in another precision than the run's is
  not stale and is never started over (that would discard it): it raises OtherPrecisionError, whose
  remedy is the file's ``--dtype`` or leaving the cell out.
- ``meta`` carries dtype, GPU name, device, batch size, the provenance (model revision, harness,
  template, the stack, ``decoding``: the decoding settings without the dtype, ``intervention``:
  ``lift.cachekey.intervention_digest`` of the variant's spec, ``adapter_hash``: None for stock,
  ``run_config``) and, for CAST wrappers, ``gate_fired``. The run manifest records the same
  provenance. The judge pipeline reads a cell only if its provenance is this code's harness, the pinned
  revision of Global and the variant's current entry in configs/variants.yaml (which, for a variant that
  names an adapter, must pin its hash: ``artifacts.adapter``).
- Every run writes a manifest (lift.manifest) with prompts/sec and tokens/sec.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from lift import config
from lift.cachekey import generation_key, intervention_digest
from lift.io import append_jsonl, read_jsonl, sha256_obj, sha256_text
from lift.models.load import DTYPES
from lift.models.variants import LoadedVariant
from lift.paths import generations_dir, judgements_dir, metrics_dir, output_dir
from lift.schemas import GenerationRecord, PromptRecord

log = logging.getLogger(__name__)

HARNESS_VERSION = 1
STACK_FIELDS = ("torch", "transformers", "gpu_capability")
ASIDE_DIR = ".aside"       # in paths.output_dir(): what `lift restart` set aside; never synced (lift.sync)


class StaleOutputError(RuntimeError):
    """A cell's file holds records this run must not extend (made from other inputs, under another
    stack, or in another precision: ``OtherPrecisionError``). ``command`` is the ``lift restart`` line
    that starts the cell over (None for a file no such line covers, or that must not be started over);
    the message names it. The judge pipeline's refusals subclass it, so a command can skip the cell and
    go on with the others. ``keys_changed`` is False only when the cell made again has the same
    generation keys (only the stack moved: the key leaves it out); any other cause may change them (a
    fixed prompt text, another configuration), so a key list drawn from the cell (P04's) is drawn again."""

    def __init__(self, message: str, *, command: str | None = None, keys_changed: bool = True) -> None:
        super().__init__(message)
        self.command = command
        self.keys_changed = keys_changed


class OtherPrecisionError(StaleOutputError):
    """A cell's generations were made in another precision than this run's (``dtype``: the file's). Not
    stale, and never started over: a restart would discard the other precision's outputs. Run with
    ``--dtype`` of the file, or leave the cell out (``--sets``/``--langs``). ``command`` is always None."""

    def __init__(self, message: str, *, dtype: str) -> None:
        super().__init__(message, command=None)
        self.dtype = dtype


class NonFiniteOutputError(StaleOutputError):
    """A row's decoding went non-finite: a NaN or +inf score before its end-of-turn token
    (``GreedyOutput.nonfinite``), as an fp16 overflow does. Greedy argmax decodes garbage from then on,
    which would be written as the row's output (often a cap hit cut short), so its batch is not written.
    ``pids`` are the batch's non-finite rows, ``dtype`` the run's precision. A non-finite fp16 run is not
    reported (decisions item 27): the cell is made in bf16, and ``command`` is the cell's ``lift restart
    ... --dtype fp16`` line when its file already holds fp16 records (a bf16 run never extends them: the
    line sets them aside), else None. In bf16 or fp32 it is a fault to report (CLAUDE.md rule 9), never a
    restart: ``command`` is None. The message names the remedy. Never retried like an out-of-memory error;
    a StaleOutputError, so lift generate and run-dev skip the cell, list it apart ("nonfinite") and go on
    with the others."""

    def __init__(self, message: str, *, pids: Sequence[str], dtype: str, command: str | None = None) -> None:
        super().__init__(message, command=command)
        self.pids = list(pids)
        self.dtype = dtype


def generations_path(variant_name: str, set_name: str, lang: str, split: str) -> Path:
    return generations_dir() / variant_name / f"{set_name}_{lang}_{split}.jsonl"


def restart_command(variant: str, set_name: str, lang: str, split: str, *, judge: str | None = None,
                    dtype: str | None = None) -> str:
    """The job line that starts one cell over (``restart_cell``); ``judge`` (a role) limits it to that
    judge's judgements file. ``dtype``, the precision the cell is made in, is named when it is not
    ``generation.dtype``: ``lift restart`` refuses a cell made in another precision than its --dtype."""
    line = f"lift restart --variant {variant} --split {split} --sets {set_name} --langs {lang}"
    line += f" --judge {judge}" if judge else ""
    return line + (f" --dtype {dtype}" if dtype and dtype != config.get("generation.dtype") else "")


def aside_root() -> Path:
    """A new outputs/.aside/<UTC stamp> folder for one ``lift restart`` (suffixed if that one exists)."""
    from datetime import UTC, datetime

    base = output_dir() / ASIDE_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path, n = base, 0
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


class SetAside(NamedTuple):
    """What ``restart_cell`` did with one file of the cell."""

    path: Path                 # the cell's file, now empty
    records: int | None        # records set aside (an AuditKIT record file is one); None: not here
    copy: Path | None          # where they went; None when the file held nothing


def _subdirs(top: Path) -> list[Path]:
    return sorted(p for p in top.iterdir() if p.is_dir()) if top.is_dir() else []


def cell_judgements(variant: str, set_name: str, lang: str, split: str) -> list[Path]:
    """The judgements files of one cell that exist, one per judge dir."""
    name = f"{set_name}_{lang}_{split}.jsonl"
    return [p for root in _subdirs(judgements_dir()) if (p := root / variant / name).is_file()]


def auditkit_records(variant: str, set_name: str, lang: str, split: str, *,
                     judge_dir: str | None = None) -> list[Path]:
    """The AuditKIT records of one cell under every judge dir (or ``judge_dir``'s): {cell}.json,
    {cell}-{dtype}.json and their .full.json forms (lift.audit.auditkit_path names them)."""
    tags = "|".join(DTYPES)
    name = re.compile(re.escape(f"{set_name}_{lang}_{split}") + rf"(?:-(?:{tags}))?(?:\.full)?\.json")
    top = metrics_dir() / "auditkit"
    dirs = [top / judge_dir] if judge_dir is not None else _subdirs(top)
    return [p for d in dirs if (d / variant).is_dir()
            for p in sorted((d / variant).iterdir()) if p.is_file() and name.fullmatch(p.name)]


def restart_cell(variant: str, set_name: str, lang: str, split: str, *, aside: Path,
                 judge_dir: str | None = None) -> list[SetAside]:
    """Start one cell over (``lift restart``): its generations file, every judgements file of it and
    every AuditKIT record of it, or only ``judge_dir``'s judgements file and records, are copied to the
    same path under ``aside`` and left empty.

    Emptied, never removed: sync never deletes, so a removed file comes back from the runs repo at the
    next pull, while an emptied one is a local edit the next push sends (an update), replacing the runs
    repo's copy when that copy has not changed since this machine's last sync; otherwise the push reports
    an emptied-file conflict (a merge would fill it again), which ``lift sync push --force`` settles by
    completing the restart (R1-08). generate and judge_file then fill it again (an empty file has no
    stack). An empty record is retired (readers skip it): a record must not outlive the outputs it
    describes, and the record of another precision's cell is never rewritten. The copy is written before
    the file is emptied, so a kill in between loses nothing."""
    if judge_dir is not None:
        files = [judgements_dir() / judge_dir / variant / f"{set_name}_{lang}_{split}.jsonl"]
    else:
        files = [generations_path(variant, set_name, lang, split),
                 *cell_judgements(variant, set_name, lang, split)]
    files += auditkit_records(variant, set_name, lang, split, judge_dir=judge_dir)
    out = []
    for path in files:
        if not path.is_file():
            out.append(SetAside(path, None, None))
            continue
        data = path.read_bytes()
        copy = None
        if data:
            copy = aside / path.relative_to(output_dir())
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
            path.write_bytes(b"")
        jsonl = path.suffix == ".jsonl"
        n = sum(1 for line in data.splitlines() if line.strip()) if jsonl else int(bool(data))
        out.append(SetAside(path, n, copy))
    return out


def made_in(variant: str, set_name: str, lang: str, split: str) -> set[str]:
    """The precisions (``meta.dtype``) the cell's generations were made in, read line by line (a line
    cut by a kill, or a record without one, is passed over): ``lift restart`` must read any file it
    may set aside, a corrupt one included."""
    path = generations_path(variant, set_name, lang, split)
    found: set[str] = set()
    for line in path.read_bytes().splitlines() if path.is_file() else []:
        try:
            dtype = json.loads(line)["meta"]["dtype"]
        except (ValueError, KeyError, TypeError):
            continue
        if dtype:
            found.add(str(dtype))
    return found


def _cell_restart(out_path: Path, variant: str, dtype: str) -> str | None:
    """The ``lift restart`` line of the cell whose generations file is ``out_path`` (every lift command
    writes there) made in ``dtype``, None for any other file (a test's, Gate 0's)."""
    cell = out_path.stem.rsplit("_", 2)
    if len(cell) != 3 or out_path.resolve() != generations_path(variant, *cell).resolve():
        return None
    return restart_command(variant, *cell, dtype=dtype)


def _start_over(restart: str | None, out_path: Path) -> str:
    if restart is None:
        return f"move {out_path} aside (no lift restart line covers that file)"
    return (f"run `{restart}` as a job line right after `lift sync pull`, followed by the line that failed "
            "(once: it sets the cell's files aside and leaves them empty, so the next push replaces the runs "
            "repo's copies, unless one changed since this machine's last sync: then that push reports an "
            "emptied-file conflict, which `lift sync push --force` settles)")


def gen_config(variant: LoadedVariant, *, max_new_tokens: int | None = None) -> dict[str, Any]:
    """Everything about decoding that can change an output. It goes into every cache key, so bf16
    and fp16 outputs (or a different stop set or cap) never share a key."""
    from lift.models.load import stop_token_ids

    gen = config.get("generation")
    if gen["do_sample"] is not False:
        raise ValueError("generation.do_sample must be false: decoding is greedy (locked Oct 28)")
    return {
        "harness": HARNESS_VERSION,
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens or gen["max_new_tokens"]),
        "dtype": variant.dtype,
        "template": gen["template"],
        "stop_ids": stop_token_ids(variant.tokenizer, variant.hf_model),
    }


def stack(device: str) -> dict[str, str]:
    """The software and GPU class a generation runs on (``STACK_FIELDS``). Neither in the cache key nor
    in ``run_config``, so a library upgrade never re-keys finished cells; ``generate`` keeps one stack
    per file instead (``meta.stack``)."""
    from importlib import metadata

    out = {name: metadata.version(name) for name in ("torch", "transformers")}
    out["gpu_capability"] = "cpu"
    if str(device).startswith("cuda"):
        import torch

        major, minor = torch.cuda.get_device_capability(torch.device(device))
        out["gpu_capability"] = f"{major}.{minor}"
    return out


def stack_digest(env: Mapping[str, Any]) -> str:
    """16 hex of exactly the ``STACK_FIELDS`` of ``env`` (a ``stack()``, or a record's meta)."""
    return sha256_obj({name: env.get(name) for name in STACK_FIELDS})[:16]


def record_stack(meta: Mapping[str, Any]) -> str:
    """A record's ``meta.stack``; older records without one are read by their torch, transformers and
    GPU class."""
    return str(meta.get("stack") or stack_digest(meta))


def decoding_digest(gen_cfg: Mapping[str, Any]) -> str:
    """16 hex of the decoding settings without the dtype (harness, do_sample, max_new_tokens, template,
    stop_ids): the same for a bf16 and an fp16 run, so a precision comparison can check it."""
    return sha256_obj({k: v for k, v in gen_cfg.items() if k != "dtype"})[:16]


def run_config_digest(variant: LoadedVariant, gen_cfg: dict[str, Any]) -> str:
    """The inputs: everything in the cache key except the prompt. A generations file holds one, also
    for prompts outside the current run (a --limit run in another dtype is refused)."""
    from lift.cachekey import intervention_digest

    return sha256_obj({"model_rev": variant.model_rev, "adapter_hash": variant.adapter_hash,
                       "intervention": intervention_digest(variant.spec),
                       "template_hash": variant.template_hash, "gen_cfg": gen_cfg})[:16]


def _was(records: Sequence[GenerationRecord], name: str) -> list[str]:
    return sorted({str(r.meta.get(name)) for r in records})


def _moved_fields(records: Sequence[GenerationRecord], env: Mapping[str, str]) -> list[str]:
    """The ``STACK_FIELDS`` in which ``records`` differ from ``env``."""
    return [name for name in STACK_FIELDS if _was(records, name) != [str(env.get(name))]]


def stack_change(records: Sequence[GenerationRecord], env: Mapping[str, str]) -> str:
    """How the stack of ``records`` differs from ``env`` ("transformers 5.22.1 -> 5.22.2"), or "" when
    every record has this stack."""
    current = stack_digest(env)
    if all(record_stack(r.meta) == current for r in records):
        return ""
    moved = [f"{name} {'/'.join(_was(records, name))} -> {env.get(name)}"
             for name in _moved_fields(records, env)]
    return ", ".join(moved) or f"stack {sorted({record_stack(r.meta) for r in records})} -> {current}"


def _resume_hint(moved: Collection[str]) -> str:
    """Who can still finish a cell begun under another stack, by what moved (decision 32):
    constraints.txt pins transformers, while torch is each runtime's own CUDA build."""
    runtime = bool({"torch", "gpu_capability"} & set(moved))
    if "transformers" in moved and runtime:
        return ("Only the runtime and GPU class that began this cell (torch is the runtime's own CUDA "
                "build), with constraints.txt installed (it pins transformers), can finish it.")
    if "transformers" in moved:
        return ("constraints.txt pins transformers: a runtime installed with it (pip install -c "
                "constraints.txt) can finish this cell when that is the version that began it.")
    if runtime:
        return ("torch is the runtime's own CUDA build, so only the runtime and GPU class that began this "
                "cell can finish it.")
    return "Only the stack that began this cell can finish it."


def record_key(variant: LoadedVariant, prompt: PromptRecord, gen_cfg: dict[str, Any]) -> str:
    return generation_key(
        model_rev=variant.model_rev, adapter_hash=variant.adapter_hash, spec=variant.spec,
        template_hash=variant.template_hash, gen_cfg=gen_cfg, pid=prompt.pid, lang=prompt.lang,
        prompt_text=prompt.text,
    )


def prompt_id(set_name: str, lang: str, pid: str) -> tuple[str, str, str]:
    """A prompt's identity in a generations file. pids are shared across languages by design."""
    return (set_name, lang, pid)


def read_generations(path: Path) -> list[GenerationRecord]:
    """All records of one generations file. Raises on a duplicate key or prompt (a corrupt file)."""
    records = [GenerationRecord.model_validate(row) for row in read_jsonl(path)]
    seen_keys: set[str] = set()
    seen_prompts: set[tuple[str, str, str]] = set()
    for r in records:
        ident = prompt_id(r.set, r.lang, r.pid)
        if r.key in seen_keys or ident in seen_prompts:
            raise ValueError(f"{path}: duplicate record for {r.set}/{r.lang}/{r.pid} (key {r.key[:12]})")
        seen_keys.add(r.key)
        seen_prompts.add(ident)
    return records


def repair_tail(path: Path) -> bool:
    """Make a file killed mid-append readable again: a last line without its newline is completed
    if it is valid JSON, dropped otherwise. Returns True if the file changed.

    Only the partial tail is touched (an append, or an in-place truncate), so a kill during the
    repair itself can never lose a complete record."""
    import json

    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        complete = True
    except (UnicodeDecodeError, json.JSONDecodeError):
        complete = False
    with open(path, "r+b") as f:
        if complete:
            f.seek(0, 2)
            f.write(b"\n")
        else:
            f.truncate(cut)
    log.warning("%s: repaired a truncated last line (%d bytes)", path, len(tail))
    return True


@dataclass(frozen=True)
class GreedyOutput:
    ids: list[int]          # content ids, without the stop token
    stopped: bool           # an end-of-turn token ended it (False: hit the cap)
    n_generated: int        # every generated token, including the stop token
    nonfinite: bool = False  # a step's scores held a NaN or +inf before it stopped (_NonFiniteWatch)


class _NonFiniteWatch:
    """A pass-through logits processor (``LogitsProcessorList`` calls it with each step's ids and
    scores) that marks a row whose scores hold a NaN or +inf at a step where it is still decoding: no
    stop id among its new tokens (past ``width``) yet. A finished row is fed pad and its scores are
    never read, so they are never checked; -inf is left alone, as processors set it on purpose. It keeps
    one bool per row, never the scores.

    It tells the inner decodes of ``model.generate`` apart: a call continues the previous call's decode only
    when its ids are the previous call's with one token appended (same batch size, equal prefix); any other
    call starts a new one, and ``decodes`` records each one's batch size (the previous call's ids are all it
    keeps for this). Inner decodes of another batch size than the ``n_rows`` greedy_batch decodes are a
    wrapper decoding the batch in pieces (P05's CAST, whose per-prompt gate may): which rows a piece holds is
    unknown, so a NaN or +inf anywhere in its scores, a finished row's included, marks every row, and
    generate refuses the batch whole, as it refuses a batch anyway. The pieces must partition the batch, each
    decoded through the processor: ``rows`` refuses pieces whose batch sizes do not add up to ``n_rows``, as
    a piece decoded without it would go unchecked (a decode that never calls it is invisible here, so the
    sizes are what can be checked). Inner decodes of the whole batch are marked row by row."""

    def __init__(self, width: int, stop_ids: Sequence[int], n_rows: int) -> None:
        self.width, self.stop_ids, self.n_rows = width, list(stop_ids), n_rows
        self.marked: Any = None             # bool per row, on the scores' device
        self._stops: Any = None
        self.decodes: list[int] = []        # each inner decode's batch size, in call order
        self._last: Any = None              # the previous call's ids

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        last, self._last = self._last, input_ids
        if not (last is not None and last.device == input_ids.device and last.shape[0] == input_ids.shape[0]
                and last.shape[1] + 1 == input_ids.shape[1] and torch.equal(input_ids[:, :-1], last)):
            self.decodes.append(int(input_ids.shape[0]))   # not the previous call's next step: a new decode
        if scores.shape[0] != self.n_rows:          # a piece of the batch: its rows cannot be told apart
            bad = (torch.isnan(scores) | torch.isposinf(scores)).any().reshape(1).expand(self.n_rows)
        else:
            if self._stops is None:
                self._stops = torch.tensor(self.stop_ids, dtype=input_ids.dtype, device=input_ids.device)
            running = ~torch.isin(input_ids[:, self.width:], self._stops).any(dim=-1)
            bad = (torch.isnan(scores).any(dim=-1) | torch.isposinf(scores).any(dim=-1)) & running
        self.marked = bad.clone() if self.marked is None else self.marked | bad.to(self.marked.device)
        return scores

    def rows(self) -> list[bool]:
        """Whether each row was marked. Raises when generate never called the processor (then nothing
        was checked), or decoded the batch in pieces whose batch sizes do not add up to its rows (then a
        piece may have been decoded without it)."""
        if self.marked is None:
            raise RuntimeError("model.generate never called the logits processor that checks decoding stays "
                               "finite: a wrapper must pass logits_processor on to the model's generate")
        if any(n != self.n_rows for n in self.decodes) and sum(self.decodes) != self.n_rows:
            raise RuntimeError(
                f"model.generate decoded the batch of {self.n_rows} rows in pieces, and the batch sizes of "
                "the inner decodes that called the logits processor checking decoding stays finite, "
                f"{self.decodes}, add up to {sum(self.decodes)}, not {self.n_rows}: a wrapper must pass "
                "logits_processor on to every inner decode, its pieces holding each row once")
        return [bool(x) for x in self.marked.tolist()]


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def greedy_config(*, max_new_tokens: int, stop_ids: Sequence[int], pad_id: int) -> Any:
    """An explicit greedy GenerationConfig.

    Values are spelled out as their neutral settings where they have one: transformers fills every
    None from the model's own generation_config, which ``check_neutral_generation_config`` has
    already checked sets nothing but special-token ids. So a checkpoint's repetition penalty or
    sampling defaults can never reach greedy decoding.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens,
        min_length=0, eos_token_id=list(stop_ids), pad_token_id=pad_id,
        temperature=1.0, top_k=50, top_p=1.0, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0,
        repetition_penalty=1.0, encoder_repetition_penalty=1.0, length_penalty=1.0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0, renormalize_logits=False,
        remove_invalid_values=False, use_cache=True, output_scores=False, return_dict_in_generate=False,
    )


def greedy_batch(model: Any, rows: Sequence[Sequence[int]], *, pad_id: int, stop_ids: Sequence[int],
                 max_new_tokens: int, device: str, gen_config_obj: Any | None = None) -> list[GreedyOutput]:
    """Greedy-decode a batch of token id lists with left padding and an attention mask. Each output says
    whether its row's decoding went non-finite before it stopped (``_NonFiniteWatch``); the caller decides
    what that means (``generate`` refuses the batch). ``model.generate`` (a wrapper's too) must call the
    logits processor it is handed and return one sequence per row. It may decode the batch in pieces, passing
    the processor on to every inner decode, the pieces holding each row once (when their batch sizes do not
    add up to the rows, the wrapper is refused); the check then marks every row of a non-finite batch."""
    import torch
    from transformers import LogitsProcessorList

    if not rows:
        return []
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, width - len(r):] = torch.tensor(list(r), dtype=torch.long)
        mask[i, width - len(r):] = 1
    cfg = gen_config_obj or greedy_config(max_new_tokens=max_new_tokens, stop_ids=stop_ids, pad_id=pad_id)
    watch = _NonFiniteWatch(width, stop_ids, len(rows))
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                             generation_config=cfg, logits_processor=LogitsProcessorList([watch]))
    sequences = out if isinstance(out, torch.Tensor) else out.sequences
    if sequences.shape[0] != len(rows):
        raise RuntimeError(f"model.generate returned {sequences.shape[0]} sequences for the {len(rows)} rows "
                           "of the batch: a wrapper must return one per row, in their order")
    stop = set(stop_ids)
    results = []
    for new, nonfinite in zip(sequences[:, width:].tolist(), watch.rows(), strict=True):
        at = next((j for j, t in enumerate(new) if t in stop), None)
        if at is None:
            if len(new) != max_new_tokens:
                raise RuntimeError(f"generation ended after {len(new)} tokens with no stop token "
                                   f"and below the cap of {max_new_tokens}")
            results.append(GreedyOutput(ids=new, stopped=False, n_generated=len(new), nonfinite=nonfinite))
        else:
            results.append(GreedyOutput(ids=new[:at], stopped=True, n_generated=at + 1, nonfinite=nonfinite))
    return results


def auto_batch_size(variant: LoadedVariant, max_len: int, *, cap: int = 64) -> int:
    """A batch size the KV cache of ``max_len``-token sequences should fit in, from free memory.
    Deliberately rough: an OOM still halves the batch."""
    import torch

    if not str(variant.device).startswith("cuda") or not torch.cuda.is_available():
        return 8
    cfg = variant.hf_model.config
    layers = int(cfg.num_hidden_layers)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    elem = 4 if variant.dtype == "fp32" else 2
    per_seq = 2 * layers * kv_heads * head_dim * elem * max_len          # K and V
    per_seq += 2 * int(cfg.vocab_size) * 4                                # last-step logits, fp32
    per_seq += 8 * int(cfg.hidden_size) * max_len * elem                  # prefill activations
    free, _total = torch.cuda.mem_get_info(torch.device(variant.device))
    return int(max(1, min(cap, (0.6 * free) // per_seq)))


def _device_name(device: str) -> str:
    from lift.models.load import device_name

    return device_name(device)


@dataclass
class _Row:
    prompt: PromptRecord
    key: str
    ids: list[int]


def _check_precision(existing: Sequence[GenerationRecord], dtype: str, out_path: Path) -> None:
    """One file is one precision, and a cell made in another one is not stale: starting it over would
    discard it (and the next push would replace the runs repo's copy). Its remedies are its own
    ``--dtype``, where that precision runs, or leaving it out of this run."""
    made = sorted({str(r.meta["dtype"]) for r in existing if r.meta.get("dtype")} - {dtype})
    if made:
        was = "/".join(made)
        raise OtherPrecisionError(
            f"{out_path} holds generations made in {was}, and this run is {dtype}: one file is one "
            f"precision, so nothing is appended. Run this cell with --dtype {was} where {was} runs, or leave "
            f"it out of this run with --sets/--langs; starting it over would discard its {was} outputs.",
            dtype=was)


def _nonfinite_error(out_path: Path, variant: LoadedVariant, pids: Sequence[str], n_batch: int, *,
                     restart: str | None) -> NonFiniteOutputError:
    """The refusal of a batch whose rows ``pids`` (of ``n_batch``) went non-finite, with the remedy for
    the run's precision. ``restart``: the cell's lift restart line when its file already holds records (in
    the run's precision, which is checked before any batch), else None; it is the error's ``command`` only
    in fp16, whose cell is made in bf16 instead."""
    dtype = variant.dtype
    command = restart if dtype == "fp16" else None
    if dtype == "fp16":
        remedy = ("A non-finite fp16 run is not reported (decisions item 27): make this cell in bf16, on an "
                  "A100 or L4.")
        if command:
            remedy += (f" Its file holds fp16 records, which a bf16 run never extends: first run `{command}` "
                       "as a job line right after `lift sync pull` (it sets them aside), then the line that "
                       "failed without --dtype.")
    else:
        remedy = (f"In {dtype} this is no fp16 overflow: the variant (its weights or intervention) or the "
                  "harness makes NaN or inf. Stop and report it (CLAUDE.md rule 9); never tune around it.")
    return NonFiniteOutputError(
        f"{out_path}: decoding of variant {variant.spec.name} in {dtype} went non-finite (a NaN or +inf "
        f"score before the end-of-turn token) for {len(pids)} of the {n_batch} prompts of a batch, e.g. "
        f"{list(pids[:5])}; nothing of that batch is written: from then on such a row decodes garbage "
        f"(argmax over NaN picks id 0), which would be written as its output. {remedy}",
        pids=pids, dtype=dtype, command=command)


def generate(variant: LoadedVariant, prompts: Sequence[PromptRecord], *, out_path: Path,
             batch_size: int | None = None, resume: bool = True, max_new_tokens: int | None = None,
             manifest: bool = True) -> Path:
    """Generate one record per prompt into ``out_path`` (see the module docstring)."""
    from lift.manifest import finish_run, start_run
    from lift.models.load import check_neutral_generation_config, encode_chat

    if batch_size is not None and batch_size < 1:   # a negative one would decode empty batches for ever
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    out_path = Path(out_path)
    check_neutral_generation_config(variant.hf_model)
    gen_cfg = gen_config(variant, max_new_tokens=max_new_tokens)
    cap = gen_cfg["max_new_tokens"]
    stop_ids = gen_cfg["stop_ids"]
    tok = variant.tokenizer
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer has no pad token; left-padded batches need one")
    idents = [prompt_id(p.set, p.lang, p.pid) for p in prompts]
    if len(set(idents)) != len(idents):
        raise ValueError("duplicate prompts (same set, language and pid) in the prompt list")

    rows = [_Row(p, record_key(variant, p, gen_cfg), encode_chat(tok, p.text)) for p in prompts]
    context = getattr(variant.hf_model.config, "max_position_embeddings", None)
    if context:
        too_long = [r.prompt.pid for r in rows if len(r.ids) + cap > context]
        if too_long:
            raise ValueError(f"{len(too_long)} prompts plus {cap} new tokens exceed the {context}-token "
                             f"context, e.g. {too_long[:5]}; nothing is truncated")

    env = stack(variant.device)
    run_config = run_config_digest(variant, gen_cfg)
    restart = _cell_restart(out_path, variant.spec.name, variant.dtype)
    existing: list[GenerationRecord] = []
    if out_path.exists():
        if not resume:
            raise FileExistsError(f"{out_path} exists; pass resume=True or move it aside")
        repair_tail(out_path)
        existing = read_generations(out_path)
        _check_precision(existing, variant.dtype, out_path)
        other = sorted({r.pid for r in existing if r.meta.get("run_config") != run_config})
        if any(r.variant != variant.spec.name for r in existing) or other:
            raise StaleOutputError(
                f"{out_path} holds records made under another configuration (variant, template, model, "
                f"adapter, decoding settings or harness version), e.g. {other[:5]}; one file is one "
                "configuration and outputs are never mixed. Run with the file's configuration, or start the "
                f"cell over: {_start_over(restart, out_path)}.", command=restart)
        unnamed = sorted({r.pid for r in existing if not r.meta.get("intervention")})
        if unnamed:                         # made before generations recorded it (R1-03)
            raise StaleOutputError(
                f"{out_path}: {len(unnamed)} records name no intervention (meta.intervention), e.g. "
                f"{unnamed[:5]}: they were made before generations recorded it, and every read refuses such "
                "a cell (lift.judge.pipeline), so it is neither extended nor reused. Start the cell over: "
                f"{_start_over(restart, out_path)}.", command=restart, keys_changed=False)
    done = {prompt_id(r.set, r.lang, r.pid): r.key for r in existing}

    def ident(row: _Row) -> tuple[str, str, str]:
        return prompt_id(row.prompt.set, row.prompt.lang, row.prompt.pid)

    stale = [row.prompt.pid for row in rows if ident(row) in done and done[ident(row)] != row.key]
    if stale:
        raise StaleOutputError(
            f"{out_path}: {len(stale)} records were made from other inputs (a fixed prompt text, or another "
            f"cache-key version), e.g. {stale[:5]}; outputs are never mixed. Start the cell over: "
            f"{_start_over(restart, out_path)}.", command=restart)
    todo = sorted((row for row in rows if ident(row) not in done),
                  key=lambda row: (-len(row.ids), row.prompt.lang, row.prompt.pid))
    moved = stack_change(existing, env)
    if moved and todo:
        raise StaleOutputError(
            f"{out_path} was generated under another stack ({moved}); appending {len(todo)} records under "
            f"this one would mix stacks in one file. {_resume_hint(_moved_fields(existing, env))} Otherwise "
            f"start the cell over: {_start_over(restart, out_path)}. Finished cells are reused under any "
            "stack.", command=restart, keys_changed=False)

    gpu = _device_name(variant.device)
    provenance = {"model_rev": variant.model_rev, "harness": HARNESS_VERSION,
                  "template": variant.template_hash[:16], **env, "stack": stack_digest(env),
                  "decoding": decoding_digest(gen_cfg), "intervention": intervention_digest(variant.spec),
                  "adapter_hash": variant.adapter_hash}
    run = start_run("generate", variant=variant.spec.name, out_path=str(out_path), dtype=variant.dtype,
                    device=variant.device, gpu=gpu, gen_cfg=gen_cfg, n_prompts=len(rows),
                    n_resumed=len(rows) - len(todo), run_config=run_config,
                    template_hash=variant.template_hash, **provenance) if manifest else None
    if not todo:
        if moved:
            log.warning("%s: all %d prompts already generated, under another stack (%s); reused as they are",
                        out_path.name, len(rows), moved)
        else:
            log.info("%s: all %d prompts already generated", out_path.name, len(rows))
        if run is not None:
            finish_run(run, status="ok", n_written=0)
        return out_path

    size = batch_size or auto_batch_size(variant, max(len(r.ids) for r in todo) + cap)
    log.info("%s: generating %d prompts (%d resumed), batch size %d, %s on %s",
             out_path.name, len(todo), len(rows) - len(todo), size, variant.dtype, gpu)
    stats = {"n_written": 0, "n_generated_tokens": 0, "cap_hits": 0, "oom_retries": 0, "seconds": 0.0,
             "batch_sizes": set()}
    gcfg = greedy_config(max_new_tokens=cap, stop_ids=stop_ids, pad_id=pad_id)

    def run_batch(batch: list[_Row]) -> None:
        nonlocal size
        oom = False
        try:
            t0 = time.perf_counter()
            outs = greedy_batch(variant.model, [r.ids for r in batch], pad_id=pad_id, stop_ids=stop_ids,
                                max_new_tokens=cap, device=variant.device, gen_config_obj=gcfg)
            seconds = time.perf_counter() - t0
        except Exception as exc:  # only an out-of-memory error is retried; anything else re-raises
            if not is_oom(exc) or len(batch) == 1:
                raise
            oom = True
        if oom:
            # Retry outside the except clause: its traceback would keep the failed batch's tensors
            # (KV cache, logits) alive, and the halves could run out of memory again.
            _free_memory()
            stats["oom_retries"] += 1
            half = len(batch) // 2
            size = max(1, min(size, half))
            log.warning("out of memory at batch size %d; retrying as %d + %d", len(batch), half,
                        len(batch) - half)
            run_batch(batch[:half])
            run_batch(batch[half:])
            return
        nonfinite = [r.prompt.pid for r, o in zip(batch, outs, strict=True) if o.nonfinite]
        if nonfinite:                       # after the try, so never retried as an out-of-memory error
            held = bool(existing) or stats["n_written"] > 0         # records of this cell in the file
            raise _nonfinite_error(out_path, variant, nonfinite, len(batch),
                                   restart=restart if held else None)
        extra = variant.batch_meta() if variant.batch_meta is not None else [{}] * len(batch)
        records = []
        for r, o, more in zip(batch, outs, extra, strict=True):
            meta = {**more, **provenance, "run_config": run_config}
            rec = _record(variant, r, o, seconds / len(batch), len(batch), gpu, tok, meta)
            records.append(rec.model_dump())
            stats["n_generated_tokens"] += o.n_generated
            stats["cap_hits"] += int(not o.stopped)
        append_jsonl(out_path, records)
        stats["n_written"] += len(records)
        stats["seconds"] += seconds
        stats["batch_sizes"].add(len(batch))

    def summary() -> dict[str, Any]:
        secs = stats["seconds"] or 1e-9
        return {
            "n_written": stats["n_written"], "cap_hits": stats["cap_hits"],
            "oom_retries": stats["oom_retries"],
            "batch_sizes": sorted(stats["batch_sizes"]), "generate_seconds": round(stats["seconds"], 3),
            "prompts_per_sec": round(stats["n_written"] / secs, 4),
            "tokens_per_sec": round(stats["n_generated_tokens"] / secs, 2),
            "n_generated_tokens": stats["n_generated_tokens"], "devices": 1,
        }

    i = 0
    try:
        while i < len(todo):
            batch = todo[i: i + size]
            run_batch(batch)
            i += len(batch)
            log.info("%s: %d/%d", out_path.name, stats["n_written"], len(todo))
    except BaseException as exc:
        # A crashed run still records the GPU time it spent (the cost section reads manifests).
        if run is not None:
            finish_run(run, status="failed", error=type(exc).__name__, **summary())
        raise
    done_stats = summary()
    log.info("%s: wrote %d records, %.2f prompts/s, %.1f tokens/s, %d hit the cap", out_path.name,
             done_stats["n_written"], done_stats["prompts_per_sec"], done_stats["tokens_per_sec"],
             done_stats["cap_hits"])
    if run is not None:
        finish_run(run, status="ok", **done_stats)
    return out_path


def _record(variant: LoadedVariant, row: _Row, out: GreedyOutput, seconds: float, batch_size: int,
            gpu: str, tok: Any, more: dict) -> GenerationRecord:
    from lift.models.load import special_token_ids

    special = special_token_ids(tok)
    return GenerationRecord(
        key=row.key, variant=variant.spec.name, pid=row.prompt.pid, lang=row.prompt.lang,
        set=row.prompt.set, split=row.prompt.split, prompt_sha=sha256_text(row.prompt.text),
        output=tok.decode(out.ids, skip_special_tokens=True),
        n_prompt_tokens=len(row.ids), n_new_tokens=sum(t not in special for t in out.ids),
        hit_cap=not out.stopped, seconds=round(seconds, 4),
        meta={"dtype": variant.dtype, "gpu": gpu, "device": variant.device, "batch_size": batch_size,
              "n_generated": out.n_generated, **more},
    )


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def greedy_token_ids(model: Any, tokenizer: Any, texts: Sequence[str], *, device: str,
                     max_new_tokens: int | None = None, batch_size: int = 4) -> list[GreedyOutput]:
    """Greedy outputs for raw user texts, through the same template, stop set and decoder as
    ``generate`` but without records (merge parity checks use it)."""
    from lift.models.load import check_neutral_generation_config, encode_chat, stop_token_ids

    check_neutral_generation_config(model)
    cap = int(max_new_tokens or config.get("generation.max_new_tokens"))
    stop_ids = stop_token_ids(tokenizer, model)
    ids = [encode_chat(tokenizer, t) for t in texts]
    out: list[GreedyOutput] = []
    for i in range(0, len(ids), batch_size):
        out.extend(greedy_batch(model, ids[i: i + batch_size], pad_id=tokenizer.pad_token_id,
                                stop_ids=stop_ids, max_new_tokens=cap, device=device))
    return out
