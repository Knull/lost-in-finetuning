"""Machine translation of the English prompts, and NLLB back-translation. Implemented in P01.

What gets translated, always from the English source (never from another translation), with Google
Translate through deep-translator, the way PolyRefuse itself was built:
- hi, 1,554 strings: PolyRefuse harmful test 572 + train 246 + val 36, harmless test 500 + harmless
  train 200 (PolyRefuse has no Hindi at all). lift.data.build first drops, in every language, the
  17 harmful train/val prompts that copy or nearly copy a test prompt (of 260 + 39).
- zh, 700 strings: harmless test 500 + harmless train 200. PolyRefuse has no zh harmless test, and
  its zh harmless train file is a different, misaligned draw (docs/decisions.md, Nov 3).
- The OR-Bench-Hard sample, 400 strings, into each of hi zh es ar th (2,000).
That is 4,254 strings and about 4,244 Google calls: a repeated source is one call (the harmful
test repeats 10 prompts). No English source comes near the length limit (the longest is 842 characters). At
about one request a second (google_backend spaces request starts by 1 s plus up to 0.5 s of
jitter) a full run takes roughly 1.5 to 2 hours, and a stopped run resumes from the cache.

Cache: interim/translations/{engine}/{lang}.jsonl, one row per engine call keyed by (engine, lang,
sha256(source)), appended and flushed row by row, so a killed run loses at most the item in flight.
Later rows for a key win (--retranslate-failed appends), and a truncated last line is ignored.
Records take TranslationInfo.date from the cache row, so a rebuild is byte-identical.

Failure rules, applied to every output, cached or new, in this order. All are collected and raised
together as one TranslationError after every item was tried; a failed item never falls back to
English, it is listed (and goes to the native readers' failures sheet):
    empty       None or blank. deep-translator returns None when Google echoes a source that has
                no letters or digits.
    identical   the source again, ignoring case and whitespace. deep-translator silently returns the
                stripped source when Google echoes it.
    wrong_lang  GlotLID does not assign it to the target (detail "top=<label> <prob>"). On
                PolyRefuse's own Google output this rule fails 15/572 and 11/500 es, 0/572 zh
                (24 before lift.lang spaced Chinese ideographs, Nov 4), 0/572 and 6/500 ar,
                0/572 and 1/500 th (short strings, mostly genuine): still failures, for the
                readers to accept or fix.
    too_long    4,999 characters or more (deep-translator refuses 5,000): never sent.
    error       every retry raised a transient error (detail: exception types and counts, e.g.
                "ConnectionError x5"), or, with cache_only, the item is not cached ("not cached").

Errors (is_transient): only a transient error is retried, 4 times with backoff 5, 10, 20, 40 s (+ up
to 10%): an OSError (requests' RequestException and its ConnectionError, Timeout, SSLError; socket
timeouts; the builtin ConnectionError and TimeoutError) that is not also a ValueError or TypeError
(requests' MissingSchema, InvalidURL, StreamConsumedError: bugs), or deep-translator's RequestError,
ServerException or TranslationNotFound. MAX_CONSECUTIVE_ERRORS (10) items in a row failing every
retry trip a circuit breaker: TranslationBlocked kind "streak", a network or service outage, not an
IP block. Everything else propagates at once, unretried and uncounted, with its traceback (and its
own type, except deep-translator's BaseError family, below): a missing package (ImportError:
deep-translator, or torch for the NLLB engine) or a bug stops the run with its real error, never a
breaker trip that reads like a network outage.

Google (deep-translator 1.11.4, verified from its source): it scrapes translate.google.com/m with
requests.get and NO timeout, so google_backend patches one into deep_translator.google only. Our code
"zh" is Google's "zh-CN" (configs/project.yaml languages.google_translate). GoogleTranslator.translate
raises TooManyRequests on HTTP 429, RequestError on any other non-2xx status (a retry can clear a
5xx, not a 4xx, and RequestError does not carry the status), TranslationNotFound when a 2xx page
holds no result div (a glitch, or a consent or changed page, which the breaker stops),
NotValidPayload or NotValidLength for a non-string or a 5,000-character input (never sent: see
too_long), and lets requests' exceptions through. It never raises ServerException (deep-translator's
other engines do, for an API server error by HTTP status). HTTP 429 (Google's /sorry CAPTCHA page,
no Retry-After) raises TranslationBlocked kind "http_429" at once with no retry: a blocked call
costs ~3 s and extends the block. Then wait 30 minutes or more and rerun; never solve the CAPTCHA,
rotate IPs or user agents, or run two jobs on one IP. str(e) of deep-translator's BaseError family
(TranslationNotFound, NotValidPayload, NotValidLength, ...) starts with the input text, so only
exception type names are logged or kept in details, and such an error that is not transient is
re-raised as a RuntimeError naming its type, with its traceback but without its message. Logs
carry counts, indices, pids, rules and labels, never text.

NLLB-200 distilled 600M (transformers, imported on use, CPU): back_translate turns ar/th generated
outputs into English for the judge sensitivity sample (P07); they are raw model completions, so its
cache defaults to outputs/backtranslations/{engine}/{x}-en.jsonl (outputs/ is gitignored and goes
only to the private runs repo: CLAUDE.md rule 6), never the data tree that P02 syncs as data and P08
publishes. nllb_backend is an opt-in en -> X fallback for OR-Bench items Google failed (lossy for
zh: 77 of the Han characters in PolyRefuse's zh harmful test are <unk> to it). check_nllb() raises
ImportError with an install hint unless torch and transformers import, for a caller to run before
any work. transformers 5.22's cached_files raises OSError for a model it cannot fetch (offline and
not cached, gated, an unknown repo or revision, a Hub HTTP error); _Nllb re-raises it as
RuntimeError, so it stops the run at once instead of being retried as a network error.
"""

from __future__ import annotations

import copy
import functools
import importlib
import json
import logging
import os
import random
import re
import sys
import time
import types
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lift import config
from lift.data.text import clean_text, dup_key
from lift.io import append_jsonl, sha256_obj, sha256_text
from lift.lang import LangIdentifier, lid_info
from lift.paths import interim_dir, output_dir
from lift.schemas import PromptRecord, TranslationInfo

log = logging.getLogger(__name__)

ENGINE_NAMES = {"google": "google", "nllb": "nllb-200-distilled-600M"}   # TranslationInfo.engine values
MAX_CHARS = 4999               # sources this long are "too_long" (deep-translator refuses 5,000)
MAX_CONSECUTIVE_ERRORS = 10    # circuit breaker: items in a row whose every retry raised
RETRY_JITTER = 0.1             # each backoff delay gets up to 10% more, at random
BACKTRANSLATION_DIRNAME = "backtranslations"   # back_translate's cache: output_dir() / this

# deep-translator 1.11.4 errors a retry can cure (deep_translator.exceptions; see is_transient)
_DT_TRANSIENT = ("RequestError", "ServerException", "TranslationNotFound")

# (english_text, our lang code) -> output, which may be None; raises on errors (is_transient ones
# are retried)
Backend = Callable[[str, str], str | None]
Rule = Literal["empty", "identical", "wrong_lang", "error", "too_long"]
BlockKind = Literal["http_429", "streak"]

_ROW_FIELDS = ("key", "engine", "lang", "source_sha", "source", "output", "date")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_rng = random.Random()


class TranslationFailure(BaseModel):
    """One failed item. No field holds text except ``output`` (the engine's output when there was
    one, for the native readers), which is kept out of repr so a logged failure cannot leak it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    pid: str | None = None
    lang: str
    engine: str
    source_sha: str
    rule: Rule
    detail: str = ""                 # e.g. "top=mar_Deva 0.62" or "ConnectionError x5"; never text
    output: str | None = Field(default=None, repr=False)


class TranslationError(RuntimeError):
    """Some items failed. Raised after every item was attempted; the message holds counts only.

    ``outputs[i]`` is None for a failed item (never the source); ``records`` holds translate_records'
    successful records, in input order.
    """

    def __init__(self, failures: Sequence[TranslationFailure], outputs: Sequence[str | None],
                 records: Sequence[PromptRecord] = ()) -> None:
        self.failures = list(failures)
        self.outputs = list(outputs)
        self.records = list(records)
        by_rule = Counter(f.rule for f in self.failures)
        where = f" into {self.failures[0].lang} ({self.failures[0].engine})" if self.failures else ""
        counts = ", ".join(f"{rule} {n}" for rule, n in sorted(by_rule.items()))
        super().__init__(f"{len(self.failures)} of {len(self.outputs)} translations{where} failed: {counts}")


class TranslationBlocked(RuntimeError):
    """Translation stops for this run, at once. ``done`` items were handled before the stop and
    their outputs are cached: rerun later to resume.

    ``kind`` says why. "http_429": Google answered HTTP 429 (its /sorry CAPTCHA page), so this IP is
    blocked; wait 30 minutes or more. "streak": MAX_CONSECUTIVE_ERRORS items in a row failed every
    retry with transient errors (is_transient), so the network or the service is down; not an IP
    block, rerun once it is back. The message holds counts and exception type names, never text.
    """

    KINDS: tuple[str, ...] = ("http_429", "streak")

    def __init__(self, message: str = "translation blocked", *, done: int = 0,
                 kind: BlockKind = "http_429") -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown TranslationBlocked kind {kind!r}; one of {self.KINDS}")
        self.done = done
        self.kind: BlockKind = kind
        super().__init__(message)


class BackTranslationError(RuntimeError):
    """Some back-translations were empty or identical to their input. The message lists indices and
    rules; ``outputs[i]`` is None for a failed item."""

    def __init__(self, failures: Sequence[TranslationFailure], outputs: Sequence[str | None]) -> None:
        self.failures = list(failures)
        self.outputs = list(outputs)
        items = ", ".join(f"{f.index} {f.rule}" for f in self.failures[:20])
        more = f" and {len(self.failures) - 20} more" if len(self.failures) > 20 else ""
        lang = self.failures[0].lang if self.failures else "?"
        super().__init__(f"{len(self.failures)} of {len(self.outputs)} back-translations ({lang}) failed: "
                         f"{items}{more}")


def _blank(output: Any) -> bool:
    return not isinstance(output, str) or not clean_text(output)


def _now_day() -> str:
    return datetime.now(UTC).date().isoformat()


class TranslationCache:
    """Engine outputs on disk, append-only: ``root/{engine}/{lang}.jsonl``, rows {key, engine, lang,
    source_sha, source, output, date}, key = sha256_obj([engine, lang, sha256(source)]).

    Each file is read once, on first use. Later rows for a key win. A line a killed run left half
    written is skipped (and closed off before the next append, so the new row stays whole); any
    other malformed row raises.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else interim_dir() / "translations"
        self._rows: dict[tuple[str, str], dict[str, dict]] = {}
        self._half_line: set[tuple[str, str]] = set()     # files whose last line has no newline

    @staticmethod
    def key(engine: str, lang: str, source: str) -> str:
        return sha256_obj([engine, lang, sha256_text(source)])

    def path(self, engine: str, lang: str) -> Path:
        for part in (engine, lang):
            if not _SAFE_NAME.fullmatch(part):
                raise ValueError(f"not a safe cache file name part: {part!r}")
        return self.root / engine / f"{lang}.jsonl"

    def get(self, engine: str, lang: str, source: str) -> dict | None:
        row = self._load(engine, lang).get(self.key(engine, lang, source))
        return dict(row) if row is not None else None

    def put(self, engine: str, lang: str, source: str, output: str) -> dict:
        """Append one row (flushed) and return it. Only non-blank outputs are cached."""
        if _blank(output):
            raise ValueError("only non-blank outputs are cached")
        rows = self._load(engine, lang)
        row = {"key": self.key(engine, lang, source), "engine": engine, "lang": lang,
               "source_sha": sha256_text(source), "source": source, "output": output, "date": _now_day()}
        path = self.path(engine, lang)
        if (engine, lang) in self._half_line:             # end the cut line, so the new row stays whole
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n")
            self._half_line.discard((engine, lang))
        append_jsonl(path, [row])
        rows[row["key"]] = row
        return dict(row)

    def _load(self, engine: str, lang: str) -> dict[str, dict]:
        if (engine, lang) in self._rows:
            return self._rows[(engine, lang)]
        path = self.path(engine, lang)
        rows: dict[str, dict] = {}
        if path.exists():
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                self._half_line.add((engine, lang))
            lines = data.split(b"\n")
            for lineno, raw in enumerate(lines, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if lineno == len(lines):
                        log.info("%s/%s: ignoring a truncated last line", engine, path.name)
                    else:
                        log.warning("%s/%s:%d: skipping a half-written line", engine, path.name, lineno)
                    continue
                problem = _row_problem(row, engine, lang)
                if problem:
                    raise ValueError(f"{path}:{lineno}: {problem}")
                rows[row["key"]] = row
        self._rows[(engine, lang)] = rows
        return rows


def _row_problem(row: Any, engine: str, lang: str) -> str | None:
    if not isinstance(row, dict):
        return "not a JSON object"
    missing = [k for k in _ROW_FIELDS if k not in row]
    if missing:
        return f"missing fields {missing}"
    if row["engine"] != engine or row["lang"] != lang:
        return f"a {row['engine']}/{row['lang']} row in the {engine}/{lang} file"
    if not isinstance(row["source"], str) or row["key"] != TranslationCache.key(engine, lang, row["source"]):
        return "key does not match the source"
    if _blank(row["output"]):
        return "blank or non-text output"
    return None


class _LazyLID:
    """GlotLID loaded on first use, so a run that has nothing to check never loads it."""

    def __init__(self, cfg: Mapping[str, Any] | None) -> None:
        self._cfg = cfg
        self._lid: LangIdentifier | None = None

    def _get(self) -> LangIdentifier:
        if self._lid is None:
            from lift.lang import LangID

            self._lid = LangID(cfg=self._cfg)
        return self._lid

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        return self._get().top(text, k)

    def is_lang(self, text: str, lang: str) -> bool:
        return self._get().is_lang(text, lang)


def _check_output(source: str, output: str | None) -> tuple[Rule, str] | None:
    if output is None:
        return "empty", "returned None"
    if not isinstance(output, str):
        return "empty", f"returned {type(output).__name__}"
    if _blank(output):
        return "empty", "blank"
    if dup_key(output) == dup_key(source):
        return "identical", ""
    return None


def validate(source: str, output: str | None, lang: str, lid: LangIdentifier | None,
             cfg: Mapping[str, Any] | None = None) -> tuple[Rule, str] | None:
    """The failure rules, in order; None when the output is fine.

    empty (None or blank) -> identical (same ``dup_key`` as the source: case and whitespace
    ignored) -> wrong_lang (not ``lid.is_lang(output, lang)``; detail "top=<label> <prob>").
    ``lid`` None loads GlotLID with ``cfg``, and only if the check gets that far.
    """
    verdict = _check_output(source, output)
    if verdict is not None:
        return verdict
    lid = lid if lid is not None else _LazyLID(cfg)
    if not lid.is_lang(output, lang):
        info = lid_info(lid, output)
        return "wrong_lang", f"top={info.label} {info.prob}"
    return None


class _Google:
    """deep-translator's GoogleTranslator, one instance per language (it is not thread-safe), one
    request at a time, request starts at least ``min_interval`` + U(0, ``jitter``) seconds apart."""

    def __init__(self, codes: Mapping[str, str], *, min_interval: float, jitter: float,
                 timeout: tuple[float, float], sleep: Callable[[float], Any],
                 clock: Callable[[], float]) -> None:
        import deep_translator.google as dtg
        import requests
        from deep_translator.exceptions import TooManyRequests

        # deep-translator calls requests.get with no timeout, so a stalled socket hangs forever.
        # Patch one in for its module only; the global requests.get stays as it is.
        dtg.requests = types.SimpleNamespace(get=functools.partial(requests.get, timeout=timeout))
        self._dtg = dtg
        self._too_many = TooManyRequests
        self.codes = dict(codes)
        self.translators: dict[str, Any] = {}
        self.min_interval, self.jitter = float(min_interval), float(jitter)
        self._sleep, self._clock = sleep, clock
        self._rng = random.Random()
        self._next: float | None = None

    def __call__(self, text: str, lang: str) -> str | None:
        translator = self.translators.get(lang)
        if translator is None:
            if lang not in self.codes:
                raise ValueError(f"no Google Translate code for {lang!r} in languages.google_translate")
            translator = self.translators[lang] = self._dtg.GoogleTranslator(source="en",
                                                                             target=self.codes[lang])
        if self._next is not None:
            wait = self._next - self._clock()
            if wait > 0:
                self._sleep(wait)
        gap = self.min_interval + (self._rng.uniform(0, self.jitter) if self.jitter > 0 else 0.0)
        self._next = self._clock() + gap
        try:
            return translator.translate(text)
        except self._too_many:
            raise TranslationBlocked("Google Translate answered HTTP 429 (its /sorry CAPTCHA page): this "
                                     "IP is blocked. Stop, wait 30 minutes or more, then rerun; the "
                                     "cache keeps everything done", kind="http_429") from None


def google_backend(cfg: Mapping[str, Any] | None = None, *, min_interval: float = 1.0,
                   jitter: float = 0.5, timeout: tuple[float, float] = (10, 30),
                   sleep: Callable[[float], Any] = time.sleep,
                   clock: Callable[[], float] = time.monotonic) -> Backend:
    """Google Translate through deep-translator (imported here: ImportError if it is missing).
    Returns the raw output, which may be None or the echoed source (validate catches both). HTTP 429
    raises TranslationBlocked kind "http_429" at once, never retried. Every other error propagates:
    translate_texts retries the transient ones (is_transient) and stops on the rest."""
    cfg = cfg if cfg is not None else config.load("project")
    return _Google(cfg["languages"]["google_translate"], min_interval=min_interval, jitter=jitter,
                   timeout=timeout, sleep=sleep, clock=clock)


def check_nllb() -> None:
    """Raise ImportError with an install hint unless torch and transformers import, with the two
    classes the NLLB engine loads. Cheap once they are imported; nothing is downloaded.

    The NLLB engine (``engine="nllb"``, --fallback-engine nllb, back_translate) needs both, and the
    laptop venv has neither. Call it before any work, so a missing package stops the run at once;
    _Nllb calls it again on its first load, so the error is the same either way. An OSError while
    importing (a torch DLL that fails to load on Windows) becomes the same ImportError, never an
    error the retry loop would take for a network failure.
    """
    for name, attrs in (("torch", ()), ("transformers", ("AutoModelForSeq2SeqLM", "AutoTokenizer"))):
        try:
            module = importlib.import_module(name)
            for attr in attrs:                  # transformers imports these on first access
                getattr(module, attr)
        except Exception as e:
            raise ImportError(f"--fallback-engine nllb needs torch and transformers, and {name} does not "
                              f"import ({type(e).__name__}). Install both at the versions of "
                              "pyproject.toml's gpu extra (pip install -e \".[gpu]\" brings everything)",
                              name=name) from e


class _Nllb:
    """NLLB-200 distilled 600M (cfg back_translation.engine and revision), loaded on first use.

    transformers 5.x: the source language is set with the ``tok.src_lang`` setter (``src_lang=`` in
    the tokenizer call is silently ignored), the target with ``forced_bos_token_id`` (there is no
    ``lang_code_to_id`` any more); greedy; inputs truncated at 512 tokens. The repo ships only a
    .bin, so the background safetensors conversion is switched off. Loading runs check_nllb first;
    an OSError from transformers (offline and not cached, gated, an unknown revision) is re-raised
    as RuntimeError (the OSError as its cause), so the retry loop never takes it for a network blip.
    """

    def __init__(self, cfg: Mapping[str, Any], *, device: str = "cpu", batch_size: int = 16) -> None:
        bt = cfg["back_translation"]
        self.repo, self.revision = bt["engine"], bt["revision"]
        if self.repo.rsplit("/", 1)[-1] != ENGINE_NAMES["nllb"]:
            raise ValueError(f"back_translation.engine {self.repo!r} is not {ENGINE_NAMES['nllb']}")
        self.device, self.batch_size = device, int(batch_size)
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            check_nllb()
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            try:
                tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.repo, revision=self.revision,
                                                              dtype=torch.float32)
            except OSError as e:
                raise RuntimeError(f"cannot load {self.repo} at revision {self.revision} "
                                   f"({type(e).__name__}, the cause above)") from e
            self._tok, self._model = tok, model.to(self.device).eval()
        return self._tok, self._model

    def __call__(self, texts: Sequence[str], src: str, tgt: str, *, factor: float = 1.5) -> list[str]:
        """Translate ``texts`` from NLLB code ``src`` to ``tgt``, longest first in batches;
        max_new_tokens = int(factor * input length) + 10."""
        tok, model = self._load()               # first: check_nllb's ImportError if torch is missing
        import torch

        tok.src_lang = src
        bos = tok.convert_tokens_to_ids(tgt)
        if tok.convert_tokens_to_ids(src) == tok.unk_token_id or bos == tok.unk_token_id:
            raise ValueError(f"unknown NLLB language code {src!r} or {tgt!r}")
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out: list[str] = [""] * len(texts)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                          max_length=512).to(self.device)
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=1, do_sample=False,
                                     max_new_tokens=int(factor * enc["input_ids"].shape[1]) + 10)
                for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                    out[i] = text
        return out


def nllb_backend(*, cfg: Mapping[str, Any] | None = None, device: str = "cpu",
                 batch_size: int = 16) -> Backend:
    """NLLB-200 as an en -> X Backend: the opt-in fallback (``engine="nllb"``), loaded on the first
    call (check_nllb's ImportError, or RuntimeError for a model that does not load, both raised at
    once by translate_texts). Target text takes more tokens than English, so max_new_tokens =
    2 x input + 10."""
    cfg = cfg if cfg is not None else config.load("project")
    codes = dict(cfg["languages"]["belebele"])
    model = _Nllb(cfg, device=device, batch_size=batch_size)

    def backend(text: str, lang: str) -> str | None:
        if lang == "en" or lang not in codes:
            raise ValueError(f"no NLLB code for {lang!r} in languages.belebele")
        return model([text], codes["en"], codes[lang], factor=2.0)[0]

    return backend


def _nllb_to_english(cfg: Mapping[str, Any], batch_size: int) -> Callable[[list[str], str], list[str]]:
    """back_translate's default translator: NLLB X -> eng_Latn on CPU, max_new_tokens 1.5 x input + 10."""
    codes = cfg["languages"]["belebele"]
    model = _Nllb(cfg, device="cpu", batch_size=batch_size)

    def translator(texts: list[str], lang: str) -> list[str]:
        return model(texts, codes[lang], codes["en"], factor=1.5)

    return translator


@dataclass
class _Outcome:
    output: str | None
    date: str | None
    verdict: tuple[Rule, str] | None


@dataclass
class _Run:
    outputs: list[str | None]
    dates: list[str | None]
    failures: list[TranslationFailure]


def _dt_errors(*names: str) -> tuple[type[BaseException], ...]:
    """deep_translator.exceptions classes by name, once deep_translator is imported. Before that none
    of them can have been raised, so this module never imports it."""
    module = sys.modules.get("deep_translator.exceptions")
    found = [getattr(module, name, None) for name in names] if module is not None else []
    return tuple(c for c in found if isinstance(c, type) and issubclass(c, BaseException))


def is_transient(error: BaseException) -> bool:
    """True if a retry can cure ``error``, so translate_texts retries it.

    Transient: an OSError (requests' RequestException and its ConnectionError, Timeout, SSLError,
    ChunkedEncodingError; socket timeouts; the builtin ConnectionError and TimeoutError) unless it is
    also a ValueError or TypeError (requests' MissingSchema, InvalidSchema, InvalidURL,
    InvalidHeader, StreamConsumedError: a bad argument, not an outage); and deep-translator 1.11.4's
    RequestError (Google answered a non-2xx status other than 429), TranslationNotFound (a 2xx page
    with no translation in it) and ServerException (an API server error; only deep-translator's
    other engines raise it). Anything else (ImportError, ValueError, TypeError, KeyError,
    RuntimeError, ...) is a missing package or a bug, which no retry cures.
    """
    if isinstance(error, OSError):
        return not isinstance(error, ValueError | TypeError)
    return isinstance(error, _dt_errors(*_DT_TRANSIENT))


def _call(backend: Backend, source: str, target: str, *, max_retries: int, base_delay: float,
          sleep: Callable[[float], Any], label: str) -> tuple[bool, str | None, str]:
    """One item through the backend, retrying transient errors (is_transient) with exponential
    backoff. Returns (ok, output, error detail). Only exception type names are kept: str(e) may hold
    the text.

    TranslationBlocked and every error that is not transient propagate at once, unretried: the
    original exception, except deep-translator's BaseError family (str(e) starts with the input
    text), re-raised as a RuntimeError naming its type, with its traceback but neither its message
    nor a __context__ that holds it.
    """
    errors: list[str] = []
    for attempt in range(max_retries + 1):
        if errors:
            delay = base_delay * 2 ** (attempt - 1)
            delay += _rng.uniform(0, RETRY_JITTER * delay)
            log.warning("%s: %s; retry %d of %d in %.1f s", label, errors[-1], attempt, max_retries, delay)
            if delay > 0:
                sleep(delay)
        try:
            return True, backend(source, target), ""
        except TranslationBlocked:
            raise
        except Exception as e:
            if is_transient(e):
                errors.append(type(e).__name__)
                continue
            log.error("%s: %s is not a transient error; stopping without a retry", label, type(e).__name__)
            if not isinstance(e, _dt_errors("BaseError")):
                raise
            withheld, trace = type(e).__name__, e.__traceback__
        # deep-translator's own error holds the text: raised here, outside the handler, so the new
        # error has no __context__ either
        raise RuntimeError(f"{label}: deep-translator raised {withheld}, not a transient error (its "
                           "message is withheld: it holds the text)").with_traceback(trace)
    return False, None, ", ".join(f"{name} x{n}" for name, n in Counter(errors).items())


def _translate(texts: Sequence[str], target: str, *, engine: str = "google",
               cache_dir: Path | None = None, lid: LangIdentifier | None = None,
               backend: Backend | None = None,
               cfg: Mapping[str, Any] | None = None, max_retries: int = 4, base_delay: float = 5.0,
               sleep: Callable[[float], Any] = time.sleep, cache_only: bool = False,
               retranslate_failed: bool = False, pids: Sequence[str] | None = None,
               progress_every: int = 50) -> _Run:
    cfg = cfg if cfg is not None else config.load("project")
    texts = list(texts)
    if engine not in ENGINE_NAMES:
        raise ValueError(f"unknown engine {engine!r}; one of {sorted(ENGINE_NAMES)}")
    codes = cfg["languages"]["google_translate" if engine == "google" else "belebele"]
    if target == "en" or target not in codes:
        raise ValueError(f"cannot translate into {target!r} with {engine}")
    if pids is not None:
        pids = list(pids)
        if len(pids) != len(texts):
            raise ValueError(f"{len(pids)} pids for {len(texts)} texts")
    if cache_only and retranslate_failed:
        raise ValueError("cache_only and retranslate_failed exclude each other")
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not clean_text(text):
            raise ValueError(f"texts[{i}] ({pids[i] if pids else 'no pid'}) is not a non-empty string")

    name = ENGINE_NAMES[engine]
    cache = TranslationCache(cache_dir)
    lid = lid if lid is not None else _LazyLID(cfg)
    n = len(texts)
    outputs: list[str | None] = [None] * n
    dates: list[str | None] = [None] * n
    failures: list[TranslationFailure] = []
    called: dict[str, _Outcome] = {}        # this run's engine results by cache key: one call per source
    counts: Counter[str] = Counter()
    streak = 0
    for i, source in enumerate(texts):
        pid = pids[i] if pids else None
        label = f"{name} -> {target} item {i}" + (f" ({pid})" if pid else "")
        outcome: _Outcome | None = None
        key = cache.key(name, target, source)
        if len(source) >= MAX_CHARS:
            outcome = _Outcome(None, None, ("too_long", f"{len(source)} chars"))
        elif key in called:
            outcome = called[key]
            counts["repeat"] += 1
        else:
            row = cache.get(name, target, source)
            if row is not None:
                verdict = validate(source, row["output"], target, lid)
                if verdict is None or not retranslate_failed:
                    outcome = _Outcome(row["output"], row["date"], verdict)
                    counts["cached"] += 1
        if outcome is None and cache_only:
            outcome = _Outcome(None, None, ("error", "not cached"))
        if outcome is None:
            if backend is None:
                backend = google_backend(cfg) if engine == "google" else nllb_backend(cfg=cfg)
            try:
                ok, out, detail = _call(backend, source, target, max_retries=max_retries,
                                        base_delay=base_delay, sleep=sleep, label=label)
            except TranslationBlocked as e:
                log.error("%s: blocked (%s); stopping with %d of %d items handled", label, e.kind, i, n)
                raise TranslationBlocked(f"{e} [stopped at item {i} of {n}]", done=i, kind=e.kind) from None
            counts["called"] += 1
            if ok:
                streak = 0
                date = None if _blank(out) else cache.put(name, target, source, out)["date"]
                outcome = _Outcome(out, date, validate(source, out, target, lid))
            else:
                streak += 1
                outcome = _Outcome(None, None, ("error", detail))
            called[key] = outcome

        if outcome.verdict is None:
            outputs[i], dates[i] = outcome.output, outcome.date
        else:
            rule, detail = outcome.verdict
            kept = outcome.output if isinstance(outcome.output, str) else None
            failures.append(TranslationFailure(index=i, pid=pid, lang=target, engine=name,
                                               source_sha=sha256_text(source), rule=rule, detail=detail,
                                               output=kept))
            log.warning("%s failed: %s %s", label, rule, detail)
            if streak >= MAX_CONSECUTIVE_ERRORS:
                log.error("%s: %d items in a row failed every retry with transient errors; stopping",
                          label, streak)
                raise TranslationBlocked(f"{streak} items in a row failed every retry with transient "
                                         f"(network or service) errors ({detail}): an outage, not an IP "
                                         f"block. Stopped after item {i} of {n}; check the network, then "
                                         "rerun to resume from the cache", done=i + 1, kind="streak")
        if progress_every > 0 and (i + 1) % progress_every == 0 and i + 1 < n:
            log.info("%s -> %s: %d/%d (%d cached, %d called, %d failed)", name, target, i + 1, n,
                     counts["cached"], counts["called"], len(failures))
    log.info("%s -> %s: %d texts, %d cached, %d called, %d repeats, %d failed", name, target, n,
             counts["cached"], counts["called"], counts["repeat"], len(failures))
    return _Run(outputs, dates, failures)


def translate_texts(texts: Sequence[str], target: str, *, engine: str = "google",
                    cache_dir: Path | None = None, lid: LangIdentifier | None = None,
                    backend: Backend | None = None,
                    cfg: Mapping[str, Any] | None = None, max_retries: int = 4, base_delay: float = 5.0,
                    sleep: Callable[[float], Any] = time.sleep, cache_only: bool = False,
                    retranslate_failed: bool = False, pids: Sequence[str] | None = None,
                    progress_every: int = 50) -> list[str]:
    """Translate English ``texts`` into ``target`` (our code: hi zh es ar th) through the cache.

    Per item: a source of MAX_CHARS or more is "too_long" and never sent. A cached output is
    validated; with ``retranslate_failed`` a cached output that fails is requested again (the new
    row wins). Otherwise, unless ``cache_only`` (then: "error", "not cached"), the backend is called,
    retrying a transient error (is_transient) ``max_retries`` times after delays of ``base_delay`` *
    2**attempt (+ jitter); a non-blank output is cached, then validated. A source repeated in
    ``texts`` is requested once. ``MAX_CONSECUTIVE_ERRORS`` items in a row whose every retry raised
    raise TranslationBlocked kind "streak"; the Google backend raises kind "http_429" on HTTP 429.
    Any other error propagates at once, unretried (deep-translator's text-bearing errors as a
    RuntimeError naming their type): the items before it are cached, nothing after it is called.

    Failures are collected and raised as one TranslationError after every item; its ``outputs``
    hold None for the failed items. ``pids`` (optional) label failures and logs. ``lid`` None loads
    GlotLID once, only if something needs checking; ``backend`` None is google_backend(cfg) or
    nllb_backend(cfg=cfg), created on the first call.
    """
    run = _translate(texts, target, engine=engine, cache_dir=cache_dir, lid=lid, backend=backend, cfg=cfg,
                     max_retries=max_retries, base_delay=base_delay, sleep=sleep, cache_only=cache_only,
                     retranslate_failed=retranslate_failed, pids=pids, progress_every=progress_every)
    if run.failures:
        raise TranslationError(run.failures, run.outputs)
    return [out for out in run.outputs if out is not None]


def translate_records(records: Sequence[PromptRecord], target_lang: str, *, engine: str = "google",
                      **kw: Any) -> list[PromptRecord]:
    """Copy each English record into ``target_lang``: text = clean_text(output), text_en = the
    English text, translation = TranslationInfo(engine name, "en", date of the cache row), lid
    None; pid, set, split, source, category and extra unchanged. ``kw`` goes to translate_texts.

    Refuses any record that is not an English source (never translate a translation). On failures
    raises TranslationError whose ``records`` are the successes and whose failures carry the pid.
    """
    records = list(records)
    for i, rec in enumerate(records):
        if rec.lang != "en" or rec.translation is not None:
            raise ValueError(f"records[{i}] ({rec.pid}) is not an English source (lang {rec.lang!r}); "
                             "translate only from English")
    pids = [rec.pid for rec in records]
    if len(set(pids)) != len(pids):
        raise ValueError("duplicate pids in the records to translate")
    if "pids" in kw:
        raise TypeError("translate_records takes the pids from the records")
    run = _translate([rec.text for rec in records], target_lang, engine=engine, pids=pids, **kw)
    out = []
    for rec, text, date in zip(records, run.outputs, run.dates, strict=True):
        if text is None:
            continue
        fields = rec.model_dump()
        fields.update(lang=target_lang, text=clean_text(text), text_en=rec.text, lid=None,
                      translation=TranslationInfo(engine=ENGINE_NAMES[engine], source_lang="en", date=date),
                      extra=copy.deepcopy(rec.extra))
        out.append(PromptRecord.model_validate(fields))
    if run.failures:
        raise TranslationError(run.failures, run.outputs, out)
    return out


def back_translate(texts: Sequence[str], source_lang: str, *,
                   translator: Callable[[list[str], str], list[str]] | None = None,
                   cfg: Mapping[str, Any] | None = None, cache_dir: Path | None = None,
                   batch_size: int = 32) -> list[str]:
    """Translate ``texts`` from ``source_lang`` (our code) back into English with NLLB-200 on CPU,
    for the ar/th judge sensitivity sample (P07: the inputs are generated outputs).

    Cached in a TranslationCache under engine "nllb-200-distilled-600M", lang "{source_lang}-en",
    rooted at ``cache_dir``, by default output_dir() / BACKTRANSLATION_DIRNAME, i.e.
    outputs/backtranslations/nllb-200-distilled-600M/{source_lang}-en.jsonl: the rows hold raw
    model completions, which live only in outputs/ and the private runs repo (CLAUDE.md rule 6),
    never under data/ next to the prompt translations that get published.
    ``translator(texts, lang) -> outputs`` replaces NLLB (tests); it gets the uncached distinct
    texts in one call. Raises BackTranslationError if any output is empty or identical to its input;
    never returns input text.
    """
    cfg = cfg if cfg is not None else config.load("project")
    codes = cfg["languages"]["belebele"]
    if source_lang == "en" or source_lang not in codes:
        raise ValueError(f"cannot back-translate from {source_lang!r}")
    texts = list(texts)
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not clean_text(text):
            raise ValueError(f"texts[{i}] is not a non-empty string")
    name, lang = ENGINE_NAMES["nllb"], f"{source_lang}-en"
    cache = TranslationCache(Path(cache_dir) if cache_dir is not None else
                             output_dir() / BACKTRANSLATION_DIRNAME)
    found: dict[str, str | None] = {}
    for text in texts:
        row = cache.get(name, lang, text)
        if row is not None:
            found[text] = row["output"]
    todo = [text for text in dict.fromkeys(texts) if text not in found]
    if todo:
        run = translator if translator is not None else _nllb_to_english(cfg, batch_size)
        got = list(run(todo, source_lang))
        if len(got) != len(todo):
            raise ValueError(f"the translator returned {len(got)} outputs for {len(todo)} texts")
        for text, out in zip(todo, got, strict=True):
            if not _blank(out):
                cache.put(name, lang, text, out)
            found[text] = out
    outputs: list[str | None] = [None] * len(texts)
    failures: list[TranslationFailure] = []
    for i, text in enumerate(texts):
        out = found[text]
        verdict = _check_output(text, out)
        if verdict is None:
            outputs[i] = out
        else:
            failures.append(TranslationFailure(index=i, lang=lang, engine=name, source_sha=sha256_text(text),
                                               rule=verdict[0], detail=verdict[1],
                                               output=out if isinstance(out, str) else None))
    log.info("%s %s: %d texts, %d translated now, %d failed", name, lang, len(texts), len(todo),
             len(failures))
    if failures:
        raise BackTranslationError(failures, outputs)
    return [out for out in outputs if out is not None]
