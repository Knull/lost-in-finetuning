# P01: Data pipeline

Branch `p01-data` · ultracode · Opus 5.5 · starts now · runs alongside P02

## Goal

`lift data build` produces every prompt set, every translation and every frozen split, plus the
two adapter training candidates and the shared utility held-out set. One Kaggle CPU session with
Internet on (no GPU quota) or a laptop should be able to rebuild all of `data/` from scratch,
with the same splits every time.

## Read first

- CLAUDE.md, README.md (pre-registration), docs/decisions.md, docs/toolkit-notes.md
- configs/project.yaml and configs/train.yaml
- src/lift/schemas.py, splits.py, io.py, config.py, paths.py
- the stubs in src/lift/data/ and src/lift/lang.py: their docstrings are the spec

## Build

1. **`lift/lang.py`.**
   - Load GlotLID with `huggingface_hub.hf_hub_download("cis-lmu/glotlid", "model.bin")` and
     fasttext (>=0.9.3; older versions break under numpy 2).
   - Check that every label in `languages.accept_labels` exists in `model.get_labels()`. If one
     doesn't, fix configs/project.yaml and add a dated line to docs/decisions.md.
2. **`lift/data/polyrefuse.py`.**
   - Shallow-clone mainlp/Multilingual-Refusal into `data/raw/`. Record the commit sha in
     configs/project.yaml (`sources.polyrefuse.commit`) and check it on every build.
   - Implement the loaders exactly as the docstring describes.
   - Assert row alignment:
     - test files: the English `instruction` is identical across languages, row for row;
     - train/val translated files: the same length as their English source files, spot-checked
       on 5 rows by comparing GlotLID language and rough length ratio;
     - the 500 harmless test rows Chinese is translated from: they must be the same 500 as every
       other language's harmless test (find which English file matches).
3. **`lift/data/translate.py`.**
   - Use Google Translate through deep-translator. Cache on disk under
     `data/interim/translations/`, keyed by (engine, target, sha256(source)).
   - Retry with exponential backoff, and keep the request rate polite.
   - Apply the failure rules in the docstring: an empty output, an output identical to the
     source, or one GlotLID doesn't assign to the target all raise. Collect every failure and
     report it; never fall back to English.
   - `back_translate`: NLLB-200 distilled 600M through transformers on CPU, for the ar/th judge
     sensitivity sample. Raise on failure.
   - Volume: ~1,570 Hindi strings, 500 Chinese, and 2,000 OR-Bench strings (400 into each of 5
     languages). Design for one resumable run of about an hour.
4. **`lift/data/sources.py`.**
   - OR-Bench-Hard-1K: a stratified sample of 400, recording the HF row indices.
   - Aya Red-teaming Hindi: read the dataset card for the right config and field names, and keep
     `harm_category` and `global_or_local` in `extra`.
   - Belebele: the six configs, pids by row index.
5. **`lift/data/sft.py`.**
   - Aya Dataset: the Hindi rows.
   - IndicAlign: pick ONE general, single-turn instruction subset whose licence allows research
     use and release of derived models. NEVER a safety, toxicity or refusal subset (for example
     HHRLHF_T or Toxic_Matrix): the adapter must be benign instruction tuning, not safety
     tuning. Record the subset, its licence and its row count in docs/decisions.md.
   - Build the shared utility held-out set first (150 from each source), with near-duplicate
     removal from both training pools.
   - Then run the CuratorKIT no-API path, producing a dedup export for both candidates and a raw
     export for the Aya candidate (the stress recipe needs it).
   - If either candidate has fewer than 3,000 usable rows after cleaning, both use the smaller
     count, so their sizes stay equal. Report it.
   - Check each dataset's licence against its card and update the README's licence list if
     anything differs.
6. **`lift/data/build.py` and `lift/commands/data.py`**, with these commands:
   - `lift data build [--only SET ...] [--langs ...] [--skip-translation]`
   - `lift data stats`: counts per set, language and split; LID failures; translation failures;
     excluded items.
   - `lift data spotcheck-sheets`: for hi, zh and es, 50 random machine-translated items per
     translated set, written as CSV to `validation/sheets/` with columns item_id, set, english,
     translation, verdict (ok / fixed / bad), corrected, note.
   - `lift data apply-spotchecks --file ...`:
     - apply the corrections and set `TranslationInfo.checked` and `edited`;
     - mark unfixable items `extra.excluded = true` and drop them from that language's prompt
       file before any generation;
     - log what changed;
     - leave the splits untouched.
7. **Freeze the splits.** Write `data/splits/{harmful,easy_benign,hard_benign,utility,belebele}.json`
   with `lift.splits.save_split`, from the sorted pids and the locked sizes.
   - The harmful split covers the `prh-test-*` ids. Train and val pids get split "train" or
     "val" directly.
   - Commit the split files; a rebuild must reproduce them byte for byte.

## Tests (offline, no network)

Fixtures:
- A miniature PolyRefuse tree under `tests/fixtures/polyrefuse/`, 3 to 5 rows per file, with the
  same quirks as the real repo, including train files whose `instruction` is the translated text.
- Fake translation and LID callables, injected (no network, no fasttext needed).

Cover:
- the field mapping, and the alignment errors raising;
- the pid scheme, and split stamping identical across languages;
- the translation cache, plus each failure rule (empty, identical to source, wrong language)
  raising;
- utility held-out ids disjoint from both training pools, including near-duplicates;
- a fixture build end to end with `--skip-translation`;
- CLI help.

Mark anything that needs the network `@pytest.mark.net`.

## Done when

- `pytest` passes offline and `ruff check src tests` is clean.
- configs/project.yaml has the PolyRefuse commit and verified GlotLID labels.
- `lift data build --help` and `lift data stats --help` work.
- A PR from `p01-data` to `main` lists what was built and anything in the spec that turned out
  wrong.

## Hand back

End your final message with:
1. The Kaggle job for a CPU notebook with Internet on, in the runner-cell format from
   kaggle/README.md. Use `lift sync push` if P02 has merged. Otherwise write to
   `/kaggle/working/data` and say how to carry it to the GPU notebooks.
2. The expected runtime.
3. The human steps: send the spot-check sheets to the Hindi, Chinese and Spanish readers, then
   run `lift data apply-spotchecks` on the filled files.

## Don't

- Translate from anything but English.
- Let a failed translation fall back to English silently.
- Recompute or edit a split file once it exists.
- Commit anything under `data/` except `data/splits/`.
- Print harmful prompts in bulk to logs. Counts and ids are enough.
