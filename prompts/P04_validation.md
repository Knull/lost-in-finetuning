# P04: Human validation sheets, agreement and the judge decision

Branch `p04-validation` · normal session, high effort · Opus 5.5 · starts now, alongside P01 and
P02. It codes against `lift/schemas.py` and the file layout in CLAUDE.md, not against P02's
code, so test with fake GenerationRecord and JudgementRecord files. Touch only
`src/lift/validation/`, `src/lift/commands/validate.py` and its own tests.

## Goal

Build blind, stratified labelling sheets the readers can fill in Google Sheets, score their
labels against both judges, and implement the Nov 6 judge decision rule, so validation round 1
can go out on Nov 5.

## Read first

- CLAUDE.md, README.md (Judging), validation/guidelines.md
- configs/project.yaml (`validation`, `judges.decision_rule`)
- src/lift/validation/sample.py, src/lift/stats/agreement.py

## Build

1. **Sampling (`validation/sample.py`)**, per its docstring, seeded, and never sampling an item
   twice.
   - Round 1 (Nov 5): drawn from stock and seed-1 adapter dev outputs: 60 hi, 40 each zh, es
     and en. Stratify by model group x set.
   - Round 2 (Nov 8): drawn from repaired-model dev outputs and frozen final outputs: 60 hi,
     20 each zh, es and en.
   - Garbled outputs are valid items, since unclear is a label.
   - Optional (config flag): a 20-item Hindi overlap for two Hindi readers, to measure
     inter-annotator agreement.
2. **Sheets.**
   - One CSV per language and round in `validation/sheets/` (gitignored), with columns item_id,
     prompt, response, label, note.
   - The keys go in `validation/keys/` (gitignored, synced to the private runs repo): item_id ->
     generation key, variant, set.
   - Each sheet starts with a header line pointing to validation/guidelines.md.
   - Sheets show the full response, which annotators need. They are private files: never commit
     them.
3. **`lift validate import --round N --lang L --file filled.csv`** writes HumanLabel records to
   `validation/labels/`.
   - Only the four classes are accepted.
   - Empty means skipped: excluded and counted.
   - Report duplicates and unknown ids.
4. **`lift validate score [--round N]`**, per language and pooled, for each judge: Qwen3Guard
   strict, Qwen3Guard loose, and the frontier judge.
   - Report agreement, kappa, the binary versions, the bar check, and confusion matrices.
   - Write markdown and JSON to `outputs/metrics/validation/`.
5. **`lift validate decide-judge`.** Apply `judges.decision_rule` to round 1, pooled, and write
   `outputs/metrics/judge_decision.json`. If the frontier judge wins:
   - print the exact config change and the docs/decisions.md entry text;
   - don't edit either file yourself: a person makes that call.
6. **`lift validate second-judge-sample`.** The list of generation keys the second judge must
   cover: a seeded 10% plus every validated item.
7. **`lift validate utility-sheet`**, then import and score.
   - 60 blind Hindi pairs (stock vs adapter answers, order randomised) for the Hindi reader.
   - Score the reader's preferences against the utility judge.

## Tests

- Stratification counts and reproducibility.
- No column in a sheet leaks the variant, set or judge label.
- Import validation: bad labels, duplicates, unknown ids.
- Scoring on synthetic labels agrees with `lift.stats.agreement`.
- The decision rule boundary: exactly 5 points, just under, just over.

## Done when

- `pytest` and `ruff` are clean.
- A PR from `p04-validation` to `main` is open.

## Hand back

- The commands to make round 1 sheets once stock and seed-1 dev outputs exist.
- Which file goes to which reader, and how to import the filled sheets.
