# P08: Hugging Face artifacts, article, notebook and Space

Branch `p08-publish` · normal session, high effort · Opus 5.5 · drafts from Nov 8 with
placeholders, final numbers after Gate 4 (Nov 11)

## Goal

Build everything the organisers require, from `outputs/metrics` only. No number gets typed by
hand.

What the kickoff slides and tracks page require:
- every artifact hosted on HF (models, adapters, logs, datasets);
- the project deployed publicly by the deadline: this repo made public, plus the HF links;
- before-and-after metrics, more than one of them;
- an HF article, plus a Space "if relevant", plus an optional (preferred) ROI comparison.

Judges score write-up clarity, practical usefulness and code or notebook quality, each out of
10, plus an overall review out of 100.

## Build

Every command does a dry run by default, printing what it would upload, and uploads only with
`--yes`. The human runs the real uploads.

1. **`lift publish model --variant <frozen winner, seed 1>`.**
   - Materialise the repaired model:
     - WiSE-FT or ReSta: merged weights.
     - CAST: the adapter-merged model plus its CAST artifacts, with a small documented
       generation wrapper.
   - Model card, filled from the metrics:
     - the CC-BY-NC-4.0 and Cohere acceptable-use notices;
     - intended use, and the verdict;
     - the per-language before/after table with intervals;
     - limitations, and how to reproduce.
   - Also upload the repair parameters (CAST vectors and threshold, the WiSE-FT lambda, the
     ReSta alpha).
   - Upload the unrepaired adapters as gated repos (automatic approval, with a terms note
     saying they are research artifacts that drifted on safety). docs/decisions.md item 14.
2. **`lift publish datasets`.**
   - The logs dataset (public): prompt ids, labels, flags, generation keys, output hashes (the
     redacted AuditKIT records), metadata and aggregate tables; no output text, and no judge text.
     A judgement goes in without `JudgementRecord.raw`, `native.reason` and the text of
     `native.judge_failure` (the judges' replies, a frontier judge's reason, and the part of a
     reply a failure quotes: up to 200 characters of a frontier provider's refusal, up to 40 of a
     Qwen3Guard reply that did not parse): they stay out with every output text, as CLAUDE.md
     rule 6 gates judge logs, so a judgements file never goes up as it is. A failed judgement
     keeps only that its judge failed, and the failure's kind
     (`lift.validation.score.failure_kind`: the text before its first ":" or quote). The
     validation reports (`outputs/metrics/validation/*.json`, `checks.failure_reasons`) and
     `outputs/metrics/judge_decision.json` (`failure_reasons`) already count failures by that
     kind, so they go in as they are. Upload files by name, never a folder: a `*.full.json`
     AuditKIT record, which holds every completion, can sit beside the public one in
     `outputs/metrics/auditkit/`. An AuditKIT record goes in only while
     `lift.judge.pipeline.load_records` (`complete=True`, with the judge role and precision its
     `metadata.lift` names) accepts its cell as it stands, and only if that read is the record's:
     its labels and keys hash to the record's `labels_sha256` and `keys_sha256` (as
     `lift.audit.record_safety` computes them) and its judge id is the record's `judge_id`; every
     other record is dropped and listed with the reason (`lift.audit.load_record` reads a retired
     one as None). A record outlives a change that makes its cell unreportable (a fixed prompt
     text, other labelling rules, a re-registered variant, a re-pinned Global, a harness bump)
     until that cell's `lift restart` retires it, so a record on disk proves nothing.
   - The raw completions (gated, manual approval): every judged output, refusals and benign
     answers included. Most outputs carry only Qwen3Guard's label, so no output is made public
     because of its label.
   - The guard reads content, not labels: `lift publish datasets` (its dry run too) and
     `lift publish preflight` fail if any public file contains any generation's output text, or a
     long substring of it (more than the ~10 words a Space example keeps), whatever its label.
   - The translated prompt sets (public), with native-check notes and the OR-Bench CC-BY-4.0
     attribution.
3. **`lift publish collections`.** Two HF collections, model artifacts and data artifacts,
   listing everything above. The article links to both.
4. **`notebooks/acceptance_test.ipynb`.** This is the practical-usefulness and notebook-quality
   piece: a Colab-ready notebook that runs the retain / repair / reject check on a small dev
   sample for any Tiny Aya adapter, then applies the frozen repair.
   - It runs end to end on an A100 in under 30 minutes.
   - Short markdown cells explain each step, and it uses only `lift` functions.
5. **The article: `hf/article.md`**, generated from a template plus the metrics. Use these
   sections, in this order (the organisers' list):
   1. Problem and approach. Name the track statement: 2b-6, with 2b-8 and 2a-4.
   2. Repos used and their features: CuratorKIT for data, SafeTune for the repairs and
      direction tools, AuditKIT for the evaluation record. Say which feature did what.
   3. Models and licences.
   4. Datasets, benchmarks and licences, with the rationale for each choice.
   5. Metrics chosen and why: harmful rate with unclear bounds, easy and hard over-refusal,
      utility win rate, Belebele.
   6. Experiments and findings:
      - the verdicts and the repair comparison table;
      - the exploratory signatures, worded "consistent with";
      - limitations: machine-translated prompts, one adapter, judge precision per language.
   7. Cost and ROI: GPU-hours per check-up and per-token latency, compared with re-running
      safety fine-tuning. No commercial claims, because Tiny Aya derivatives are
      non-commercial.
   8. Related work: the build plan's references (Wang et al. 2025, Hawkins et al. 2026, Devkar
      et al. 2026, Minionese, BabelSteering, Soteria, Aziz et al. 2026, Lexsi's Gemma 4
      article, ReSta, CAST, WiSE-FT, OR-Bench, Belebele, Tiny Aya, the Aya Red-teaming paper).
   9. What we would do with more credits.
   10. Links: the model and data collections, the repo, the notebook, the Space.
   11. Team: HF tags for the teammates. Thank the annotators only if they agree to be named;
       leave a placeholder for the human to fill in.

   Write it plainly: short sentences, no hype, and claims only where there's an interval.
6. **Space: `hf/space/app.py` (Gradio), if time allows.** It's "if relevant" and sits first in
   the cut order.
   - It reads only the public aggregate dataset, plus a few examples cut from the gated
     completions when the Space is built.
   - It shows the verdict, per-language heatmaps, utility retained, and those examples, each cut
     to about 10 words plus `[redacted]` whatever its label, refusals included. No other output
     text: the content guard of step 2 covers the Space's files too.
   - No model loading.
7. **`hf/auditkit_issue.md`.** A minimal reproduction of the `paired_bootstrap` bug, for the
   human to file upstream.
8. **Making the repo public.** Write a pre-flight check (`lift publish preflight`) that fails if
   any tracked file, in any commit (a public repo shows its history), contains a generation's
   output text or a long substring of it (step 2's content guard), a secret or a token, then list
   the steps to make the GitHub repo public.

## Done when

- All dry runs and the preflight pass, and nothing uploads without `--yes`.
- A PR from `p08-publish` to `main` is open.
