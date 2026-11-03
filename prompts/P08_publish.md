# P08: Hugging Face model, datasets, Space and article

Branch `p08-publish` · normal session, high effort · Opus 5.5 · drafts from Nov 8 with
placeholders, final numbers after Gate 4 (Nov 11)

## Goal

Build every public artifact listed in hf/README.md from `outputs/metrics` only. No number gets
typed by hand.

## Build

Every command does a dry run by default, printing what it would upload, and uploads only with
`--yes`. The human runs the real uploads.

1. **`lift publish model --variant <frozen winner, seed 1>`.**
   - Materialise the repaired model:
     - WiSE-FT or ReSta: merged weights.
     - CAST: the adapter-merged model plus its CAST artifacts, with a small documented
       generation wrapper.
   - Fill the model card template from the metrics:
     - the CC-BY-NC-4.0 and Cohere acceptable-use notices;
     - intended use, and the verdict;
     - the per-language before/after table with intervals;
     - limitations, and how to reproduce.
2. **`lift publish datasets`.**
   - The logs dataset: prompt ids, labels, metadata, refusal texts and aggregate tables.
   - Raw harmful completions are excluded. Assert that no uploaded text carries a harmful label
     from either judge or a human.
   - The translated prompt sets, with native-check notes and the OR-Bench CC-BY-4.0
     attribution.
3. **The Space: `hf/space/app.py` (Gradio)**, reading only the published aggregate dataset.
   - It shows the verdict, per-language heatmaps, utility retained, and redacted examples.
   - Examples are refusals only, or harmful examples cut to about 10 words plus `[redacted]`.
   - No model loading.
4. **The article: `hf/article.md`**, generated from a template plus the metrics. It covers:
   - the problem;
   - the setup, including which Lexsi tool did what;
   - the verdicts and the repair comparison table;
   - the exploratory signatures, worded "consistent with";
   - related work: the build plan's references (Wang et al. 2025, Hawkins et al. 2026, Devkar
     et al. 2026, Minionese, BabelSteering, Soteria, Aziz et al. 2026, Lexsi's Gemma 4 article,
     ReSta, CAST, WiSE-FT, OR-Bench, Belebele, Tiny Aya, the Aya Red-teaming paper);
   - limitations: machine-translated prompts, a single adapter, judge precision per language;
   - the cost section, and how to reproduce.

   Write it plainly: short sentences, no hype, and claims only where there's an interval.
5. **`hf/auditkit_issue.md`.** A minimal reproduction of the `paired_bootstrap` bug, for the
   human to file upstream.

## Done when

- All dry runs pass, and nothing uploads without `--yes`.
- A PR from `p08-publish` to `main` is open.
