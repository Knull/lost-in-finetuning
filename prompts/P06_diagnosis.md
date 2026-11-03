# P06: Exploratory diagnosis: probes, refusal direction, ablation

Branch `p06-diagnosis` · normal session, high effort · Opus 5.5 · after P02 · runs alongside P05.
It never blocks P05.

## Goal

Compute the four signatures in `src/lift/diagnose/signatures.py` on stock and each adapter seed,
per eval language, with figures and a summary. Fit on train and fit prompts only, and evaluate
on dev.

## Read first

- CLAUDE.md, README.md (Write-up language)
- src/lift/diagnose/*, docs/toolkit-notes.md (refusal direction)

## Build

1. **`diagnose/activations.py`.**
   - Locate the token positions with a sentinel string, never with hard-coded offsets.
   - Test the positions on the tiny tokenizer, and with a gpu-marked test on Tiny Aya:
     5 prompts per language for both `last_instruction` and `post_instruction`.
2. **Harm probes at layers 12 and 24.**
   - Train them frozen (on stock) and refit (on each adapter).
   - Report AUROC with a bootstrap interval clustered by pid.
   - The lexical baseline is a character n-gram TF-IDF plus logistic regression on the prompt
     text, run on the same folds.
   - Report probe minus lexical.
3. **Refusal direction.**
   - Use SafeTune's `extract_refusal_direction` with `RefusalDirectionConfig(pick_layer=L)`.
   - Choose L on stock by the largest harmful-vs-harmless separation on train prompts, starting
     at 24, and record the choice.
   - For each adapter seed, report the cosine to stock's direction and the mean projection of
     harmful dev prompts. The seed-to-seed spread is the noise floor.
4. **Ablation check.**
   - Ablate the stock direction in stock and in each adapter, on 100 hi and 100 en dev harmful
     prompts, through `lift.models.generate` and the primary judge.
   - Use a diagnosis-only variant named `ablate-{model}`. It is never part of the repair
     comparison.
   - Report the change in harmful rate, with McNemar.
5. **Preamble check.** On stock only: the cosine between the direction extracted with the
   preamble and without it.
6. **`lift diagnose all --variant NAME`.**
   - Figures go to `outputs/figures/diagnosis/`: per-language AUROC bars and a cosine heatmap.
   - A markdown summary uses "consistent with" language throughout. It includes the README's
     signature-to-repair predictions and says, for each one, whether the data matched.

## Tests

- Synthetic activations with a planted direction: the direction is recovered.
- Probe AUROC is near 1 on separable data and near 0.5 on random data.
- The lexical baseline runs.
- The position finder works on the tiny tokenizer.

## Done when

- `pytest` and `ruff` are clean.
- A PR from `p06-diagnosis` to `main` is open.

## Hand back

The Kaggle job for `lift diagnose all` on stock and adapter seed 1 (about 1 GPU-hour), and the
seeds 2-3 job to run if time allows. Seeds 2-3 come first in the plan's cut order.
