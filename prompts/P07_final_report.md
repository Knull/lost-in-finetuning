# P07: Final evaluation, statistics, tables, figures, cost

Branch `p07-final-report` · ultracode · Opus 5.5 · after R2 and the `recipe-frozen-*` tag

## Goal

One command per frozen variant runs the whole final evaluation. Another turns every judged
record into the results tables, figures, verdicts and the cost section.

## Read first

- CLAUDE.md, and the README pre-registration (Endpoints, Statistics, Verdict) in full
- src/lift/report/tables.py, src/lift/stats/

## Build

1. **`lift final --variant NAME`.** Runs on the final split:
   - all 6 languages: harmful (400), easy benign (300), hard benign (200);
   - Aya Red-teaming Hindi;
   - utility final: 200 pairs against stock;
   - Belebele final: 700 items x 6 languages;
   - the primary judge on everything, plus the ar/th back-translation sample (20%);
   - the second-judge 10% sample, queued for the API.

   It resumes after a crash and uses both GPUs. Each manifest records GPU-hours. A cell short of
   its frozen split (`lift.data.build.ShortCellError`: a translation failure still waiting for
   a reader's verdict) is skipped and listed with its remedy, in judging as well as generation.
   Since R1, `lift generate` and `lift run-dev` do this, but `lift judge` over every cell stops
   on the missing generations file. It refuses to run unless both of these hold (this protects
   the final split):
   - the variant is frozen: listed in `selection.json`, or it is stock or an adapter seed;
   - a `recipe-frozen-*` git tag exists.
2. **`lift final --all`.** Runs the 13 core variants, in an order that completes the primary
   endpoint first:
   - stock;
   - the three adapter seeds;
   - the overall winner on all three seeds;
   - then the other two families.

   If time runs out, the primary endpoint is already complete.
3. **`lift stats`.** Pairs are built only from records with the same dtype and judge; refuse a
   mixed pair loudly. It reports everything in `src/lift/report/tables.py`'s docstring:
   - the primary endpoint: McNemar per seed, plus a clustered bootstrap pooled over seeds, for
     both drift and repair;
   - the Aya Red-teaming check;
   - secondary claims with Holm correction, and exploratory ones with BH;
   - unclear bounds, wrong-language rates and cap-hit rates;
   - utility retained as a share of the adapter's gain, Belebele, and over-refusal per language;
   - the verdict per adapter;
   - judge sensitivity: strict vs loose, the second judge vs the primary on its sample, and
     back-translation agreement for ar/th.

   The second judge's cells can only be read with `load_records(judge="second",
   complete=False)`, which checks that what is there is current but not that the sample is whole
   (R1-10). A stopped frontier run leaves items unjudged on purpose, and they are simply absent.
   So the second-judge numbers check coverage themselves: every key on P04's lists
   (`validation/keys/second_judge/`, random.txt included) must have a judgement, and a missing one
   is reported, never silently dropped from the denominator. The same goes for Qwen3Guard if the
   decision rule moves it to the second role.
4. **`lift report`.**
   - Markdown tables and CSV in `outputs/metrics/`.
   - Figures in `outputs/figures/`:
     - per-language heatmaps;
     - the safety-utility trade-off of every dev setting, with the selection constraints drawn
       in;
     - a forest plot of the primary endpoint per seed.
   - The cost section, from the manifests: GPU-hours per check-up, per-token latency per repair,
     and CAST's gate overhead.
   - A missing arm is shown as "not run", never estimated.

## Tests

- Synthetic judged records for a few variants produce the expected tables.
- The final-split guard refuses an unfrozen variant and refuses to run without the tag.
- Holm and BH are applied to the right families of claims.
- The verdict logic covers all three outcomes.

## Done when

- `pytest` and `ruff` are clean.
- A PR from `p07-final-report` to `main` is open.

## Hand back

- The GPU schedule for the 13 final evals, in priority order, fitted to the remaining Colab
  compute units (with Kaggle as overflow), with the `lift remote` commands for each job.
- The local commands to finish: `lift sync pull && lift stats && lift report`.
