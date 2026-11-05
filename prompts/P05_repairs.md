# P05: Repairs, dev sweep and the selection rule

Branch `p05-repairs` · ultracode · Opus 5.5 · after P03 · runs alongside P06

## Goal

Run WiSE-FT, ReSta-A and CAST as variants through the same harness, then do the seed-1 dev
sweep, apply the selection rule (overall and per repair family), and hand over the freeze.

## Read first

- CLAUDE.md, README.md (Repairs, Selection rule)
- docs/decisions.md items 6, 7 and 8, and docs/toolkit-notes.md (SafeTune)
- configs/repairs.yaml and configs/variants.yaml
- src/lift/repair/*, src/lift/models/variants.py, src/lift/selection.py

## Build

1. **`repair/wiseft.py`, `repair/resta.py`, `repair/cast.py`**, per their docstrings, using
   SafeTune 0.1.12's Faithful entry points only:
   - `apply_wise_ft`, with `alpha = 1 - lambda`;
   - `apply_resta`, with `dare=False` passed explicitly and `device="cpu"`;
   - `fit_cast_condition`, `extract_caa_vectors` and `CASTModel`.

   Memory plan for one T4 (16 GB) and about 30 GB of system RAM:
   - the adapter model goes on the GPU in fp16;
   - Global and Base stay on the CPU for the weight-space repairs;
   - every setting starts from pristine adapter weights.
2. **Variants.** Add `wiseft`, `resta`, `cast` and `cast_always` to the dispatch table in
   `lift.models.variants`.
   - Put the parameters and artifact hashes in VariantSpec, so each setting gets its own cache
     keys.
   - Every file a builder loads must match the hash its spec declares, checked at build time
     (R1-05). Today only `checked_adapter` does this, for `artifacts.adapter`. Load each CAST
     artifact through a check that refuses a declared sha256 the file no longer has, and have
     `build_variant` refuse a builder whose loaded hashes differ from `spec.artifacts`. A CAST fit
     re-run in place must change the keys or stop the run, never resume a cell under it.
   - Every repair's registry entry pins its adapter (`artifacts.adapter`): the hash the adapter's
     own entry pins now (`lift register-adapter` writes that one). An adapter trained and
     registered again changes only its own entry, so register each of its repairs again with the
     new pin, then make their cells again. Since R1, the reporting door refuses a cell of a
     variant that names an adapter without a pinned hash or with a pin the adapter's own entry
     does not share, and refuses a cell whose recorded adapter hash
     is not the pin. That is how a repair arm and its adapter arm are known to come from the same
     weights.
   - A CAST wrapper's `generate` must pass `logits_processor` (with every other keyword) on to
     the model's `generate`. The harness's non-finite watch (R1-25) refuses a wrapper that drops
     it. A wrapper that decodes the batch in pieces is accepted when it passes `logits_processor`
     on to every piece and the pieces hold each row once (their batch sizes must add up to the
     batch's), but then a NaN in any piece refuses the whole batch. A steered re-run of the rows
     the gate fired on, after a decode of the whole batch, is refused when it passes the processor
     on and goes unchecked when it does not (the watch never sees it): decode the gated rows and
     the others as separate pieces instead.
   - ReSta depends on Base's weights, so Base's revision must reach the key (R1-04). The resta
     builder refuses a `models.base.revision` that is not a 40-hex sha (the `require_pinned` rule),
     and records that sha in the spec it builds (for example `artifacts.base`) and in the registry
     entry, so a re-pin of Base changes the ReSta keys. `lift generate` and `lift run-dev` require
     the Base pin for a resta variant before torch loads.
   - Name each one by the scheme in configs/variants.yaml.
   - Belebele scoring for a CAST variant uses the same gate as generation.
3. **`lift fit-cast --adapter NAME`.**
   - Fit on that adapter's activations, using the train and fit prompts from configs/repairs.yaml
     (`cast.fit`), with all 6 languages pooled.
   - Save the condition vector, layer, threshold, comparator, held-out F1, steering vectors and
     the SafeTune config to `outputs/artifacts/cast/{adapter}/`.
   - Report the gate's firing rate per language on the fit set's held-out half, as a
     diagnostic: it should fire on harmful prompts and not on benign ones.
4. **`lift sweep --adapter adapter-X-s1 [--repairs wiseft,resta,cast]`.** Run all 10 settings:
   3 WiSE-FT, 3 ReSta, 3 CAST strengths, and CAST with the gate always on. For each setting:
   - dev safety in all 6 languages (harmful, easy benign, hard benign) through `lift run-dev`;
   - utility dev, 100 pairs against stock;
   - Belebele-hi dev.

   It resumes after a crash, and each setting gets a manifest with its GPU time and per-token
   latency. CAST adds a prefill gate pass, so measure that overhead separately.
5. **`lift select --adapter adapter-X-s1`.**
   - Build DevScores from the metrics and call `lift.selection.select_repair` for the overall
     winner.
   - Also run the same rule within each family. Each family's best setting goes to the final
     eval, which makes 9 repaired variants (3 families x 3 seeds).
   - Write `outputs/metrics/selection.json` and markdown with the full table, then print:
     - the exact tag command, `git tag recipe-frozen-YYYYMMDD && git push --tags`;
     - the configs/variants.yaml entries for every frozen variant on seeds 1-3;
     - the fit-cast commands for seeds 2-3, which re-fit at the frozen strength on train data
       only.
6. **Language-aware CAST stays cut** (docs/decisions.md item 2).

## Tests (tiny model, CPU)

- WiSE-FT through `apply_wise_ft(alpha = 1 - lambda)` equals merging the LoRA delta scaled by
  lambda. This pins the sign convention.
- ReSta adds exactly alpha x (Global - Base), and a pristine reload undoes it.
- DARE is never enabled.
- CAST: fitting on tiny synthetic prompts returns a condition; the gate fires per prompt; the
  always-on wrapper steers every prompt; artifacts round-trip with stable hashes.
- Every setting yields a distinct cache key.
- A CAST artifact changed after registration is refused at build time (R1-05), and a resta
  variant is refused while Base is unpinned, with another Base sha giving another key (R1-04).
- `lift select` on synthetic metrics gives the same answers as `tests/test_selection.py`.

## Done when

- `pytest` and `ruff` are clean.
- A PR from `p05-repairs` to `main` is open.

## Hand back

End with these GPU jobs, as `lift remote --gpu A100` commands (Kaggle runner cells as the
fallback), with GPU-hour estimates:

1. `lift fit-cast` for seed 1.
2. `lift sweep` (about 4 T4-hours, so roughly 1 A100-hour; say how to split it if a session
   drops).
3. `lift select`.
4. After the tag: `lift fit-cast` for seeds 2-3.
