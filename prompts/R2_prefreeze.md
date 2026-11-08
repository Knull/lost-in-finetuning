# R2: Review before the recipe freeze and the final eval

Branch `r2-review` · ultracode · Opus 5.5 · after P05's sweep and selection outputs exist, before
the `recipe-frozen-*` tag and before any final-split run

## Goal

Confirm that the repairs, the sweep and the selection did what the pre-registration says, and
that the final split hasn't been touched. Fix anything wrong, with a failing test first. Stop
when two rounds in a row find nothing new.
At most three review rounds in total, even if the third still finds something: then fix
what has a failing test, list the rest in the PR body as "suspected", and open the PR. (R1
ran without this cap on Nov 4-5 and was still looping after 13 hours.)

## Check

- **Repairs.**
  - The WiSE-FT sign convention is right: lambda 0.25 is closer to stock.
  - ReSta runs with `dare=False` and starts from pristine weights for every alpha.
  - CAST is fitted on train and fit data only, per seed, on the adapter being repaired. The gate
    decides per prompt, and the always-on setting really is always on.
- **Equal budgets.** At most 4 settings per family, all on the same dev prompts, all seed 1.
- **Selection.**
  - `selection.json` matches `lift.selection` applied to the logged dev metrics. Recompute it
    from the raw judgement files, not from cached summaries.
  - Constraint arithmetic is in points, measured against stock.
  - Ties go to the cheaper repair.
- **Final split untouched.** No generation file under `outputs/generations/*/*_final.jsonl`
  exists for any variant yet, except stock and the adapters if their final runs were started
  deliberately. List anything you find. `lift generate` and `lift judge` still accept
  `--split final` for any variant: R1 left that guard to P07's `lift final` (R1-06). So this
  listing, and the runs repo's history of final files, is the only check before the tag.
- **Repair cache keys (from R1).** Each CAST artifact a builder loads matches the hash its spec
  declares (R1-05). ReSta's keys carry Base's pinned sha (R1-04). Every repair entry pins its
  adapter's current hash.
- **CAST and the non-finite check (from R1).** The CAST wrapper passes `logits_processor` on to
  every inner decode and decodes each row exactly once. The harness can't see an inner decode
  that drops the processor (decisions item 60), so confirm it by reading the wrapper and by a
  test that poisons one gated row.
- **Cache keys and manifests.** Every sweep setting has its own keys, and each manifest
  records GPU time.
- **Judge decision.** It follows `judges.decision_rule` on round 1 only.

## Hand back

- Go or no-go for tagging `recipe-frozen-*`.
- The tag command.
- Anything that has to be re-run first.
