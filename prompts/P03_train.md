# P03: Adapter training, utility, Belebele, and Gates 1 and 2

Branch `p03-train` · ultracode · Opus 5.5 · after R1 · runs alongside P04

## Goal

Train the Hindi adapters, measure their gain (utility win rate, held-out loss, Hindi fidelity,
Belebele), and compute Gates 1 and 2, so the recipe can be picked on Nov 5 by the README's rules.

## Read first

- CLAUDE.md, README.md (Adapter, Gates), docs/decisions.md items 3 and 10
- configs/train.yaml
- the stubs in src/lift/train/lora.py, models/loglik.py and judge/frontier.py (`pairwise`)
- src/lift/stats/paired.py (`win_rate`)

## Build

1. **`train/lora.py` and `lift train --candidate aya|indicalign --recipe standard|stress --seed N`**,
   per configs/train.yaml and the stub docstring.
   - Train with TRL SFTTrainer and PEFT, chat template on, loss on assistant tokens only.
   - Record in the manifest: supervised-token counts per batch, the loss curve, wall time and
     peak memory.
   - After training:
     - compute the adapter hash and run `lift.train.merge.parity_check`;
     - register the adapter in configs/variants.yaml, with its name from the scheme, its
       location in the runs repo, and its hash;
     - push the adapter with `lift sync push` as soon as it is saved, so a later crash loses
       nothing.
2. **`lift utility --variant NAME --split dev|final`.**
   - Generate answers to the Hindi utility set through `lift.models.generate` (set "utility"),
     for both stock and the candidate.
   - Judge each pair in both orders with `FrontierJudge.pairwise` and
     configs/rubrics/utility_v1.md, and write UtilityVerdict records.
   - Compute the win rate with its interval (`lift.stats.win_rate`).
   - Record it in AuditKIT (`audit.record_utility`) through its pairwise metrics.
   - The fallback judge is Qwen3-8B in 4-bit behind the same interface, used only if there is
     no API.
3. **Held-out loss and Hindi fidelity.**
   - `lift heldout-loss --variant NAME`: mean per-token NLL of the utility set's reference
     responses (assistant tokens only, chat template on), for stock and each adapter.
   - Hindi fidelity: the share of utility outputs whose top GlotLID label is Hindi.
4. **`models/loglik.py` and `lift belebele --variant NAME --split dev|final --langs ...`**, in
   the frozen format from the docstring. For a CAST variant the gate decides, as at generation
   time; P05 wires that in through the variant interface.
5. **`lift gate 1` and `lift gate 2`.** Read the metrics and print the decision exactly as the
   README's gate table states it, written to `outputs/metrics/gate{1,2}.json` plus markdown:
   - Gate 1: each seed-1 standard adapter's dev win rate, with its interval, against the 0.55
     bar. Also the validation round 1 bar from P04's metrics when it exists; until then report
     "pending".
   - Gate 2: the Hindi dev harmful rate change, adapter minus stock, on the 172 dev prompts,
     with McNemar (`lift.stats.mcnemar_exact`), for each seed-1 standard adapter and for the
     stress adapter. It also prints which branch of the gate rule applies.
   - Don't decide anything beyond the rule. The human picks.

## Tests (tiny model, CPU)

- A 5-step LoRA run on 4 fake rows completes.
- Assistant-only masking: preamble and user tokens get label -100, and a batch with zero
  supervised tokens raises.
- Seeds change the data order.
- Belebele letter scoring on a stub model with a forced answer.
- Utility scores average both orders correctly. When the candidate is shown as B and the judge
  says B, that's a win for the candidate.
- Gate computations on synthetic metrics, including both branches of Gate 2.

## Done when

- `pytest` and `ruff` are clean.
- A PR from `p03-train` to `main` is open.

## Hand back

End with these GPU jobs, as `lift remote --gpu A100` commands (with the Kaggle runner-cell
version as the fallback) and GPU-hour estimates:

1. Pilots: `adapter-aya-std-s1`, `adapter-indicalign-std-s1` and `adapter-aya-stress-s1`, each
   with its parity check. All three seeds of a recipe use the same GPU type and precision.
2. Pilot dev evals:
   - `lift run-dev` for each pilot;
   - `lift utility --split dev` for each;
   - `lift belebele --split dev --langs hi,en` for stock and each pilot;
   - `lift heldout-loss` for each.
3. `lift gate 1` and `lift gate 2`.
4. The seeds 2-3 training job for whichever recipe the human picks.
