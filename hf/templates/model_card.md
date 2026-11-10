# {{ variant }}

{{ access_note }}
A Hindi LoRA adapter of Tiny Aya Global with a cheap post-hoc safety repair applied. No gradient
step was taken on the adapter to repair it. Built for the Lexsi AI Safety and Alignment
Hackathon, Track 2b, statement 6 (repairing safety drift without retraining).

{{ recipe_note }}
{{ usefulness_note }}

- **Base model:** `{{ base_model }}` at revision `{{ base_revision }}`
- **Adapter:** {{ adapter }} (gated: {{ adapter_repo }})
- **Repair:** {{ method }}
- **Frozen parameters:** {{ params }} (all of them: {{ params_repo }})

## Verdict

{{ verdicts }}

## Before and after

All numbers are from the final split, which was used only to score frozen variants. Each effect
is paired by prompt. The pooled effects describe the recipe (the same frozen setting applied to
each seed's adapter), with a 95% interval from a bootstrap clustered by prompt id over the seeds
that were run (drift: seeds {{ drift_seeds }}; repair: seeds {{ repair_seeds }}). These weights
are the seed-1 repair: their own effect is the seed-1 row, as in the table below. A rate's upper
bound counts unclear outputs as harmful.

- **Drift** (the adapter minus stock, Hindi harmful rate, pooled over seeds): {{ drift }}, {{ drift_p }}
- **Repair, the frozen recipe** (repair minus adapter, Hindi harmful rate, pooled over seeds): {{ repair }}, {{ repair_p }}
- **This model** (seed 1: these weights minus the seed-1 adapter, Hindi harmful rate, McNemar): {{ repair_this }}, {{ repair_this_p }}
- **Utility kept** (share of the adapter's win-rate gain over stock): {{ utility_retained }}

{{ per_seed }}

### Harmful rate per language

{{ per_language }}

### Over-refusal on benign prompts

{{ overrefusal }}

### Belebele

{{ belebele }}

## Intended use

{{ intended_use }} Keep Tiny Aya's default chat template, which always injects its safety
preamble: the evaluation used it.

Out of scope: any commercial use, deployment without your own safety evaluation, and any use
the Cohere Labs acceptable-use policy forbids. {{ safety_claim }}

{{ cast_usage }}

## Limitations

- Most prompts are machine translations. Native readers checked a sample in Hindi, Chinese and
  Spanish only; Arabic and Thai are exploratory.
- One adapter and one base model. This is an acceptance test of one adapter, not a general
  repair policy.
- Safety labels come from an automatic judge, validated against human labels on a sample. Its
  precision differs by language; see the judge sensitivity table in the article.
- Greedy decoding with at most {{ max_new_tokens }} new tokens.

## How to reproduce

The code, the frozen recipe tag and every command are in {{ github }}. The labels and aggregate
tables behind every number here are in {{ logs_repo }}; the raw completions are gated in
{{ raw_repo }}.

## Licence

CC-BY-NC-4.0, plus Cohere Labs' acceptable-use policy, as for Tiny Aya Global
(`{{ base_model }}`), from which this model is derived. Non-commercial use only. The Hindi
training data came from the Aya Dataset (Apache-2.0) or IndicAlign Dolly_T (whose upstream,
databricks-dolly-15k, is CC-BY-SA-3.0); see the repository README for details.
