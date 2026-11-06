# {{ variant }}

A Hindi LoRA adapter of Tiny Aya Global with a cheap post-hoc safety repair applied. No gradient
step was taken on the adapter to repair it. Built for the Lexsi AI Safety and Alignment
Hackathon, Track 2b, statement 6 (repairing safety drift without retraining).

- **Base model:** `{{ base_model }}` at revision `{{ base_revision }}`
- **Adapter:** {{ adapter }} (gated: {{ adapter_repo }})
- **Repair:** {{ method }}
- **Frozen parameters:** {{ params }} (all of them: {{ params_repo }})

## Verdict

{{ verdicts }}

## Before and after

All numbers are from the final split, which was used only to score frozen variants. Each effect
is paired by prompt, with a 95% interval from a bootstrap clustered by prompt id over the three
seeds. A rate's upper bound counts unclear outputs as harmful.

- **Drift** (adapter minus stock, Hindi harmful rate): {{ drift }}, p = {{ drift_p }}
- **Repair** (this model minus the adapter, Hindi harmful rate): {{ repair }}, p = {{ repair_p }}
- **Utility kept** (share of the adapter's win-rate gain over stock): {{ utility_retained }}

### Harmful rate per language

{{ per_language }}

### Over-refusal on benign prompts

{{ overrefusal }}

### Belebele

{{ belebele }}

## Intended use

Research on safety drift and post-hoc repair of fine-tuned multilingual models, and Hindi
instruction following in non-commercial settings, under the terms below. Keep Tiny Aya's
default chat template, which always injects its safety preamble: the evaluation used it.

Out of scope: any commercial use, deployment without your own safety evaluation, and any use
the Cohere Labs acceptable-use policy forbids. A repaired model is safer than the adapter on our
benchmarks, not safe in general.

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
