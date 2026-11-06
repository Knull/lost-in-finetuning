# Lost in Fine-Tuning: evaluation logs

The public record of the retain / repair / reject test of a Hindi LoRA adapter of Tiny Aya Global
(Lexsi AI Safety and Alignment Hackathon, Track 2b, statement 6). The numbers in the article and the
model card come from the aggregate tables here (`aggregates/`), which `lift report` computed from the
records below: these labels, the per-item scores, the back-translation verdicts and the run manifests.
The utility verdicts quote both answers they compare, so they are in the gated raw completions instead.

## Verdicts

{{ verdicts }}

## What is here

- `labels/{variant}.jsonl`: {{ n_rows }} rows, one per judged output of the primary safety judge.
  Each row has the prompt id (`pid`, shared across languages), language, prompt set, split, the
  final label (harmful, refusal, benign or unclear), the judge's own label, the unclear flags
  (too short, repetitive, garbled), the wrong-language and code-mixing flags, whether generation
  hit the generation token cap, the second judge's label and the human labels where they exist, and the
  generation key that links the row to the gated raw completions.
- `refusal_text`: the output text, only when the primary label is a refusal and no judge and no
  human labelled that output harmful ({{ n_texts }} texts). Every other text is withheld.
- `examples/refusals.jsonl`: a small sample of those refusals per variant, language and set.
- `aggregates/`: the result tables as CSV and `final_report.json`, plus the repair selection
  (`selection.json`).
- `auditkit/`: the AuditKIT evaluation records, with every output replaced by its sha256.
{{ records }}

Variants: {{ variants }}.

Aggregate files:

{{ aggregates }}

## Harmful content

No harmful completion is in this dataset. The publishing code refuses to write a text whose
generation any judge or human labelled harmful, and checks every staged row again before upload.
The full completions, harmful ones included, are in a gated dataset with manual approval:
{{ raw_repo }}.

## Licence

CC-BY-NC-4.0. The outputs come from Tiny Aya Global and adapters of it (CC-BY-NC-4.0 plus Cohere
Labs' acceptable-use policy), and the easy-benign prompts come from Stanford Alpaca
(CC-BY-NC-4.0). Non-commercial use only.
