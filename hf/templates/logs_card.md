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
  hit the generation token cap, the second judge's label and the human labels where they exist, the
  sha256 of the output judged, and the generation key that links the row to the gated raw
  completions. A judgement whose judge failed keeps the kind of the failure, not its message.
- No output text and no judge text: no completion, refusals and benign answers included, and none
  of the judges' replies or reasons. They are in the gated dataset only (below).
- `aggregates/`: the result tables as CSV and `final_report.json`, the repair selection
  (`selection.json`, and the secondary selection on held-out loss), the gate reports, the judge
  decision and the human-validation reports.
- `auditkit/`: the AuditKIT evaluation records, with every output replaced by its sha256.
- `figures/`: the report's figures (below).
{{ records }}

Variants: {{ variants }}.

Aggregate files:

{{ aggregates }}

## Figures

Drawn by `lift report` from the aggregate tables and the selection; the PNGs are in `figures/`.

{{ figures }}

## Harmful content

No model output is in this dataset, whatever its label. Most outputs carry only one automatic
judge's label, so no text is published because of its label: the publishing code writes ids, labels,
flags and hashes only, and refuses to upload any file that holds a run of any output's text. The
full completions, harmful ones included, are in a gated dataset:
{{ raw_repo }}.

## Licence

CC-BY-NC-4.0. The outputs come from Tiny Aya Global and adapters of it (CC-BY-NC-4.0 plus Cohere
Labs' acceptable-use policy), and the easy-benign prompts come from Stanford Alpaca
(CC-BY-NC-4.0). Non-commercial use only.
