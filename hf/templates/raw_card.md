# Lost in Fine-Tuning: raw completions (gated)

Every output the evaluation generated, with its judgements, including outputs judged harmful.
It is gated (access is granted on accepting the terms) and meant only for safety research that
needs the text, such as checking our labels. The labels, without any output text, are public: {{ logs_repo }}.

## What is here

{{ n_files }} files, exactly as the pipeline wrote them:

- `generations/{variant}/{set}_{lang}_{split}.jsonl`: GenerationRecord (prompt id, output,
  token counts, cap hit, cache key);
- `judgements/{judge}/{variant}/{set}_{lang}_{split}.jsonl`: JudgementRecord (labels, flags,
  the judge's raw output);
- `utility/{judge}/{variant}/utility_hi_{split}.jsonl`: pairwise utility verdicts against stock;
- `auditkit/`: the full AuditKIT records, which keep the outputs.

Human annotators' labels and identities are not included.

## Terms

Access is approved by hand. Do not redistribute harmful completions, do not use them to train
models to produce harmful content, and quote at most a few words of any harmful text.

## Licence

CC-BY-NC-4.0 (outputs of Tiny Aya Global derivatives: CC-BY-NC-4.0 plus Cohere Labs'
acceptable-use policy). Non-commercial use only.
