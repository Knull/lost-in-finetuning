# Hugging Face deliverables (P08)

The rules say every artifact is hosted on HF and the project is deployed publicly by the
deadline. Gating follows docs/decisions.md item 14.

| Artifact | Access | Contents |
| --- | --- | --- |
| Repaired model | public | The frozen repair applied to the chosen adapter. The model card carries the CC-BY-NC-4.0 and Cohere acceptable-use notices, the verdict, and the per-language before/after table. |
| Repair parameters | public | CAST vectors, condition and threshold; the WiSE-FT lambda; the ReSta alpha. |
| Adapters (unrepaired) | gated, automatic approval | Research artifacts that drifted on safety, with a terms note. |
| Logs dataset | public | Prompt ids, labels, flags, generation keys, output hashes (the redacted AuditKIT records), metadata and aggregate tables. No output text. |
| Raw completions | gated, manual approval | Every judged output in full, refusals and benign answers included, the harmful ones too. No other artifact holds more of an output than the Space's 10-word cut. |
| Translated prompt sets | public | Hindi PolyRefuse, Chinese harmless, OR-Bench-Hard in 5 languages, with native-check notes and the CC-BY-4.0 attribution for OR-Bench. |
| Collections | public | One for model artifacts, one for data artifacts, both linked from the article. |
| Article | public | The organisers' sections (prompts/P08_publish.md), from the metrics files only. |
| Notebook | public, in the repo | `notebooks/acceptance_test.ipynb`: run the check on any Tiny Aya adapter. |
| Space | public, if time allows | Gradio over the aggregate dataset: verdict, heatmaps, utility retained, and a few examples, each cut to about 10 words plus `[redacted]` whatever its label. No other output text, and no live generation. |

The team lead's HF username must match the registration form. The HF account must also have the
registration email on it to redeem the hackathon credits.
