# Hugging Face deliverables (P07)

| Artifact | Visibility | Contents |
| --- | --- | --- |
| Repaired model | public | The frozen repair applied to the chosen adapter. The model card carries the CC-BY-NC-4.0 and Cohere acceptable-use notices, the verdict, and the per-language before/after table. |
| Repair parameters | public | CAST vectors, condition and threshold; the WiSE-FT lambda; the ReSta alpha. |
| Adapters | private | Unrepaired adapters stay private unless the organisers say otherwise. |
| Logs dataset | public | Prompt ids, labels, metadata, refusal texts and aggregate tables. Raw harmful completions are withheld, or gated with manual approval if the organisers require them. |
| Translated prompt sets | public | Hindi PolyRefuse, Chinese harmless, OR-Bench-Hard in 5 languages, with native-check notes and the CC-BY-4.0 attribution for OR-Bench. |
| Space | public | Gradio over cached results: the verdict, per-language heatmaps, utility retained, and redacted examples. No live generation from an unrepaired adapter. |
| Article | public | Verdicts, the repair comparison, exploratory signatures, related work, limitations, and the cost section. |

The team lead's HF username must match the registration form.
