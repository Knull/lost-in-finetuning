# Hugging Face deliverables (P08)

The rules say every artifact is hosted on HF and the project is deployed publicly by the
deadline. Gating follows docs/decisions.md item 14.

| Artifact | Access | Contents |
| --- | --- | --- |
| Repaired model | public | The frozen repair applied to the chosen adapter. The model card carries the CC-BY-NC-4.0 and Cohere acceptable-use notices, the verdict, and the per-language before/after table. |
| Repair parameters | public | CAST vectors, condition and threshold; the WiSE-FT lambda; the ReSta alpha. |
| Adapters (unrepaired) | gated, automatic approval | Research artifacts that drifted on safety, with a terms note. |
| Logs dataset | public | Prompt ids, labels, metadata, refusal texts and aggregate tables. |
| Raw completions | gated, manual approval | Every judged output, including the harmful ones. |
| Translated prompt sets | public | Hindi PolyRefuse, Chinese harmless, OR-Bench-Hard in 5 languages, with native-check notes and the CC-BY-4.0 attribution for OR-Bench. |
| Collections | public | One for model artifacts, one for data artifacts, both linked from the article. |
| Article | public | The organisers' sections (prompts/P08_publish.md), from the metrics files only. |
| Notebook | public, in the repo | `notebooks/acceptance_test.ipynb`: run the check on any Tiny Aya adapter. |
| Space | public, if time allows | Gradio over the aggregate dataset: verdict, heatmaps, utility retained, redacted examples. No live generation. |

The team lead's HF username must match the registration form. The HF account must also have the
registration email on it to redeem the hackathon credits.

## How it is built (`lift publish`)

Every number comes from `outputs/metrics` (results) or `configs/` (design numbers), read through
`lift.publish.inputs.Facts`. Nothing is typed by hand: `tests/test_publish_inputs.py` fails if a
template in `hf/templates/` carries a rate, a difference or a decimal. An input that does not exist
yet renders as `[PENDING: <what> from <file> (<stage>)]`, and `--yes` refuses to upload a staged
text that still holds one. Each upload command stages under `outputs/publish/` (gitignored) and is
a dry run unless given `--yes`.

| Command | Builds | Reads |
| --- | --- | --- |
| `lift publish model [--variant V] [--build --device cuda:0] [--yes]` | repaired model, repair parameters, gated adapters | `selection.json`, `final_report.json`, configs/variants.yaml, outputs/adapters, outputs/artifacts/cast |
| `lift publish datasets [--only logs raw prompts] [--yes]` | logs (public), raw completions (gated, manual), translated prompts (public) | outputs/judgements, generations, utility, metrics/auditkit, validation keys and labels, data/prompts |
| `lift publish space [--yes]` | the Gradio Space (`hf/space/`) | nothing at build time; the Space reads the logs dataset |
| `lift publish collections [--yes]` | the two collections; writes outputs/publish/collections.json | the repo ids above |
| `lift publish article` | `hf/article.md` and `hf/article.sources.json` (every value with its file and JSON pointer) | `final_report.json`, configs, collections.json, configs/publish.yaml |
| `lift publish preflight [--no-history]` | nothing; fails on a secret, token, raw completion or forbidden path in the tracked files or the git history | the git repo, and this machine's harmful outputs |

### Order on submission day (a person runs these)

1. Set `LIFT_HF_NAMESPACE` to the team's HF organisation (not a secret), fill `team`,
   `annotators`, `references` and, once filed, `auditkit_issue_url` in `configs/publish.yaml`.
   Set `HF_TOKEN` (a write token) with a hidden prompt.
2. After Gate 4: `lift sync pull`, then P07's `lift stats` and `lift report`, which write
   `outputs/metrics/final_report.json`.
3. On the GPU VM (the model has to fit in memory): `lift publish model --build --device cuda:0`,
   check the dry run, then `lift publish model --build --device cuda:0 --yes`.
4. On the laptop: `lift publish datasets`, then `--yes`; `lift publish space`, then `--yes`;
   `lift publish collections`, then `--yes`.
5. `lift publish article` (no placeholder left), commit `hf/article.md`, and post it by hand as an
   HF community article.
6. `lift publish preflight`, then the steps it prints to make the GitHub repo public.

### Inputs from later stages (the contract)

- `outputs/metrics/selection.json` (P05 `lift select`): `lift.selection.Selection` as JSON (`winner`,
  `verdict`, `eligible`, `table`), plus `adapter`, `families` ({method: {winner, ...}}) and optionally
  `frozen` (every frozen name on seeds 1-3). The frozen variants must be in configs/variants.yaml.
- `outputs/metrics/final_report.json` (P07 `lift stats` / `lift report`), version 1: `verdicts`,
  `primary` (`drift`, `repair`, `redteam_drift`, `redteam_repair`, each with a `pooled`
  {est, lo, hi, p}), `headline` (`utility_retained`, `gpu_hours_per_checkup`), `tables` (named in
  `lift.publish.inputs.TABLES`, each with columns {key, label, format} and rows) and `not_run`. The
  full shape is in the docstring of `src/lift/publish/inputs.py`. If P07 writes something else, change
  `lift.publish.inputs` (one place), not the article.
- `outputs/artifacts/cast/{adapter}/` (P05 `lift fit-cast`): copied as they are into the model and
  parameter repos. `cast_generate.py` calls `lift.repair.cast.make` and reads `params.strength`.

### Decisions made while building P08 (Nov 6, before any result existed)

- **Repo names** are `{namespace}/lift-{variant}` for the repaired model, `lift-{adapter}` per adapter,
  `lift-repair-params`, `lift-eval-logs`, `lift-raw-completions`,
  `lift-multilingual-safety-prompts` and the Space `lift-acceptance-test` (configs/publish.yaml).
- **Gating order.** A gated repo is created private, gated, filled, and only then made public, so
  its files are never public without the gate.
- **What counts as harmful for the public logs:** any judge's final or own label, a Qwen3Guard
  native Safety of Unsafe or Controversial (so the loose reading can never expose text), or any human
  label. Only primary-judge refusals with none of these carry text. Unclear and benign outputs carry
  no text either.
- **Human labels** are public as labels only, joined to generations through the round keys;
  annotator ids and the validation sheets stay private (not even in the gated raw dataset).
- **Licences.** Logs and raw completions: CC-BY-NC-4.0 (outputs of a Tiny Aya derivative, and the
  Alpaca-derived prompts). Translated prompts: `other`, with the licence per set (PolyRefuse
  Apache-2.0, easy benign CC-BY-NC-4.0, OR-Bench CC-BY-4.0). A person should confirm these.
- **No result is asserted in fixed text.** The adapter terms note says fine-tuning *may* have made
  the adapter drift; the verdict and every effect come from the metrics.
- **Repair parameters** go to their own repo (every frozen family and seed) and also into the model
  repo (`repair/params.json`), so the model card is self-contained.
- **The article** is written, not uploaded: HF community articles have no upload API.
- **The notebook** is a screening (first 50 dev prompts per set and language, Hindi and English,
  harmful and hard benign) through `lift run-dev`; it reuses Gate 2's thresholds from `lift.gates`.
  A CAST winner needs P05's `lift fit-cast` on the new adapter first. Its 30-minute A100 budget is
  not timed yet.
- **Preflight** scans the git history too (a public repo publishes it), checks 60-character windows
  of every harmful output on this machine, and prints findings without the matched text.
