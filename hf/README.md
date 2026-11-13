# Hugging Face deliverables (P08)

The rules say every artifact is hosted on HF and the project is deployed publicly by the
deadline. Gating follows docs/decisions.md items 14 and 84. Gated repos are still hosted publicly
(anyone can see them and, after accepting the terms, download them), and the GitHub repo, Space, article, logs, prompt sets and
repair parameters are public, so the public deploy is met.

| Artifact | Access | Contents |
| --- | --- | --- |
| Repaired model | gated, automatic approval, whatever the method or verdict (item 84, pending team confirmation) | The overall frozen winner (seed 1) applied to the chosen adapter, and no other variant: the card shows the winner's repair effect. The model card carries the CC-BY-NC-4.0 and Cohere acceptable-use notices, the verdict, and the per-language before/after table. With the public parameters, any repaired model's weights give back the gated adapter: WiSE-FT as Global + (weights - Global) / lambda, ReSta as weights - alpha x (Global - Base), and a CAST winner's weights are the adapter, merged (the repair is only in the wrapper). So it is gated like the adapters, and the card and gate text say which of these applies (plus "reject": the best setting frozen anyway, not a repair). `lift publish model` refuses a public `repos.model`. |
| Repair parameters | public | CAST vectors, condition and threshold; the WiSE-FT lambda; the ReSta alpha. Public because no repaired weights are: on their own they rebuild nothing. |
| Adapters (unrepaired) | gated, automatic approval | Research artifacts whose safety may have drifted (the verdict says whether it did), with a terms note. |
| Logs dataset | public | Prompt ids, labels, flags, generation keys, output hashes (the redacted AuditKIT records), metadata and aggregate tables. No output text, and no judge text: `JudgementRecord.raw`, `native.reason` and the text of `native.judge_failure` (the judges' replies, a frontier judge's reason, and the part of a reply a failure quotes: a frontier provider's refusal, the start of a Qwen3Guard reply that did not parse) stay out with it, as CLAUDE.md rule 6 gates judge logs. A failed judgement keeps only that its judge failed, and the failure's kind. |
| Raw completions | gated, manual approval | Every judged output in full, refusals and benign answers included, the harmful ones too. No other artifact holds more of an output than the Space's 10-word cut. |
| Translated prompt sets | public | Hindi PolyRefuse, Chinese harmless, OR-Bench-Hard in 5 languages, with native-check notes and the CC-BY-4.0 attribution for OR-Bench. |
| Collections | public | One for model artifacts, one for data artifacts, both linked from the article. |
| Article | public | The organisers' sections (prompts/P08_publish.md), from the metrics files only. |
| Notebook | public, in the repo | `notebooks/acceptance_test.ipynb`: run the check on any Tiny Aya adapter. |
| Space | public, if time allows | Gradio over the aggregate dataset: verdict, heatmaps, utility retained, and a few examples, each cut to about 10 words plus `[redacted]` whatever its label. No other output text, and no live generation. |

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
| `lift publish model [--variant V] [--build --device cuda:0] [--yes]` | repaired model (gated), repair parameters, gated adapters; writes outputs/publish/model.sources.json (every value on the model card with its file and pointer) | `selection.json`, `final_report.json`, configs/variants.yaml, outputs/adapters, outputs/artifacts/cast |
| `lift publish datasets [--only logs raw prompts] [--yes]` | logs (public: ids, labels, flags and hashes, no output or judge text), raw completions (gated, manual), translated prompts (public) | outputs/judgements, generations, utility, metrics (the gate, selection and validation reports, auditkit), validation keys and labels, data/prompts |
| `lift publish space [--yes]` | the Gradio Space (`hf/space/`) with `examples.json`: two outputs per variant, language and set of the final split, each cut to at most 10 words plus `[redacted]` whatever its label | outputs/generations and the primary judgements (for the cuts); the Space itself reads the logs dataset |
| `lift publish collections [--yes]` | the two collections; writes outputs/publish/collections.json | the repo ids above |
| `lift publish article` | `hf/article.md`, the same text as the repository's `README.md`, and `hf/article.sources.json` (every value with its file and JSON pointer) | `final_report.json`, `selection.json`, the validation files (`validation/round1.json`, `validation/gate1.json`, `judge_decision.json`), configs, collections.json, configs/publish.yaml |
| `lift publish preflight [--no-history] [--no-fetch]` | nothing; fetches every branch, tag and pull-request ref of origin first (`--no-fetch` skips it: local refs only, not enough to go public), then fails on a secret, token, raw completion, notebook output, forbidden path or a run of any model output's text in the tracked files or the git history | the git repo, and every generation in this machine's `outputs/`, whatever its label (it refuses a machine that has none) |

### Order on submission day (a person runs these)

Run `lift sync pull` before every publish command, on every machine: the public-text scan, the
logs' completeness checks and the preflight read this machine's `outputs/` and `validation/`, and
refuse to run without them. Every dry run runs the public-text scan too and exits 1 on a finding.

1. `configs/publish.yaml` holds the namespace (`racetracklexsi`; `LIFT_HF_NAMESPACE` overrides
   it), `team`, `annotators` and `references` (`safetune_issue_url` is optional: without it the
   article points at `hf/safetune_issue.md`). The AuditKIT issue is not filed (decisions item
   102).
   Set `HF_TOKEN` (a write token) with a hidden prompt.
2. After Gate 4 and once every human and second-judge label is final: `lift sync pull`, then
   P07's `lift stats` and `lift report`, which write `outputs/metrics/final_report.json`.
3. On the GPU VM (the model has to fit in memory), after `lift sync pull`:
   `lift publish model --build --device cuda:0` builds the weights once and prints the dry run;
   check it, then `lift publish model --yes` (no `--build`) uploads those same weights. Without
   `--build`, staged weights are used only while their `lift_variant.json` matches the winner's
   registry entry (spec, adapter hash, Global revision); otherwise the command asks for `--build`.
4. On the laptop, after `lift sync pull`: `lift publish datasets`, then `--yes`;
   `lift publish space`, then `--yes`; `lift publish collections`, then `--yes` (it checks every
   repo is on the Hub before creating anything, so run it after the uploads).
5. `lift publish article` (no placeholder left), then commit `hf/article.md`, `README.md` and the
   figures it copies into `figures/` beside it, which the README links. The
   README is the write-up the submission form links to (docs/decisions.md item 96: an HF community
   article needs a PRO account, and the organisers accept a README in the GitHub repo).
6. `lift sync pull`, then `lift publish preflight` (without `--no-fetch` or `--no-history`), then
   the steps it prints to make the GitHub repo public.

**Retraction.** HF keeps every past commit of a repo. Each upload mirrors its staged folder
(`delete_patterns="*"`), so a withdrawn file leaves HEAD, but it stays readable in the history. To
retract a file already uploaded (say a label a reader later corrects): rebuild and upload
again, then squash the history with
`HfApi().super_squash_history(repo_id, repo_type=...)`, or delete and recreate the repo. Publish
the logs only after every label is final, so this is never needed.

### Inputs from later stages (the contract)

- `outputs/metrics/selection.json` (P05 `lift select`, `lift.repair.select.run`): `status`
  ("decided" or "incomplete"), `adapter`, `overall` ({winner, verdict, eligible}), `families`
  ({family: {winner, verdict, eligible}} once decided; a list of names while incomplete) and
  `frozen` ({family: seed-1 setting name}). Nothing is read until `status` is "decided". Seeds 2
  and 3 of each frozen setting are the same setting on the `-s2` and `-s3` adapters; all nine must
  be in configs/variants.yaml for the parameters repo.
- `outputs/metrics/final_report.json` (P07 `lift stats` / `lift report`), version 1: `verdicts`
  (each with its `kind`, Gate 1's `usefulness_verdict` and the adapter's `axes`, from which the
  article says what each check found),
  `primary_judge` (the judge id of the final judgements), `primary` (`drift`, `repair`,
  `redteam_drift`, `redteam_repair`, each with a `pooled` {est, lo, hi, p, n, seeds} or null when
  not run, and `per_seed`), `headline` (`utility_retained`, `heldout_retained`, `belebele_hi_change`,
  `overrefusal_hard_hi_change`, `gpu_hours_per_checkup`, `gpu_hours_per_variant_final`), `tables` (named in
  `lift.publish.inputs.TABLES`, each with columns {key, label, format} and rows) and `not_run`. The
  full shape is in the docstring of `src/lift/publish/inputs.py`. If P07 writes something else, change
  `lift.publish.inputs` (one place), not the article.
- `outputs/artifacts/cast/{adapter}/` (P05 `lift fit-cast`): copied as they are into the model and
  parameter repos, only while `cast.safetensors` has the sha256 its registry entry pins
  (`artifacts.cast`); a stale or missing one is listed as not built yet. `cast_generate.py` calls
  `lift.repair.cast.make`, reads `params.strength` and checks that sha again.

### Decisions made while building P08 (Nov 6, before any result existed)

- **Repo names** are `{namespace}/lift-{variant}` for the repaired model, `lift-{adapter}` per adapter,
  `lift-repair-params`, `lift-eval-logs`, `lift-raw-completions`,
  `lift-multilingual-safety-prompts` and the Space `lift-acceptance-test` (configs/publish.yaml).
- **Gating order.** A gated repo is created private, gated, filled, and only then made public, so
  its files are never public without the gate.
- **No output is public because of its label** (decisions items 62 and 85, which replace the Nov 6
  rule that published refusal texts). For about 90% of outputs the label is Qwen3Guard's alone, and
  the judge bar tolerates 15% disagreement with readers. So the logs hold ids, labels, flags and
  hashes only; a failed judgement keeps the kind of its failure, not its message; and every file
  that would be public (logs, prompts, Space, model and adapter cards) is scanned against the text
  of every model output on this machine, whatever its label. The Space's examples are cut to at
  most 10 words when it is built.
- **An AuditKIT record goes into the logs only while it is its cell's record:** the reporting door
  (`lift.judge.pipeline.load_records`) reads the cell as it stands and the read hashes to the
  record's own labels and keys under the record's judge. Any other record is left out and listed
  with the reason in the dry run.
- **Human labels** are public as labels only, joined to generations through the round keys;
  annotator ids and the validation sheets stay private (not even in the gated raw dataset).
- **Licences.** Logs and raw completions: CC-BY-NC-4.0 (outputs of a Tiny Aya derivative, and the
  Alpaca-derived prompts). Translated prompts: `other`, with the licence per set (PolyRefuse
  Apache-2.0, easy benign CC-BY-NC-4.0, OR-Bench CC-BY-4.0). A person should confirm these.
- **No result is asserted in fixed text.** The adapter terms note and gate heading say fine-tuning
  *may* affect safety; the verdict and every effect come from the metrics.
- **Repair parameters** go to their own repo (every frozen family on seeds 1-3) and also into the
  model repo (`repair/params.json`), so the model card is self-contained.
- **Only the overall winner** is published as a model (decisions item 84 for its access).
- **The article** is the repository's README.md, committed, not uploaded (decisions item 96).
- **The notebook** is a screening (first 50 dev prompts per set and language, Hindi and English,
  harmful and hard benign) through `lift run-dev`; it reuses Gate 2's thresholds from `lift.gates`.
  A CAST winner needs P05's `lift fit-cast` on the new adapter first. Its 30-minute A100 budget is
  not timed yet.
- **Preflight** scans the git history too (a public repo publishes it), checks 60-character windows
  of every model output on this machine, whatever its label, and prints findings without the matched
  text.
