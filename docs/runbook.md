# Runbook: from the dev sweep to the submission

Written Nov 10 (early morning), with `main` at the article reframing commit. It lists what a
person runs, in order, from the end of the dev sweep to the Nov 13 submission. Times for steps
that have never run on a GPU are estimates and say so. If this file and the README disagree, the
README wins.

Updated Nov 10 after the sweep was read (decisions items 87 and 88): step 5 moves in front of
the freeze and takes the dev held-out lines with it, step 1 gains a count-only breakdown, and
step 3 says what `lift select` should print.

Where things stand (Nov 10, 15:45 UK): the sweep's win rates are judged and the breakdown is read
(step 1), step 5's GPU job has finished (every dev held-out score exists, seeds 2 and 3 are
trained) and the four new adapters are registered (commit b02c737). Still to do, in order: steps
2, 3, 4, 6, 7 and 8. Step 4 no longer needs its `$dev` lines. Step 2 is the one that holds
everything else: the labels, then `decide-judge`.

Conventions: PowerShell at the repo root with the venv active (`.venv\Scripts\Activate.ps1`), one
command per line (`&&` fails in Windows PowerShell 5.1, so the one-line freeze command that
`lift select` prints is split below). Secrets are set by hidden prompt in your own window, never
pasted. A GPU job runs `main` as pushed, not your working tree, so push before `lift remote`.

## Critical path

| # | Step | Where | Time | Blocks |
| --- | --- | --- | --- | --- |
| 1 | Judge the sweep's utility pairs | laptop, API | 30 to 60 min | the selection |
| 2 | Validation round 1 and the judge decision | you and the readers, then laptop | 2 h of labelling for hi and en | the freeze |
| 3 | Gate 2 record, `lift select`, freeze | laptop | 15 min | every final |
| 4 | Finals: stock, the stress adapter, the three frozen repairs | A100 | about 4 h (estimate) | the primary endpoint |
| 5 | Train seeds 2 and 3 of aya and indicalign, with the dev held-out lines | A100 | about 3 h (training measured per seed) | pooled intervals for the standard adapters; the secondary selection |
| 6 | Finals: aya and indicalign, seeds 1 to 3 | A100 | about 4 h 30 (estimate) | the three-adapter table |
| 7 | Second judge, utility judge, `lift stats`, `lift report` | laptop, API | 1 to 2 h | the article's numbers |
| 8 | Publish, article, preflight, repo public | laptop and one short A100 job | half a day | the submission |

Steps 4 and 5 do not depend on each other, and step 5 needs no freeze: start it as soon as the
sweep is in, while steps 1 to 3 happen on the laptop. Step 6 needs step 5 only for seeds 2 and 3.

## Step 0. Update the laptop

```powershell
git switch main
git pull
pip install -e ".[sync,data,eval,report,dev]"
```

`lift remote` uses the laptop's copy of `colab/job.py`, so an old checkout runs an old runner. The
`report` extra is new on the laptop: `lift report` draws its figures with matplotlib.

## Step 1. When the sweep job ends

```powershell
lift sync pull
lift sweep --adapter adapter-indicalign-stress-s1 --judge-only
```

The second line needs `ANTHROPIC_API_KEY` and `LIFT_JUDGE_MAX_USD` in that window. It submits
Message Batches and waits; if it exits 3, run it again to collect. If the sweep job itself ended
with a setting left partial, `lift select` (step 3) names the cell and its remedy.

Then, with no API key needed:

```powershell
lift sweep --adapter adapter-indicalign-stress-s1 --breakdown
```

It counts the dev records and writes `outputs\metrics\sweep_breakdown.md` and `.json`: every
label against the unclear flags and the token cap, CAST's gate per set and language and against
the label the unrepaired adapter got on the same prompt, and each setting paired with the
adapter. Counts only, no output text, and no rule reads it (decisions item 87). It answers two
questions the sweep's log left open: how many "harmful" labels sit on degenerate outputs (CAST at
2.0 above all, and each adapter), and which prompts the gate fired on. Send it with the snapshot
at the end of this file.

## Step 2. Validation round 1, then the judge decision

Do this before the freeze. The pre-registration only asks that the judge decision is made before
any final-split number exists, but the selection reads the primary judge's labels: if the decision
moved the primary judge after `lift select`, the selection and the freeze would have to be redone.

```powershell
lift validate import --round 1 --lang hi --file "<the filled hi sheet>.csv"
lift validate import --round 1 --lang en --file "<the filled en sheet>.csv"
lift validate import --round 1 --lang zh --file "<the filled zh sheet>.csv"
lift validate import --round 1 --lang es --file "<the filled es sheet>.csv"
lift validate second-judge-sample
```

`second-judge-sample` prints one `lift judge --judge second ... --keys FILE` line per variant and
split: the validated keys plus a random 10% of each cell. For the judge decision only the
validated keys matter, and round 1 was drawn from stock and the three adapters, so run the printed
dev lines of those four variants (API key in the window). The lines of the ten sweep settings can
wait or be skipped: no step needs them. Then:

```powershell
lift validate score --round 1
lift validate decide-judge --dry-run
lift validate decide-judge
lift gate 1
```

- `decide-judge` wants all four sheets and both judges on every labelled item. If the zh or es
  sheet is late, `--allow-incomplete` decides on what is there and records that it did. Use it
  only if waiting would cost the Friday submission, and say so in the article's limitations.
- If the decision says the frontier judge becomes primary, stop and tell me before step 3. Every
  dev cell would then need the frontier judge's labels (about 40,000 API judgements), and that
  is a cost and time decision, not a default.
- `lift gate 1` now has both halves: the Hindi validation bar, and the win-rate bar that already
  failed (which is why the stress adapter carries the repair comparison).
- The article states the outcome (decisions item 94). After `decide-judge`, `lift publish article`
  fills "The judge against native readers" in section 6 of `hf/article.md` (and `README.md`) from
  `outputs\metrics\validation\round1.json`, `validation\gate1.json` and `judge_decision.json`:
  the Hindi verdict with its agreement and kappa, each judge against the readers per language,
  the decision, and the primary judge's harmful rate beside the readers' by model and by prompt
  set. Read that section before the freeze and send it to me with `round1.md`. A Hindi bar that
  is not met is a result: the article says it with the README's consequence (the Hindi claims
  rest on the validated sample), and nothing is tuned around it.

## Step 3. Record the pick, select, freeze

```powershell
lift gate 2 --candidate aya
lift select --adapter adapter-indicalign-stress-s1
```

Read `outputs/metrics/selection.md` before freezing. It has every setting's Hindi harmful rate,
hard-benign refusal rate, win rate and Belebele-hi, which conditions it met, and the winner overall
and per family. Stop and send it to me if the status is `incomplete`, or if the overall verdict is
`reject`: both are results, but what gets frozen then deserves a second look.

What it should print, from the sweep's log and the judge run (Nov 10): Hindi harmful 11, 22 and
51 of 172 for WiSE-FT 0.25, 0.50 and 0.75, 59, 59 and 88 for ReSta 0.10, 0.25 and 0.40, 54, 54
and 152 for CAST 0.5, 1.0 and 2.0, and 4 with the gate always on, against the adapter's 84 and
stock's 24. Win rates, in the same order: 0.495, 0.465, 0.383; 0.270, 0.212, 0.085; 0.280 three
times; 0.040; the floor is 0.26. Only WiSE-FT 0.25 is within 3 points of stock on Belebele-hi
(0.600 against 0.625), and it meets the other three conditions, so it is the only eligible setting:
overall winner, verdict "repair". ReSta and CAST have no eligible setting, so each family's frozen
setting comes with verdict "reject": ReSta 0.10 (one violation, Belebele; 0.25 and 0.40 also miss
the floor) and CAST 0.5 (tied with CAST 1.0 on violations and on the drop, first by name). If
`selection.md` shows other numbers or another winner, stop and send it to me before freezing:
either I misread a log or a cell changed.

The secondary selection, printed beside it, should pick the same setting with verdict "repair".
Held-out loss per token: stock 2.193, the adapter 2.004, WiSE-FT 1.840, 1.859 and 1.915, ReSta
2.120, 2.536 and 3.635, conditional CAST 2.004 at every strength, always-on 4.668. As shares of
the adapter's gain that is about 187%, 177% and 147% for WiSE-FT (scaled down, the adapter fits
the references better than at full strength), 39% for ReSta 0.10 and negative beyond it, and 100%
for conditional CAST. `lift select` uses these token-weighted losses, so its shares should match.
The `vs stock: row mean NLL` line each `lift heldout-loss` prints weighs every reference equally
and gives larger differences and smaller ratios (124%, 122% and 114% for WiSE-FT, 86% for ReSta
0.10): on both weightings every WiSE-FT setting fits the references better than the full adapter.

`lift select` also prints the freeze steps; they are these, one per line:

```powershell
git add configs/frozen_recipe.json
git commit -m "Freeze the recipe"
git tag recipe-frozen-20261110
git push
git push --tags
lift sync push
```

Use the date `lift select` printed in the tag name. The tag and `lift sync push` both matter: the
VM checks the remote tag and compares the committed recipe with the `selection.json` it pulls.
After the tag the recipe is fixed: `lift select --refreeze` is refused, by design.

The stress recipe has one seed, so there is nothing more to register: the three frozen settings
are already in `configs/variants.yaml`.

## Step 4. GPU job 1: the core finals

Replace `<WISEFT>`, `<RESTA>` and `<CAST>` with the three names under `frozen` in
`outputs/metrics/selection.json` (the overall winner is one of them). The job is long, so build it
in a variable. If step 5's job already ran the dev held-out lines, leave the `$dev` line out and
drop `$dev;` from the last line: each of its lines would find its scores done, and still cost a
model load.

```powershell
$dev = "lift heldout-loss --variant adapter-indicalign-stress-s1; lift belebele --variant adapter-indicalign-stress-s1 --split dev --langs hi en; lift heldout-loss --variant wiseft-l0.25-adapter-indicalign-stress-s1; lift heldout-loss --variant wiseft-l0.50-adapter-indicalign-stress-s1; lift heldout-loss --variant wiseft-l0.75-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.10-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.25-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.40-adapter-indicalign-stress-s1; lift heldout-loss --variant cast-m0.5-adapter-indicalign-stress-s1; lift heldout-loss --variant cast-m1.0-adapter-indicalign-stress-s1; lift heldout-loss --variant cast-m2.0-adapter-indicalign-stress-s1; lift heldout-loss --variant castalways-m1.0-adapter-indicalign-stress-s1"
$variants = "stock", "adapter-indicalign-stress-s1", "<WISEFT>", "<RESTA>", "<CAST>"
$core = ($variants | ForEach-Object { "lift final --variant $_ --stages safety utility belebele heldout" }) -join "; "
$bt = ($variants | ForEach-Object { "lift final --variant $_ --stages backtranslation" }) -join "; "
lift remote --gpu A100 --max-hours 7 --job "lift sync pull; $dev; $core; $bt"
```

- The held-out lines take about 20 seconds each. They feed the secondary selection (decisions
  item 73), which freezes nothing: after this job, `lift sync pull` and the same `lift select`
  line again writes `selection_secondary.json` and leaves the frozen recipe as it is.
- `lift final` has never run on a GPU. Stock goes first as the smoke test: watch its first ten
  minutes. A line that exits 1 or 2 stops the job, so a broken final costs the setup time only.
- The back-translation stage (NLLB on the ar and th sample) runs last, for every variant, on
  purpose. It is the least important stage and the only one that loads a model this project has
  not run on the A100 yet. If it fails, every other stage of every variant is already pushed.
- Every safety stage exits 3 for now, because `easy_benign_zh_final` is 297 prompts of 300 until
  the zh spot-check verdicts are applied. The job goes on to the next line and `lift remote` exits
  1 at the end. That is expected; read the lines it lists, not the exit code.
- Each final is about 40 to 45 minutes by the dev timings (17 minutes for a dev variant with half
  the prompts), so five variants are about 4 hours. Each line pushes its outputs when it ends.

## Step 5. GPU job 2: the dev held-out scores, then seeds 2 and 3 of the standard adapters

Nothing here needs the freeze, so it can run while steps 1 to 3 happen on the laptop.

```powershell
$dev = "lift heldout-loss --variant adapter-indicalign-stress-s1; lift belebele --variant adapter-indicalign-stress-s1 --split dev --langs hi en; lift heldout-loss --variant wiseft-l0.25-adapter-indicalign-stress-s1; lift heldout-loss --variant wiseft-l0.50-adapter-indicalign-stress-s1; lift heldout-loss --variant wiseft-l0.75-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.10-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.25-adapter-indicalign-stress-s1; lift heldout-loss --variant resta-a0.40-adapter-indicalign-stress-s1"
$train = "lift train --candidate aya --recipe standard --seed 2; lift train --candidate aya --recipe standard --seed 3; lift train --candidate indicalign --recipe standard --seed 2 --drop-unfit; lift train --candidate indicalign --recipe standard --seed 3 --drop-unfit"
$cast = "lift heldout-loss --variant cast-m0.5-adapter-indicalign-stress-s1; lift heldout-loss --variant cast-m1.0-adapter-indicalign-stress-s1; lift heldout-loss --variant cast-m2.0-adapter-indicalign-stress-s1; lift heldout-loss --variant castalways-m1.0-adapter-indicalign-stress-s1"
lift remote --gpu A100 --max-hours 4.5 --job "lift sync pull; $dev; $train; $cast"
```

- The first eight lines take a few minutes and give the stress adapter's own dev Belebele-hi and
  the held-out loss of the adapter and of the WiSE-FT and ReSta settings.
- CAST's held-out lines go last: held-out scoring through the gate has never run on a GPU, and a
  line that exits 1 or 2 stops the job. Last, a failure there costs nothing that came before it.
- The held-out scores feed the secondary selection (decisions item 73), which freezes nothing.
  `lift select` prints it beside the primary one once every setting has its score; before that it
  says which are missing and the primary selection is not affected.

Measured on seed 1: aya 17 minutes, indicalign 61 minutes. Then on the laptop:

```powershell
lift sync pull
lift register-adapter adapter-aya-std-s2 adapter-aya-std-s3 adapter-indicalign-std-s2 adapter-indicalign-std-s3
git add configs/variants.yaml
git commit -m "Register seeds 2 and 3 of the standard adapters"
git push
```

## Step 6. GPU job 3: the standard adapters' finals

```powershell
$variants = "adapter-aya-std-s1", "adapter-indicalign-std-s1", "adapter-aya-std-s2", "adapter-indicalign-std-s2", "adapter-aya-std-s3", "adapter-indicalign-std-s3"
$core = ($variants | ForEach-Object { "lift final --variant $_ --stages safety utility belebele heldout" }) -join "; "
$bt = ($variants | ForEach-Object { "lift final --variant $_ --stages backtranslation" }) -join "; "
lift remote --gpu A100 --max-hours 6 --job "lift sync pull; $core; $bt"
```

Seed 1 of both goes first, so a job that is cut short still gives the three-adapter table on one
seed. If step 5 has not run yet, put only the two seed-1 names in `$variants`.

## Step 7. Judges, statistics, report (laptop)

```powershell
lift sync pull
lift final --all --judge-only
$winner = "<overall.winner in outputs\metrics\selection.json>"
lift judge --variant stock --sets harmful --langs hi en --split final --judge second --sample 1.0
lift judge --variant adapter-indicalign-stress-s1 --sets harmful --langs hi en --split final --judge second --sample 1.0
lift judge --variant $winner --sets harmful --langs hi en --split final --judge second --sample 1.0
lift stats
lift report
lift sync push
```

- `lift final --all --judge-only` runs the second judge's 10% sample and the utility judge for
  every variant of the plan. It lists the variants that are not there yet and exits 3.
- The three `--sample 1.0` lines make the second judge label every Hindi and English harmful
  prompt of stock, the adapter and the winner (2,400 judgements). `lift stats` then reports drift
  and repair under the second judge's labels too. Without them those rows say "not run".
- `lift stats` prints one line per adapter (the README's word, the reason, Gate 1's verdict) and
  the pooled drift and repair. It exits 3 while any arm is not run, and still writes the file.
- `lift report` writes `final_report.json`, the tables and the figures. The article, the model
  card and the logs card read only that file.

## Step 8. Publish and submit

The order is in `hf/README.md` ("Order on submission day"). In short:

```powershell
lift publish article
lift publish datasets
lift publish space
```

Each is a dry run that stages under `outputs/publish/` and prints what it would upload; add
`--yes` only when the dry run is clean. Before any `--yes`: set `LIFT_HF_NAMESPACE`, fill `team`,
`annotators`, `references` and `auditkit_issue_url` in `configs/publish.yaml`, and file the two
upstream issues from `hf/auditkit_issue.md` and `hf/safetune_issue.md`. The model upload needs a
short A100 job (`lift publish model --build --device cuda:0`, then `--yes`). `lift publish
preflight` must pass before the GitHub repo goes public.

What changed in publishing on Nov 10: no output text is public, whatever its label. The logs hold
ids, labels, flags and hashes. Every dry run scans what it staged against every output on the
laptop and exits 1 if a file holds a run of one. If that happens, the message names the file and
never the text; send me the file name.

## Colab units

About 135 units should be left after the sweep (167.7 before it, about 7.9 an hour). Check the
Resources panel for the real figure.

| Job | Hours | Units |
| --- | --- | --- |
| Step 4: five finals | about 4.0 | about 31 |
| Step 5: dev held-out lines and four training runs | about 3.0 | about 24 |
| Step 6: six finals | about 4.6 | about 36 |
| Step 8: build the repaired model | about 0.4 | about 3 |
| Total | about 12 | about 94 |

## If time or units run short, cut in this order

1. Seeds 2 and 3 of aya and indicalign (step 5 and four lines of step 6). The report then pools
   over seed 1 and lists the other seeds as not run. Saves about 6 hours and 47 units.
2. The Space. It is "if relevant" in the organisers' list.
3. The `--sample 1.0` second-judge lines (the 10% sample still runs).
4. The two non-winning families' finals. Not recommended: the comparison against adapter scaling
   is the third question of the study.

Never cut: stock, the stress adapter and the overall winner on the final split, validation
round 1, and the preflight.

## Sending me numbers

Aggregate metrics and label counts only, to the `results-sweep` branch, without touching your
checkout. `lift select` prints the same table itself, so this is for a second read before you tag,
not a step you have to wait for.

```powershell
git fetch origin
git worktree add ..\lift-results results-sweep
Copy-Item outputs\metrics\*.json ..\lift-results\results\ -Force
Copy-Item outputs\metrics\*.md ..\lift-results\results\ -Force
Copy-Item outputs\metrics\utility ..\lift-results\results\ -Recurse -Force
Copy-Item outputs\metrics\belebele ..\lift-results\results\ -Recurse -Force
Copy-Item outputs\metrics\heldout_loss ..\lift-results\results\ -Recurse -Force
python -c "import json, collections, pathlib; [print(p.as_posix(), dict(collections.Counter(json.loads(x)['label'] for x in p.read_text(encoding='utf-8').splitlines() if x.strip()))) for p in sorted(pathlib.Path('outputs/judgements').rglob('*_dev.jsonl'))]" | Out-File -Encoding utf8 ..\lift-results\results\label_counts.txt
git -C ..\lift-results add -A
git -C ..\lift-results commit -m "Metrics snapshot"
git -C ..\lift-results push origin results-sweep
```

The `git worktree add` line is needed once. The first two `Copy-Item` lines take
`sweep_breakdown.json` and `.md` with them (step 1). The Python line counts labels of dev cells
only: no final-split label is read before the freeze. Copy nothing else: `outputs\metrics\auditkit\` also
holds `*.full.json` records with every completion, and nothing under `outputs\generations`,
`outputs\judgements` or `outputs\utility` may go into git. A `Copy-Item` line whose folder does
not exist yet (held-out loss before step 4) just errors; go on.

## Still open, for people

- Validation round 1 sheets: hi (60) and en (40) by you, zh and es by the readers.
- The utility sheet: 60 blind stock-vs-adapter Hindi pairs (`lift validate utility-sheet`,
  `utility-import`, `utility-score`).
- The zh spot-check sheet has four invalid verdict rows. Fix them, then
  `lift data apply-spotchecks --file <sheet>` and regenerate the short zh cell, or accept that
  the zh easy-benign cell is reported as not run.
- Validation round 2, if the first round leaves time. Draw it only if it will be finished: once a
  round's keys exist, `lift publish datasets` refuses to stage the logs until every sheet of that
  round is imported, and the second judge has covered its keys.
- Team names, annotators' consent to be named, full citations, the two upstream issues.
