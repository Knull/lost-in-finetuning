# Running on Colab (primary GPU)

Reported runs use a Colab A100 in bf16. The Google AI Pro plan gives 200 Colab compute units a
month. A plain A100 is 40 GB; turning on High-RAM gives the 80 GB card.

Rates from Colab's GPU price list, April 2024. Check the Resources panel after connecting,
since rates change:

| GPU | Units per hour | Hours from 200 units |
| --- | --- | --- |
| T4 | 1.19 | ~168 |
| L4 | 1.71 | ~117 |
| A100 40 GB | 5.40 | ~37 |
| A100 80 GB | 7.52 | ~26 |

For batched generation of a 3B model, an A100 should be roughly 5 times as fast as a T4. That
puts the project's ~40 T4-hours at about 8 to 10 A100-hours, or 45 to 75 units. Gate 0 measures
the real speed. Idle runtimes still burn units, so stop a runtime as soon as its job ends.

## The Colab CLI (preferred)

Google's open-source Colab CLI (`google-colab-cli`, command `colab`) starts a runtime from a
terminal, runs a script on it, fetches outputs, and stops it. A local Claude Code session can
drive it, and the CLI's repo ships a `COLAB_SKILL.md` for agents. P02 verifies this workflow
before anything depends on it.

```bash
uv tool install google-colab-cli
colab --help                                   # log in as prompted, with the Google AI Pro account
colab run --gpu A100 colab/job.py --job "lift sync pull; lift smoke; lift sync push"
```

`colab/job.py` (written in P02) does three things on the runtime:
- clones the repo using `GITHUB_TOKEN` and installs it;
- runs each `lift ...` line in turn;
- pushes outputs to the runs repo before exiting.

The command names above come from the CLI's announcement. P02 checks them against
`colab --help` and fixes this file.

Secrets the job needs: `HF_TOKEN`, `GITHUB_TOKEN`, `LIFT_RUNS_REPO` and the judge API key.
Pass them from your local environment; never write them into the repo.

## Notebook fallback

If the CLI won't cooperate, use a normal Colab notebook:

1. Set Runtime > Change runtime type to A100 GPU (High-RAM for the 80 GB card).
2. Add the four secrets under the key icon (Colab secrets) and read them with
   `google.colab.userdata.get`.
3. Use the same two cells as `kaggle/README.md`, with
   `/content/lift` in place of `/kaggle/working/lift`.

Colab Pro has no background execution, so keep the tab open while a notebook job runs. Every job
starts with `lift sync pull` and ends with `lift sync push`, and generation and judging resume by
cache key, so a dropped session costs only the unfinished batch.
