Each row listed here is left out of its set's published `sft_messages.jsonl`. A stretch of its
answer is also in one of this project's model outputs: a model can repeat text it was trained on.
Every public upload of this project must pass a scan of its text against the text of every
model output on the team's machine, whatever the output's label, and the scan reads content, not
where a text came from. So the row is withheld, and the scan is given no exception. A withheld row
is training data from the public datasets named under "Licences and attribution", not a model
output, and it is in the training file that the adapters' training runs read.

{{ list }}

Every other line of a published file is its training file's line, byte for byte and in the same
order. `selection.json`, CuratorKIT's files and `report.json` describe the training files: they
count the withheld rows, and `selection.json` lists their ids.

**Rebuilding the training files.** They follow from public sources and the code repository
({{ github }}). In the repository, `pip install -e ".[data]"` and then `lift data build --only sft`
rebuild all three. The build reads the Aya Dataset, IndicAlign and databricks-dolly-15k at the
revisions under "Licences and attribution" and GlotLID at the revision `configs/project.yaml` pins,
runs CuratorKIT {{ curatorkit }} as described under "How they were made", and samples each file by
the seed and the rules its `selection.json` records. Every source is public, so no key is needed. A
rebuild after the first adapters had trained gave the same rows and the same sha256 as the files
they had read (`docs/decisions.md` item 67 in the code repository), so the training files' sha256
above tell whether a rebuild gives the same files. {{ lookup }}
