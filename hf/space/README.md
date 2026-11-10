---
title: Lost in Fine-Tuning
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Retain, repair or reject a Hindi adapter of Tiny Aya
---

# Lost in Fine-Tuning: results explorer

The verdicts, per-language harmful-rate heatmaps, utility retained and a few cut examples from the
retain / repair / reject test of a Hindi LoRA adapter of Tiny Aya Global (Lexsi AI Safety and
Alignment Hackathon, Track 2b, statement 6).

Its dependencies are pinned (`requirements.txt`, `sdk_version`) to the versions the project's
tests ran it with. This Space reads only the public aggregate logs dataset named in
`space_config.json`, which holds no output text. It loads no model and generates nothing. The
examples in `examples.json` are the first words of a few outputs, each cut to at most ten words
and marked `[redacted]` whatever its label, refusals included; the full outputs are in the gated
dataset only.
