---
title: "Lost in Fine-Tuning: results explorer"
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: "Results explorer: a Hindi Tiny Aya adapter's safety test"
---

# Lost in Fine-Tuning: results explorer

A results explorer for the retain / repair / reject test of a Hindi LoRA adapter of Tiny Aya Global
(Lexsi AI Safety and Alignment Hackathon, Track 2b, statement 6): the verdicts, per-language
harmful-rate heatmaps, utility retained and a few cut examples. It loads no model and generates
nothing. To run the test on an adapter of your own, use the acceptance-test notebook:
{{ notebook }}

A static page (`index.html`), drawn by `lift publish space` from the public aggregate logs
dataset's `aggregates/final_report.json`, which holds no output text. The examples (`examples.json`,
shown on the page) are the first words of a few outputs, each cut to at most ten words and marked
`[redacted]` whatever its label, refusals included; the full outputs are in the gated dataset only.
