---
title: Lost in Fine-Tuning
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Retain, repair or reject a Hindi adapter of Tiny Aya
---

# Lost in Fine-Tuning: results explorer

The verdicts, per-language harmful-rate heatmaps, utility retained and example refusals from the
retain / repair / reject test of a Hindi LoRA adapter of Tiny Aya Global (Lexsi AI Safety and
Alignment Hackathon, Track 2b, statement 6).

This Space reads only the public aggregate logs dataset named in `space_config.json`. It loads no
model and generates nothing. It shows refusals only; any other text would be cut to about ten
words and marked `[redacted]`, and the dataset holds no harmful completion.
