# Lost in Fine-Tuning: translated safety prompt sets

The prompt translations made for the retain / repair / reject test of a Hindi adapter of Tiny
Aya Global: PolyRefuse harmful and harmless prompts in Hindi, PolyRefuse harmless prompts in
Chinese, and the OR-Bench-Hard-1K sample in five languages. PolyRefuse's own translations are
not republished here; get them from its repository.

Prompt ids (`pid`) are shared across languages, so `prh-test-0042` is the same prompt in every
language. `split` is the role the prompt played in the study (train, fit, dev or final), from
the frozen split files in the code repository.

## Files

{{ files }}

Each row keeps the English source (`text_en`) and a `translation` record: the engine
(`google-cloud-v2` is Google Translate through the Cloud Translation API v2, the tool PolyRefuse
itself used; `nllb-200-distilled-600M` is the fallback), the date, and the native spot check:
`checked` (a native reader looked at it), `edited` (the reader fixed it) and their `note`.

## Limitations

These are machine translations. Only a sample was checked by native readers (Hindi, Chinese and
Spanish); Arabic and Thai were checked by back-translation only. Some prompts are harmful
requests by design: they are prompts, not completions.

## Licences and attribution

Each set keeps the licence of its source:

{{ licences }}

OR-Bench: Cui et al., "OR-Bench: An Over-Refusal Benchmark for Large Language Models"
(bench-llm/or-bench), CC-BY-4.0. Changes: a fixed random sample of or-bench-hard-1k, machine
translated. PolyRefuse: mainlp/Multilingual-Refusal, Apache-2.0. The harmless set comes from
Stanford Alpaca, so it and its translations are non-commercial (CC-BY-NC-4.0).
