# {{ name }} (unrepaired research adapter)

A Hindi instruction LoRA adapter of `{{ base_model }}` (revision `{{ base_revision }}`), seed
{{ seed }}. It is one of the adapters studied in the retain / repair / reject test.
{{ purpose }}

**It is gated because fine-tuning may have made its safety behaviour drift.** It is published only so the evaluation
can be reproduced. Do not deploy it. {{ model_repo_note }}

- Adapter hash (lift.train.merge.adapter_hash): `{{ adapter_hash }}`
- Training recipe, data and evaluation: {{ github }}
- Repair parameters: {{ params_repo }}
- Evaluation labels: {{ logs_repo }}

## Licence

CC-BY-NC-4.0 plus Cohere Labs' acceptable-use policy (as for Tiny Aya Global). Research use only.
