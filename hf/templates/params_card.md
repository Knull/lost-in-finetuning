# Lost in Fine-Tuning: frozen repair parameters

The parameters of every repair setting frozen by the selection rule (seed-1 dev split only,
before the final split was touched), for each repair family and seed:

{{ table }}

- `params/{variant}.json`: the variant's registry entry (method, adapter, seed, parameters and
  the sha256 of every artifact it loads), the base and reference models, and the selection rule.
- `cast/{adapter}/`: the CAST condition (vector, layer, threshold, comparator) and steering
  vectors fitted with SafeTune on that adapter's train and fit prompts.

WiSE-FT uses our lambda (0 is stock, 1 is the adapter); SafeTune's `apply_wise_ft` takes
alpha = 1 - lambda. ReSta adds alpha x (Global - Base) with DARE off.

These parameters are public on their own. With them, a repaired model's weights would give back
the unrepaired adapter, so every repaired model is gated with automatic approval, like the
adapters.

Code and the frozen recipe tag: {{ github }}. Evaluation labels: {{ logs_repo }}.

## Licence

CC-BY-NC-4.0 plus Cohere Labs' acceptable-use policy (the vectors derive from Tiny Aya Global).
