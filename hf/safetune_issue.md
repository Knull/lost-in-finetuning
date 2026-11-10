# `fit_cast_condition` and `extract_caa_vectors` keep every prompt's whole hidden-state sequence in host memory until the fit ends

<!-- For a person to file upstream on the SafeTune repository (submission checklist). Checked
against safetune 0.1.12 (`safetune/interventions/steer/cast.py` and
`safetune/core/runtime/inference/vector_extraction.py`). After filing, put the issue url in
configs/publish.yaml `safetune_issue_url` so the article links it. -->

**Version:** safetune 0.1.12 (PyPI), Python 3.12, torch on a Colab A100 (about 83 GB of host RAM).

## What happens

Both fits need one pooled vector per prompt and layer, but each keeps the full sequence it pooled
from until the fit returns. Host memory grows with the number of prompts times the prompt length,
and a fit on a few thousand prompts of a model with a long chat preamble is killed by the OOM
killer (exit 137) before it finishes.

1. `fit_cast_condition`. `_pool_hidden(h, "last")` returns `h[-1, :]`. That is a view, so every
   pooled vector the fit keeps also keeps its prompt's whole float32 `(seq, hidden)` tensor alive,
   at every candidate layer.
2. `extract_caa_vectors`. The hooks made by `SteeringVectorExtractor._make_hook` append each
   batch's whole `(batch, seq, hidden)` output to `self._activations` and pool only after the
   last batch.

## Our case

Tiny Aya Global (`CohereLabs/tiny-aya-global`, hidden size 2048, 36 layers). Its chat template
injects a preamble of about 366 tokens, so every prompt is at least that long.

- `fit_cast_condition` on 1,476 harmful and 1,800 benign prompts, 19 candidate layers: about
  70 MB per prompt, so about 100 GB for the harmful prompts alone. The process was killed with
  exit 137.
- `extract_caa_vectors` on the same prompts at five layers: 15 to 20 GB held for the benign
  prompts before the first pooling.

## Minimal reproduction

```python
import torch
from safetune.interventions.steer.cast import _pool_hidden

h = torch.zeros(400, 2048)  # one prompt's hidden states at one layer
v = _pool_hidden(h, "last")
print(v.shape, v.untyped_storage().nbytes())
# torch.Size([2048]) 3276800: the 8 KB vector keeps the 3.3 MB sequence alive
del h  # the sequence is still held through v
```

Keeping one such `v` per prompt and layer keeps `prompts x layers x seq x hidden x 4` bytes.

## Expected

Memory proportional to `prompts x layers x hidden`: the fit only ever reads the pooled vectors.

## Suggested fix

Copy the pooled vector out of its sequence, and pool inside the hook:

```python
# cast.py
def _pool_hidden(h_seq, pool):
    ...
    return h_seq[-1, :].clone()  # "last": a copy, not a view


# vector_extraction.py, SteeringVectorExtractor._make_hook
def hook_fn(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    self._activations.setdefault(layer_idx, []).append(self._pool(h.detach().cpu()))
```

(The later pooling over the stored activations then has to accept an already pooled tensor. We
keep a sequence dimension of one, `self._pool(...).unsqueeze(1).clone()`, which the existing
pooling returns unchanged.)

## What we do meanwhile

We patch both functions for the duration of the fit only (`lift.repair.cast.fit`,
`_pooled_without_whole_sequences`). A test builds the CAST artifact with and without the patch on
a tiny model and checks the condition vector, the threshold and every steering vector are equal:
the numbers are the same and only the memory differs. With the patch the same fit completed on
the same VM.

A related saving, in case it is useful: the fits read hidden states through hooks and never the
logits, so calling the model with `logits_to_keep=1` during the fit avoids materialising
`(batch, seq, vocab)` logits. On a 262k-token vocabulary that is gigabytes per padded batch.
