# Utility: adapter-aya-std-s1 against stock, dev split (Hindi)

Judge `anthropic/claude-sonnet-5-5@utility_v1:821bed4ebfe8+provider-default+effort-low+max1024+api2026-10-21` (frontier); answers in bf16. Made by `lift utility` on 2026-11-09T19:50:04+00:00 from `outputs/utility/frontier/adapter-aya-std-s1/utility_hi_dev.jsonl`.

Both answer files were checked against configs/variants.yaml on a GPU runtime: the candidate's adapter `673ce508cd4d`, intervention `c1ec3afbd72e`.

Each pair is judged with the candidate shown as A, then as B; a win scores 1, a tie 0.5 and a loss 0, and the pair's score is the mean of its two orders. A failed order counts as a tie.

| measure | value |
|---|---|
| win rate (n=100 pairs) | 0.417 [0.330, 0.507], p vs parity 0.075 |
| failed orders | 0 of 200 (0 pairs) |
| win rate, failures as losses / as wins | 0.417 / 0.417 |
| pairs won / tied / lost | 35 / 14 / 51 |
| pair scores 1 / 0.75 / 0.5 / 0.25 / 0 | 33 / 2 / 14 / 1 / 50 |
| position: winner A / B / TIE over judged orders | 86 / 95 / 19 |
| position: pairs whose two orders agree | 91 of 100 |
| Hindi top label: candidate / stock | 0.990 / 1.000 |
| is_lang hi: candidate / stock | 0.990 / 0.990 |
| cap-hit rate: candidate / stock | 0.160 / 0.680 |

By source:

| source | pairs | win rate | failed orders | Hindi top label, candidate / stock |
|---|---|---|---|---|
| aya: Aya Dataset (human-written Hindi) | 51 | 0.387 [0.275, 0.500] | 0 | 1.000 / 1.000 |
| ind: IndicAlign Dolly_T (machine-translated Hindi) | 49 | 0.449 [0.316, 0.582] | 0 | 0.980 / 1.000 |
