# Utility: adapter-indicalign-stress-s1 against stock, dev split (Hindi)

Judge `anthropic/claude-sonnet-5-5@utility_v1:821bed4ebfe8+provider-default+effort-low+max1024+api2026-10-21` (frontier); answers in bf16. Made by `lift utility` on 2026-11-09T19:50:10+00:00 from `outputs/utility/frontier/adapter-indicalign-stress-s1/utility_hi_dev.jsonl`.

Both answer files were checked against configs/variants.yaml on a GPU runtime: the candidate's adapter `302669756c29`, intervention `756cd006ee85`.

Each pair is judged with the candidate shown as A, then as B; a win scores 1, a tie 0.5 and a loss 0, and the pair's score is the mean of its two orders. A failed order counts as a tie.

| measure | value |
|---|---|
| win rate (n=100 pairs) | 0.280 [0.200, 0.362], p vs parity 0.0001 |
| failed orders | 0 of 200 (0 pairs) |
| win rate, failures as losses / as wins | 0.280 / 0.280 |
| pairs won / tied / lost | 24 / 9 / 67 |
| pair scores 1 / 0.75 / 0.5 / 0.25 / 0 | 21 / 3 / 9 / 1 / 66 |
| position: winner A / B / TIE over judged orders | 99 / 89 / 12 |
| position: pairs whose two orders agree | 91 of 100 |
| Hindi top label: candidate / stock | 0.990 / 1.000 |
| is_lang hi: candidate / stock | 0.990 / 0.990 |
| cap-hit rate: candidate / stock | 0.130 / 0.680 |

By source:

| source | pairs | win rate | failed orders | Hindi top label, candidate / stock |
|---|---|---|---|---|
| aya: Aya Dataset (human-written Hindi) | 51 | 0.240 [0.142, 0.343] | 0 | 0.980 / 1.000 |
| ind: IndicAlign Dolly_T (machine-translated Hindi) | 49 | 0.321 [0.199, 0.449] | 0 | 1.000 / 1.000 |
