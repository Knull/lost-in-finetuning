# R3: Claims audit before submission

Branch `r3-claims` · ultracode · Opus 5.5 · after P08's drafts are filled with final numbers,
Nov 12

## Goal

Every number and claim in the public artifacts can be traced to a file, and nothing public
leaks harmful content. Fix discrepancies in the generators (P08), not in the text by hand.

## Check

- **Every number traced.** Every number in hf/article.md, the model card, the Space and the
  dataset cards matches a value in `outputs/metrics/`. List each number with its source file.
- **Claims carry intervals.** Every claim of an effect has an interval.
- **Wording.**
  - Diagnosis says "consistent with", never "caused by".
  - No claim that Tiny Aya resists drift unless an interval supports it.
  - Arms that weren't run are listed as not run.
- **Pre-registration followed.**
  - The primary endpoint, the corrections (Holm, BH) and the selection rule are exactly as
    pre-registered.
  - Every change appears in docs/decisions.md with its date.
- **Nothing harmful is public.**
  - No raw harmful completion appears in any public file or repo, or in the Space.
  - The adapters and the runs repo are still private.
- **Licences.** The licence notices and the OR-Bench attribution are present.
- **Submission checklist.** Walk the checklist in the build plan:
  - the toolkit chain is described;
  - Tiny Aya is the model;
  - per-language before/after metrics with intervals are there;
  - the HF artifacts exist;
  - the cost section is there;
  - the public repo carries the dated README and the frozen-recipe tag;
  - the AuditKIT issue has been filed;
  - the HF username matches the registration form.

## Hand back

A pass/fail table for the checklist, each discrepancy with its fix, and anything only a person
can resolve.
