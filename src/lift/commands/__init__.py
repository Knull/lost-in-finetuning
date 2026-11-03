"""CLI command modules. Each defines ``register(subparsers)``.

Planned modules (one per prompt, so sessions never collide):
    info.py        lift info                       (done)
    data.py        lift data build|translate|spotcheck-sheets|apply-spotchecks   (P01)
    run.py         lift smoke|generate|judge|belebele                             (P02)
    train.py       lift train|merge|parity|utility                                (P03)
    repair.py      lift fit-cast|sweep|select                                     (P04)
    diagnose.py    lift diagnose probes|direction|ablation|preamble               (P05)
    report.py      lift validate sample|score|decide-judge, lift stats, lift report (P06)
    publish.py     lift publish model|datasets|space                              (P07)
    sync.py        lift sync push|pull  (outputs <-> private HF dataset repo)     (P02)
"""
