"""CLI command modules. Each defines ``register(subparsers)``.

Planned modules (one per prompt, so sessions never collide):
    info.py        lift info                                                  (done)
    data.py        lift data build|stats|spotcheck-sheets|apply-spotchecks    (done)
    run.py         lift smoke|generate|judge|run-dev                          (P02)
    sync.py        lift sync push|pull|status                                 (P02)
    remote.py      lift remote [watch|stop]: a job on a Colab GPU VM          (P02)
    train.py       lift train|register-adapter                                (P03)
    utility.py     lift utility                                               (P03)
    loglik.py      lift belebele|heldout-loss                                 (P03)
    gate.py        lift gate 1|2                                              (P03)
    validate.py    lift validate sample|import|score|decide-judge|...         (P04)
    repair.py      lift fit-cast|sweep|select                                 (P05)
    diagnose.py    lift diagnose ...                                          (P06)
    final.py       lift final|stats|report                                    (P07)
    publish.py     lift publish model|datasets|space|article                  (P08)
"""
