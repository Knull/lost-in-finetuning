"""Human validation of the judges (P04).

    store         where validation files live; sheet CSV writing and reading
    outputs       prompts, generations and judgements as the validation reads them
    sample        rounds 1 and 2: stratified, blind, seeded sheets and their keys
    labels        importing filled sheets as HumanLabel records
    score         judge-vs-human agreement, gate1.json and the Nov 6 judge decision
    second_judge  the generation keys the frontier judge must cover
    utility       the blind 60-pair check of the utility judge

``lift validate ...`` (lift.commands.validate) runs each step.
"""
