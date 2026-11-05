import pytest

from lift import config
from lift.cachekey import generation_key, intervention_config
from lift.schemas import VariantSpec
from lift.splits import assign_splits, check_disjoint

GEN = {"do_sample": False, "max_new_tokens": 256}


def test_splits_are_deterministic_and_exact():
    pids = [f"prh-test-{i:04d}" for i in range(572)]
    sizes = config.get("sizes.harmful")
    a = assign_splits(pids, sizes, seed=config.get("split_seed"))
    b = assign_splits(list(reversed(pids)), sizes, seed=config.get("split_seed"))
    assert a == b
    assert sum(v == "dev" for v in a.values()) == 172
    assert sum(v == "final" for v in a.values()) == 400


def test_split_sizes_in_config_match_the_sources():
    assert sum(config.get("sizes.harmful").values()) == 572
    assert sum(config.get("sizes.easy_benign").values()) == 500
    assert sum(config.get("sizes.hard_benign").values()) == config.get("sources.orbench.n")
    assert sum(config.get("sizes.utility").values()) == 300
    assert sum(config.get("sizes.belebele").values()) == 900


def test_split_errors():
    with pytest.raises(ValueError):
        assign_splits(["a", "a"], {"dev": 2}, seed=1)
    with pytest.raises(ValueError):
        assign_splits(["a", "b", "c"], {"dev": 2}, seed=1)
    with pytest.raises(ValueError):
        check_disjoint(["utl-aya-00001"], ["utl-aya-00001"], "utility vs train")


def _key(**over):
    args = dict(model_rev="abc123", adapter_hash="h1",
                spec=VariantSpec(name="wiseft-l0.50-x", method="wiseft", adapter="x", seed=1,
                                 params={"lambda": 0.5}),
                template_hash="t1", gen_cfg=GEN, pid="prh-test-0001", lang="hi", prompt_text="नमस्ते")
    args.update(over)
    return generation_key(**args)


def test_cache_key_changes_with_every_component():
    base = _key()
    other_spec = VariantSpec(name="wiseft-l0.75-x", method="wiseft", adapter="x", seed=1,
                             params={"lambda": 0.75})
    assert _key(spec=other_spec) != base
    assert _key(adapter_hash="h2") != base
    assert _key(template_hash="t2") != base
    assert _key(gen_cfg={**GEN, "max_new_tokens": 128}) != base
    assert _key(lang="en") != base
    assert _key(prompt_text="नमस्ते!") != base     # a fixed translation never reuses an output
    renamed = VariantSpec(name="some-other-name", method="wiseft", adapter="x", seed=1,
                          params={"lambda": 0.5})
    assert _key(spec=renamed) == base               # the name is not behaviour
    # R2-22: and each other component alone, so a key that drops one of them fails here.
    assert _key(model_rev="def456") != base
    assert _key(pid="prh-test-0002") != base
    same = dict(name="wiseft-l0.50-x", method="wiseft", adapter="x", seed=1, params={"lambda": 0.5})
    for change in ({"method": "resta"}, {"seed": 2}, {"adapter": "y"}, {"artifacts": {"cast": "c" * 64}}):
        assert _key(spec=VariantSpec(**{**same, **change})) != base, change
    # cast and cast_always at one strength, adapter and artifacts differ in method alone (CLAUDE.md rule 4).
    steer = dict(adapter="x", seed=1, params={"strength": 1.0}, artifacts={"cast": "c" * 64})
    assert (_key(spec=VariantSpec(name="cast-m1.0-x", method="cast", **steer))
            != _key(spec=VariantSpec(name="castalways-m1.0-x", method="cast_always", **steer)))
    # Every field of a variant but its name is behaviour, so a field added to VariantSpec reaches the key.
    assert set(intervention_config(VariantSpec(**same))) == set(VariantSpec.model_fields) - {"name"}


def test_cache_key_refuses_unresolved_revision_and_missing_adapter():
    with pytest.raises(ValueError):
        _key(model_rev="main")
    with pytest.raises(ValueError):
        _key(adapter_hash=None)


def test_jsonl_is_written_with_lf_only(tmp_path):
    from lift.io import append_jsonl, write_jsonl

    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"a": "नमस्ते"}, {"b": 2}])
    append_jsonl(path, [{"c": 3}])
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.count(b"\n") == 3
