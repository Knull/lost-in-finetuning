import hashlib
import json
from collections import Counter

import pytest

from lift import config, paths, splits
from lift.cachekey import generation_key, intervention_config
from lift.schemas import VariantSpec
from lift.splits import assign_splits, check_disjoint, load_split

GEN = {"do_sample": False, "max_new_tokens": 256}

# R2-23: the frozen dev/final membership. The README "Prompt sets" table gives the per-split sizes (per
# language); the committed data/splits/{set}.json files, frozen Nov 4, are pinned by the sha256 of their bytes
# (LF everywhere, .gitattributes eol=lf), so a same-size swap of two pids, or a file edited, merged or
# regenerated, fails here, as does a sizes.* edit that keeps a set's sum. Changing either is a locked decision
# (CLAUDE.md rules 1 and 5): a dated docs/decisions.md entry, and these pins move with it.
README_SIZES = {
    "harmful": {"dev": 172, "final": 400},
    "easy_benign": {"dev": 200, "final": 300},
    "hard_benign": {"fit": 100, "dev": 100, "final": 200},
    "utility": {"dev": 100, "final": 200},
    "belebele": {"dev": 200, "final": 700},
}
FROZEN_SPLITS = {
    "harmful": "f8f046ccc27d01e180d319ef4ed3186bb8cf2f7a70fb471365032e8a4426d78d",
    "easy_benign": "e8946495e299203df0b79d60e359f6c31da50f8c6efc2a9a716572d0391ddc8b",
    "hard_benign": "460f38ff17a22fe8a557d578b4d4772ad33814fd01494e80d33b4d3ebe34cfd3",
    "utility": "49d31617b78409f2b259f26b4469dad38259505cfdae0ae78cd639b488717e64",
    "belebele": "5bfa0c6f835c23b2021a5c115f7c912f66f4d58c4d7178ea1c9d91a9e4888d99",
}


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


def test_split_sizes_in_config_are_the_readme_sizes():
    for name, sizes in README_SIZES.items():
        assert config.get(f"sizes.{name}") == sizes, name
        assert list(config.get(f"sizes.{name}")) == list(sizes), name      # assign_splits takes config order


def test_the_committed_split_files_are_the_frozen_ones():
    # The repo's own files (lift.paths.SPLITS_DIR, which no test moves), read as every stage reads them.
    assert paths.SPLITS_DIR == paths.REPO_ROOT / "data" / "splits" and splits.SPLITS_DIR == paths.SPLITS_DIR
    assert sorted(p.stem for p in paths.SPLITS_DIR.glob("*.json")) == sorted(FROZEN_SPLITS)
    for name, digest in FROZEN_SPLITS.items():
        path = paths.SPLITS_DIR / f"{name}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"data/splits/{name}.json changed"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert (doc["name"], doc["seed"], doc["sizes"], doc["n"]) == (
            name, config.get("split_seed"), README_SIZES[name], sum(README_SIZES[name].values())), name
        assert Counter(load_split(name).values()) == Counter(README_SIZES[name]), name


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
