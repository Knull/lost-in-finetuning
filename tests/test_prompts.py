import pytest
from fakes import make_prompt

from lift import prompts


def fill(store, sets=("harmful", "easy_benign"), langs=("en", "hi"), n=10):
    for s in sets:
        for lang in langs:
            store[(s, lang)] = [make_prompt(f"{s[:3]}-{i:04d}", lang, s, "dev" if i % 3 else "final")
                                for i in range(n)]


def test_load_filters_split_and_sorts(fake_prompt_files):
    fake_prompt_files[("harmful", "en")] = [make_prompt("b"), make_prompt("a"),
                                           make_prompt("c", split="final")]
    got = prompts.load("harmful", "en", "dev")
    assert [r.pid for r in got] == ["a", "b"]
    assert len(prompts.load("harmful", "en")) == 3


def test_load_rejects_duplicates_and_mislabelled_records(fake_prompt_files):
    fake_prompt_files[("harmful", "en")] = [make_prompt("a"), make_prompt("a")]
    with pytest.raises(ValueError, match="duplicate"):
        prompts.load("harmful", "en")
    fake_prompt_files[("harmful", "hi")] = [make_prompt("a", lang="en")]
    with pytest.raises(ValueError, match="belong elsewhere"):
        prompts.load("harmful", "hi")


def test_load_accepts_dict_rows(fake_prompt_files):
    fake_prompt_files[("harmful", "en")] = [make_prompt("a")]
    rec = make_prompt("a").model_dump()
    fake_prompt_files[("harmful", "es")] = [dict(rec, lang="es")]
    assert prompts.load("harmful", "es")[0].lang == "es"


def test_mixed_sample_is_even_deterministic_and_dev_only(fake_prompt_files):
    fill(fake_prompt_files)
    sets, langs = ("harmful", "easy_benign"), ("en", "hi")
    a = prompts.mixed_sample(10, sets=sets, langs=langs, seed=7)
    b = prompts.mixed_sample(10, sets=sets, langs=langs, seed=7)
    assert [(r.set, r.lang, r.pid) for r in a] == [(r.set, r.lang, r.pid) for r in b]
    assert len(a) == 10 and all(r.split == "dev" for r in a)
    counts = {}
    for r in a:
        counts[(r.set, r.lang)] = counts.get((r.set, r.lang), 0) + 1
    assert sorted(counts.values()) == [2, 2, 3, 3]
    c = prompts.mixed_sample(10, sets=sets, langs=langs, seed=8)
    assert [r.pid for r in c] != [r.pid for r in a]


def test_mixed_sample_cells_are_independent(fake_prompt_files):
    fill(fake_prompt_files)
    sets, langs = ("harmful", "easy_benign"), ("en", "hi")
    before = prompts.mixed_sample(8, sets=sets, langs=langs, seed=3)
    # Removing a prompt from one cell must not change any other cell's draw.
    fake_prompt_files[("harmful", "en")] = fake_prompt_files[("harmful", "en")][:-1]
    after = prompts.mixed_sample(8, sets=sets, langs=langs, seed=3)
    keep = [(r.set, r.lang, r.pid) for r in before if (r.set, r.lang) != ("harmful", "en")]
    assert keep == [(r.set, r.lang, r.pid) for r in after if (r.set, r.lang) != ("harmful", "en")]


def test_mixed_sample_refuses_more_than_available(fake_prompt_files):
    fill(fake_prompt_files, n=3)
    with pytest.raises(ValueError, match="only"):
        prompts.mixed_sample(40, sets=("harmful",), langs=("en", "hi"))
