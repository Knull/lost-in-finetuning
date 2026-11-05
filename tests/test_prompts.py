import pytest
from fakes import make_prompt

from lift import prompts, splits
from lift.io import write_jsonl
from lift.paths import prompts_dir


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


# ------------------------------------------------- the real reader: the file against the committed split


FROZEN = {"prh-test-0000": "final", "prh-test-0001": "dev"}     # a committed data/splits/harmful.json


@pytest.fixture
def committed_split(tmp_path, monkeypatch):
    """data/splits/harmful.json in tmp, holding FROZEN (the prompt files go to the tmp data dir)."""
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    splits.save_split("harmful", FROZEN, seed=0, sizes={"dev": 1, "final": 1})
    return FROZEN


def write_prompt_file(set_name, lang, stamps):
    """data/prompts/{set}_{lang}.jsonl with one placeholder record per pid, stamped as ``stamps`` says."""
    write_jsonl(prompts_dir() / f"{set_name}_{lang}.jsonl",
                [make_prompt(pid, lang, set_name, split).model_dump(mode="json")
                 for pid, split in sorted(stamps.items())])


def test_load_takes_no_split_stamp_the_committed_split_does_not_give(committed_split):
    # R1-19: data/prompts travels through the runs repo, data/splits through git; a file stamped from another
    # assignment would put a frozen-final prompt into dev selection and a dev prompt into the final eval
    write_prompt_file("harmful", "hi", {"prh-test-0000": "dev", "prh-test-0001": "final"})
    for split in ("dev", "final", None):
        with pytest.raises(ValueError) as e:
            prompts.load("harmful", "hi", split)
        message = str(e.value)
        assert ("harmful_hi: 2 records of harmful_hi.jsonl carry another split than the committed split "
                "gives them, e.g. prh-test-0000 dev (frozen final), prh-test-0001 final (frozen dev)"
                ) in message
        assert "rebuild it (`lift data build --only harmful`), then `lift sync push`" in message
    write_prompt_file("harmful", "hi", committed_split)
    assert [(r.pid, r.split) for r in prompts.load("harmful", "hi")] == sorted(committed_split.items())
    assert [r.pid for r in prompts.load("harmful", "hi", "dev")] == ["prh-test-0001"]


def test_load_checks_every_stamp_by_the_pid_scheme_before_the_split_filter(committed_split):
    # R1-19: PolyRefuse train/val pids carry their own segment, redteam is final, and a pid the split file
    # does not cover has no split at all; a record outside the requested split is checked too
    for pid, stamp, why in (("prh-train-0003", "val", "its pid says train"),
                            ("prh-val-0001", "dev", "its pid says val"),
                            ("prh-test-0002", "dev", "not in the split file"),
                            ("prb-train-0001", "train", "not a prh test, train or val pid")):
        write_prompt_file("harmful", "hi", {**committed_split, pid: stamp})
        shown = rf"1 records of harmful_hi.jsonl .* e.g. {pid} {stamp} \({why}\)"
        with pytest.raises(ValueError, match=shown):
            prompts.load("harmful", "hi", "final")                    # pid is not a final record
    write_prompt_file("harmful", "hi", {**committed_split, "prh-train-0003": "train", "prh-val-0001": "val"})
    assert [r.pid for r in prompts.load("harmful", "hi", "val")] == ["prh-val-0001"]
    write_prompt_file("redteam", "hi", {"art-hi-0000": "final", "art-hi-0001": "dev"})
    with pytest.raises(ValueError, match=r"e.g. art-hi-0001 dev \(redteam is final\)"):
        prompts.load("redteam", "hi", "final")


def test_a_missing_split_file_is_refused_never_read_as_a_missing_prompt_file(tmp_path, monkeypatch):
    # R1-19: FileNotFoundError tells second_judge_pids (and Gate 0's all_prompts) that a language has no
    # prompt file, and they go on without it; a missing committed split file must stop them instead
    from lift.judge import pipeline

    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "no-splits")
    write_prompt_file("harmful", "hi", FROZEN)
    missing = r"harmful_hi: the committed split file .*harmful.json is missing"
    with pytest.raises(ValueError, match=missing) as e:
        prompts.load("harmful", "hi")
    assert not isinstance(e.value, FileNotFoundError)
    with pytest.raises(ValueError, match="harmful.json is missing"):
        pipeline.second_judge_pids("harmful", "final", 1.0, 0)
    with pytest.raises(FileNotFoundError, match="lift data build --only harmful --langs en"):
        prompts.load("harmful", "en")                                  # no prompt file is still that
