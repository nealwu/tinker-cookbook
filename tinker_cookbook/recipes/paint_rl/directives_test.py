import re
from collections import Counter
from typing import cast

from tinker_cookbook.recipes.paint_rl.directives import (
    _BY_WATER,
    _FROM_ABOVE,
    _LIGHT_SET,
    _NO_SKY,
    DIRECTED_TIERS,
    HELDOUT_DIRECTIONS,
    TRAIN_DIRECTIONS,
    build_directed_heldout,
    build_directed_prompts,
)
from tinker_cookbook.recipes.paint_rl.env import (
    PaintEnvGroupBuilder,
    PaintRLDatasetBuilder,
    RewardWeights,
    build_run_prompts,
)
from tinker_cookbook.recipes.paint_rl.prompts import (
    COLOURS,
    NOVEL_SUBJECTS,
    STYLES,
    SUBJECTS,
    PaintPrompt,
    build_prompt_pool,
    build_prompt_splits,
)
from tinker_cookbook.recipes.paint_rl.verifier import VerifierConfig
from tinker_cookbook.renderers import Renderer

ALL_DIRECTIONS = TRAIN_DIRECTIONS + HELDOUT_DIRECTIONS


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _spread(counts: Counter) -> int:
    return max(counts.values()) - min(counts.values())


def test_deterministic():
    assert build_directed_prompts(seed=0) == build_directed_prompts(seed=0)
    assert build_directed_heldout(seed=0) == build_directed_heldout(seed=0)
    assert build_directed_prompts(seed=0) != build_directed_prompts(seed=1)


def test_count_and_balance():
    train = build_directed_prompts()
    assert len(train) == 256
    assert _spread(Counter(p.tier for p in train)) <= 1
    assert set(Counter(p.family for p in train).values()) == {256 // 4}
    assert set(Counter(p.colour for p in train).values()) == {256 // len(COLOURS)}
    assert set(Counter(p.style for p in train).values()) == {256 // len(STYLES)}
    per_subject = Counter(p.subject for p in train)
    assert len(per_subject) == len(SUBJECTS) - len(NOVEL_SUBJECTS)
    assert _spread(per_subject) <= 1
    for tier in DIRECTED_TIERS:
        in_tier = [p for p in train if p.tier == tier]
        assert _spread(Counter(p.family for p in in_tier)) <= 2
        assert _spread(Counter(p.colour for p in in_tier)) <= 2
        assert _spread(Counter(p.style for p in in_tier)) <= 2
        # every phrasing of the tier gets used, none much more than another
        used = Counter(p.id[:5] for p in in_tier)
        assert len(used) == sum(d.tier == tier for d in TRAIN_DIRECTIONS)
        assert _spread(used) <= 2
    held = build_directed_heldout()
    assert len(held) == 16
    assert _spread(Counter(p.tier for p in held)) <= 1
    assert set(Counter(p.style for p in held).values()) == {16 // len(STYLES)}
    # 16 prompts, 20 reserved phrasings: no phrasing twice
    assert len({p.id[:5] for p in held}) == 16


def test_ids_are_unique():
    train, held = build_directed_prompts(), build_directed_heldout(n=40)
    base = {p.id for p in build_prompt_pool()}
    ids = [p.id for p in train + held]
    assert len(set(ids)) == len(ids)
    assert not set(ids) & base
    assert all(i.startswith("d") for i in ids)


def test_no_novel_subject_words():
    # Words only a novel subject uses (its noun and anything else no
    # trainable subject or style says) never appear in a directed prompt.
    novel = set().union(*(_words(s) for s in NOVEL_SUBJECTS))
    seen = set().union(
        *(_words(s.phrase) for s in SUBJECTS if s.phrase not in NOVEL_SUBJECTS),
        *(_words(s) for s in STYLES),
    )
    banned = novel - seen
    assert {"hydrangea", "cat", "pear", "lake", "mountain"} <= banned

    def hits(text: str) -> set[str]:
        stems = {w[:-1] for w in _words(text) if w.endswith("s")}
        stems |= {w[:-2] for w in _words(text) if w.endswith("es")}
        return (_words(text) | stems) & banned

    for d in ALL_DIRECTIONS:
        for colour in COLOURS:
            assert not hits(d.render(colour)), (d.text, hits(d.render(colour)))
    for p in build_directed_prompts() + build_directed_heldout(n=40):
        assert p.subject not in NOVEL_SUBJECTS
        assert not hits(p.text), p.text


def test_no_heldout_triples():
    _, test = build_prompt_splits(seed=0)
    held = {(p.subject, p.colour, p.style) for p in test}
    for p in build_directed_prompts() + build_directed_heldout(n=40):
        assert (p.subject, p.colour, p.style) not in held, p.id
    # an explicit exclude list is honoured too
    first = build_directed_prompts(n=100, seed=3)
    again = build_directed_prompts(n=100, seed=3, exclude=first)
    assert not {(p.subject, p.colour, p.style) for p in first} & {
        (p.subject, p.colour, p.style) for p in again
    }


def test_train_and_heldout_phrasings_are_disjoint():
    assert not {d.text for d in TRAIN_DIRECTIONS} & {d.text for d in HELDOUT_DIRECTIONS}
    trained = {d.render(c) for d in TRAIN_DIRECTIONS for c in COLOURS}
    for p in build_directed_heldout(n=40):
        assert not any(p.text.endswith(" " + t) for t in trained), p.text


def test_tiers_and_kinds():
    for d in ALL_DIRECTIONS:
        assert d.tier in DIRECTED_TIERS
        if d.tier == 8:
            assert 2 <= len(d.parts) <= 3 and len(set(d.parts)) == len(d.parts)
            assert set(d.parts) <= {4, 5, 6, 7}
        else:
            assert d.parts == ()
    train, held = build_directed_prompts(), build_directed_heldout()
    assert all(4 <= p.tier <= 8 for p in train + held)
    assert all(p.kind == "train" for p in train)
    assert all(p.kind == "novel_direction" for p in held)


def test_phrasings_are_curated_and_usable():
    for tier in DIRECTED_TIERS:
        assert 15 <= sum(d.tier == tier for d in TRAIN_DIRECTIONS) <= 25
        assert sum(d.tier == tier for d in HELDOUT_DIRECTIONS) >= 4
    assert len({d.text for d in ALL_DIRECTIONS}) == len(ALL_DIRECTIONS)
    trainable = [p for p in build_prompt_pool() if p.subject not in NOVEL_SUBJECTS]
    for d in ALL_DIRECTIONS:
        for colour in COLOURS:
            assert 15 <= len(d.render(colour).split()) <= 35, d.text
        assert any(d.fits(p) for p in trainable), d.text
    # the subject lists that gate directions name real subjects
    phrases = {s.phrase for s in SUBJECTS}
    assert phrases >= _NO_SKY | _BY_WATER | _FROM_ABOVE | _LIGHT_SET


def test_directed_off_reproduces_the_base_split():
    assert build_run_prompts(n_train=256, n_test=16, seed=0) == build_prompt_splits(seed=0)
    builder = PaintRLDatasetBuilder(
        model_name_for_tokenizer="m", renderer_name="r", batch_size=8, group_size=5, n_batches=1
    )
    assert builder.n_directed_prompts == 0 and builder.n_directed_test == 0


def test_directed_prompts_are_appended():
    base_train, base_test = build_prompt_splits(seed=0)
    train, test = build_run_prompts(256, 16, 0, n_directed=256, n_directed_test=16)
    assert train[:256] == base_train and test[:16] == base_test
    assert len(train) == 512 and len(test) == 32
    assert {p.kind for p in test[16:]} == {"novel_direction"}
    assert len({p.id for p in train + test}) == len(train) + len(test)


def _tags(prompt: PaintPrompt, split: str) -> list[str]:
    return PaintEnvGroupBuilder(
        prompt=prompt,
        renderer=cast(Renderer, None),
        system="",
        policy_effort=0.7,
        group_size=1,
        canvas_size=512,
        verifier_config=VerifierConfig(),
        evaluator_config=None,
        weights=RewardWeights(),
        artifact_dir=None,
        split=split,
        iteration=0,
    ).logging_tags()


def test_logging_tags_keep_base_aggregates():
    train, test = build_run_prompts(256, 16, 0, n_directed=5, n_directed_test=5)
    # base prompts are tagged as before
    assert _tags(train[0], "train") == ["paint", "train", f"tier{train[0].tier}"]
    assert _tags(test[0], "test") == ["paint", "test", f"tier{test[0].tier}", "novel_subject"]
    # directed ones swap the split tag for "directed", so env/train/... and
    # test/env/test/... still cover only the base prompts
    assert _tags(train[-1], "train") == ["paint", "directed", f"tier{train[-1].tier}"]
    assert _tags(test[-1], "test") == [
        "paint",
        "directed",
        f"tier{test[-1].tier}",
        "novel_direction",
    ]
