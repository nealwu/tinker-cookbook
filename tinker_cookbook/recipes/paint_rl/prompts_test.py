from collections import Counter

from tinker_cookbook.recipes.paint_rl.prompts import (
    COLOURS,
    NOVEL_SUBJECTS,
    STYLES,
    SUBJECTS,
    build_prompt_pool,
    build_prompt_splits,
    system_prompt,
)

# The held-out set every arm of the verifier-compute experiment is evaluated
# on (seed 0, 8 novel-subject + 8 novel-combo). Pinned on purpose: an edit to
# the pool or the splitter that moves these prompts makes runs incomparable,
# so it has to be a deliberate change here as well.
PINNED_HELDOUT_SEED0 = [
    ("s25c01t3", "Paint a coral pear in layered watercolor glazes."),
    (
        "s10c07t1",
        "Paint a sepia hydrangea bush beside a white picket fence in minimal watercolor with lots of white paper.",
    ),
    ("s17c02t0", "Paint a golden cat asleep on a windowsill in wet-on-wet watercolor wash."),
    ("s40c04t2", "Paint a teal mountain lake at dawn in watercolor with soft ink outlines."),
    ("s25c00t0", "Paint a crimson pear in wet-on-wet watercolor wash."),
    (
        "s10c06t2",
        "Paint a violet hydrangea bush beside a white picket fence in watercolor with soft ink outlines.",
    ),
    ("s17c05t3", "Paint an indigo cat asleep on a windowsill in layered watercolor glazes."),
    (
        "s40c03t1",
        "Paint an emerald mountain lake at dawn in minimal watercolor with lots of white paper.",
    ),
    ("s01c00t0", "Paint a crimson rose in wet-on-wet watercolor wash."),
    (
        "s16c07t2",
        "Paint a sepia heron standing in shallow water in watercolor with soft ink outlines.",
    ),
    (
        "s35c01t3",
        "Paint a coral straw hat hanging on a chair beside an open window in layered watercolor glazes.",
    ),
    (
        "s45c06t1",
        "Paint a violet lighthouse with a sailboat passing in front of it in minimal watercolor with lots of white paper.",
    ),
    ("s01c05t3", "Paint an indigo rose in layered watercolor glazes."),
    (
        "s16c02t1",
        "Paint a golden heron standing in shallow water in minimal watercolor with lots of white paper.",
    ),
    ("s28c03t0", "Paint an emerald cup of coffee on a wooden table in wet-on-wet watercolor wash."),
    ("s39c04t2", "Paint a teal barn in watercolor with soft ink outlines."),
]


def test_pool_shape():
    pool = build_prompt_pool()
    assert len(pool) == len(SUBJECTS) * len(COLOURS) * len(STYLES) == 1536
    assert len({p.id for p in pool}) == len(pool)
    assert Counter(s.family for s in SUBJECTS) == dict.fromkeys(
        ("flower", "animal", "object", "scene"), 12
    )
    assert Counter(s.tier for s in SUBJECTS) == {1: 16, 2: 20, 3: 12}


def test_articles():
    pool = build_prompt_pool()
    for p in pool:
        article = p.text.split()[1]
        assert article == ("an" if p.colour[0] in "aeiou" else "a"), p.text


def test_novel_subject_nouns_appear_nowhere_else():
    phrases = {s.phrase for s in SUBJECTS}
    assert phrases >= NOVEL_SUBJECTS
    for novel in NOVEL_SUBJECTS:
        noun = novel.split()[0]
        others = [s.phrase for s in SUBJECTS if s.phrase != novel]
        assert not any(noun in o.split() for o in others), (noun, others)


def test_default_split_is_pinned():
    _, test = build_prompt_splits(seed=0)
    assert [(p.id, p.text) for p in test] == PINNED_HELDOUT_SEED0
    kinds = [p.kind for p in test]
    assert kinds == ["novel_subject"] * 8 + ["novel_combo"] * 8


def test_split_invariants():
    train, test = build_prompt_splits(seed=0)
    train2, test2 = build_prompt_splits(seed=0)
    assert [p.id for p in train] == [p.id for p in train2]
    assert [p.id for p in test] == [p.id for p in test2]
    assert not {p.id for p in train} & {p.id for p in test}
    assert len(train) == 256
    # novel subjects never appear in training in any form
    assert not any(p.subject in NOVEL_SUBJECTS for p in train)
    # training is balanced over the 25 remaining subjects, colours and styles
    per_subject = Counter(p.subject for p in train)
    assert len(per_subject) == len(SUBJECTS) - len(NOVEL_SUBJECTS)
    assert max(per_subject.values()) - min(per_subject.values()) <= 1
    assert set(Counter(p.colour for p in train).values()) == {256 // len(COLOURS)}
    assert set(Counter(p.style for p in train).values()) == {256 // len(STYLES)}
    # each held-out half covers every colour once and every style twice
    for kind in ("novel_subject", "novel_combo"):
        half = [p for p in test if p.kind == kind]
        assert len(half) == 8
        assert Counter(p.colour for p in half) == dict.fromkeys(COLOURS, 1)
        assert Counter(p.style for p in half) == dict.fromkeys(STYLES, 2)
    # every training prompt is tagged train
    assert all(p.kind == "train" for p in train)


def test_other_seeds_and_sizes_work():
    train, test = build_prompt_splits(
        n_train=40, n_test_novel_subject=5, n_test_novel_combo=3, seed=7
    )
    assert len(train) == 40 and len(test) == 8
    assert not {p.id for p in train} & {p.id for p in test}


def test_system_prompt_mentions_canvas_size():
    text = system_prompt(512)
    assert "createCanvas(512, 512, WEBGL)" in text
    assert "-256" in text and "{size}" not in text
