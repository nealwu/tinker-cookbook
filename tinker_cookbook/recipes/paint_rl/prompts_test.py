from collections import Counter

from tinker_cookbook.recipes.paint_rl.prompts import (
    COLORS,
    N_TEST_PROMPTS,
    N_TRAIN_PROMPTS,
    SKILL_PROMPTS,
    STYLES,
    SUBJECTS,
    build_prompt_pool,
    build_prompt_splits,
    system_prompt,
)

# The held-out set every arm (verifier and judge) is evaluated on (seed 0,
# 50 prompts spread over every subject). Pinned on purpose: an edit to
# the pool or the splitter that moves these prompts makes runs incomparable,
# so it has to be a deliberate change here as well.
PINNED_HELDOUT_SEED0 = [
    ("s27c06t0", "Paint a violet windmill in wet-on-wet watercolor wash."),
    (
        "s12c03t1",
        "Paint an emerald field of flowers on a rolling hillside in minimal watercolor with lots of white paper.",
    ),
    ("s42c07t3", "Paint a sepia fish swimming in a clear stream in layered watercolor glazes."),
    (
        "s40c02t2",
        "Paint a golden forest path in dappled sunlight in watercolor with soft ink outlines.",
    ),
    (
        "s41c05t2",
        "Paint an indigo flock of birds in flight across the sky in watercolor with soft ink outlines.",
    ),
    (
        "s07c00t1",
        "Paint a crimson magnolia branch in a glass vase in minimal watercolor with lots of white paper.",
    ),
    (
        "s05c04t0",
        "Paint a teal cherry blossom branch against a pale sky in wet-on-wet watercolor wash.",
    ),
    (
        "s35c01t3",
        "Paint a coral village street leading to a church under a crescent moon in layered watercolor glazes.",
    ),
    ("s01c05t0", "Paint an indigo rose in wet-on-wet watercolor wash."),
    (
        "k06t3",
        "Paint a winding road that becomes narrower toward the horizon, in layered watercolor glazes.",
    ),
    ("s39c04t2", "Paint a teal Golden Gate Bridge in watercolor with soft ink outlines."),
    ("s00c03t1", "Paint an emerald sunflower in minimal watercolor with lots of white paper."),
    (
        "s04c01t0",
        "Paint a coral lotus blossom floating on a still pond in wet-on-wet watercolor wash.",
    ),
    (
        "s34c06t3",
        "Paint a violet lighthouse with a sailboat passing in front of it in layered watercolor glazes.",
    ),
    (
        "s20c07t2",
        "Paint a sepia bicycle leaning against a brick wall in watercolor with soft ink outlines.",
    ),
    (
        "s14c02t1",
        "Paint a golden field of flowers with a path leading to a distant cottage in minimal watercolor with lots of white paper.",
    ),
    (
        "k07t0",
        "Paint a row of trees along a canal, shrinking into the distance, in wet-on-wet watercolor wash.",
    ),
    ("s28c00t1", "Paint a crimson cottage in minimal watercolor with lots of white paper."),
    ("s33c02t3", "Paint a golden birch forest in autumn in layered watercolor glazes."),
    ("k00t2", "Paint exactly five red tulips in a vase, in watercolor with soft ink outlines."),
    (
        "s38c04t2",
        "Paint a teal creepypasta figure lurking at the edge of a foggy forest in watercolor with soft ink outlines.",
    ),
    ("s11c07t3", "Paint a sepia field of flowers in layered watercolor glazes."),
    ("s29c05t1", "Paint an indigo barn in minimal watercolor with lots of white paper."),
    ("s17c03t0", "Paint an emerald watering can in wet-on-wet watercolor wash."),
    ("s15c00t1", "Paint a crimson pear in minimal watercolor with lots of white paper."),
    (
        "s10c06t0",
        "Paint a violet clump of daffodils growing beside a stone birdbath in wet-on-wet watercolor wash.",
    ),
    (
        "s21c01t2",
        "Paint a coral stack of old books on a windowsill in watercolor with soft ink outlines.",
    ),
    (
        "k04t3",
        "Paint a round ceramic bowl with a narrow base and wide opening, in layered watercolor glazes.",
    ),
    (
        "s23c04t2",
        "Paint a teal bowl of lemons beside a glass bottle in watercolor with soft ink outlines.",
    ),
    ("s03c07t3", "Paint a sepia daisy in layered watercolor glazes."),
    ("k01t0", "Paint exactly three white sailboats on a calm lake, in wet-on-wet watercolor wash."),
    (
        "s09c00t1",
        "Paint a crimson wisteria vine trailing over a garden gate in minimal watercolor with lots of white paper.",
    ),
    (
        "k03t1",
        "Paint a red apple on top of a stack of two green books, in minimal watercolor with lots of white paper.",
    ),
    (
        "s06c01t2",
        "Paint a coral poppy field under a summer sky in watercolor with soft ink outlines.",
    ),
    (
        "s36c03t3",
        "Paint an emerald stone bridge arching over a quiet river in layered watercolor glazes.",
    ),
    ("s18c02t0", "Paint a golden cup of coffee on a wooden table in wet-on-wet watercolor wash."),
    (
        "s08c05t1",
        "Paint an indigo bouquet of peonies in a ceramic jug in minimal watercolor with lots of white paper.",
    ),
    ("k02t3", "Paint a blue vase to the left of a yellow teacup, in layered watercolor glazes."),
    ("s13c06t2", "Paint a violet field of flowers at sunset in watercolor with soft ink outlines."),
    (
        "s37c04t0",
        "Paint a teal classical building with columns and arched windows in wet-on-wet watercolor wash.",
    ),
    (
        "s22c07t0",
        "Paint a sepia basket of apples on a checked tablecloth in wet-on-wet watercolor wash.",
    ),
    ("s30c03t2", "Paint an emerald mountain lake at dawn in watercolor with soft ink outlines."),
    (
        "s19c05t1",
        "Paint an indigo lantern hanging in a stone doorway in minimal watercolor with lots of white paper.",
    ),
    ("s25c06t3", "Paint a violet mailbox in layered watercolor glazes."),
    ("s31c02t0", "Paint a golden old oak tree alone in a meadow in wet-on-wet watercolor wash."),
    ("s32c01t2", "Paint a coral sailboat on a calm sea in watercolor with soft ink outlines."),
    ("s16c00t1", "Paint a crimson hot air balloon in minimal watercolor with lots of white paper."),
    ("s02c06t3", "Paint a violet tulip in layered watercolor glazes."),
    ("s26c05t1", "Paint an indigo lighthouse in minimal watercolor with lots of white paper."),
    (
        "k05t2",
        "Paint a diamond-shaped kite with a long ribbon tail, in watercolor with soft ink outlines.",
    ),
]


def test_pool_shape():
    pool = build_prompt_pool()
    # one color is left out for the creepypasta figure and the Golden Gate
    # Bridge; skill prompts are crossed with the styles only
    base = len(SUBJECTS) * len(COLORS) * len(STYLES) - 2 * len(STYLES)
    assert len(pool) == base + len(SKILL_PROMPTS) * len(STYLES) == 1400
    assert len({p.id for p in pool}) == len(pool)
    assert len({s.phrase for s in SUBJECTS}) == len(SUBJECTS)
    assert Counter(s.family for s in SUBJECTS) == {
        "flower": 15, "object": 11, "scene": 15, "animal": 2,
    }  # fmt: skip
    assert Counter(s.tier for s in SUBJECTS) == {1: 14, 2: 20, 3: 9}


def test_animal_family():
    assert {s.name for s in SUBJECTS if s.family == "animal"} == {
        "flock of birds", "fish",
    }  # fmt: skip


def test_excluded_combos_and_variants():
    pool = build_prompt_pool()
    assert sum(s.name == "field of flowers" for s in SUBJECTS) == 4
    assert not [p for p in pool if p.subject.startswith("creepypasta") and p.color == "indigo"]
    assert len([p for p in pool if p.subject.startswith("creepypasta")]) == 7 * len(STYLES)
    assert not [p for p in pool if "golden Golden" in p.text]


def test_articles():
    pool = build_prompt_pool()
    for p in pool:
        if p.family == "skill":  # worded in full, no color slot
            continue
        article = p.text.split()[1]
        assert article == ("an" if p.color[0] in "aeiou" else "a"), p.text


def _spread(counts: Counter[str]) -> int:
    return max(counts.values()) - min(counts.values())


def test_default_split_is_pinned():
    _, test = build_prompt_splits(seed=0)
    assert [(p.id, p.text) for p in test] == PINNED_HELDOUT_SEED0
    assert all(p.kind == "test" for p in test)


def test_split_invariants():
    train, test = build_prompt_splits(seed=0)
    train2, test2 = build_prompt_splits(seed=0)
    assert [p.id for p in train] == [p.id for p in train2]
    assert [p.id for p in test] == [p.id for p in test2]
    assert not {p.id for p in train} & {p.id for p in test}
    assert len(train) == N_TRAIN_PROMPTS and len(test) == N_TEST_PROMPTS
    # no held-out subjects: every subject and skill prompt is trained on, and
    # the 50 held-out prompts are 50 different ones of them
    everything = {s.phrase for s in SUBJECTS} | {k.request for k in SKILL_PROMPTS}
    assert {p.subject for p in train} == everything
    assert len({p.subject for p in test}) == min(N_TEST_PROMPTS, len(everything))
    # the simple subject base is balanced over subjects, colors and styles
    for prompts in (train, test):
        base = [p for p in prompts if p.family != "skill"]
        assert _spread(Counter(p.subject for p in base)) <= 1
        colors = Counter(p.color for p in base)
        styles = Counter(p.style for p in prompts)
        assert set(colors) == set(COLORS) and _spread(colors) <= 1
        assert set(styles) == set(STYLES) and _spread(styles) <= 2
    assert all(p.kind == "train" for p in train)


def test_skill_prompts_are_a_small_share():
    train, test = build_prompt_splits(seed=0)
    skill_train = [p for p in train if p.family == "skill"]
    assert 0.05 <= len(skill_train) / len(train) <= 0.12
    assert {k.category for k in SKILL_PROMPTS} == {
        "object counting", "spatial relationships", "shape and geometry", "depth and perspective",
    }  # fmt: skip
    for p in build_prompt_pool():
        if p.family == "skill":
            assert p.color == "" and p.text.endswith(f", in {p.style}."), p.text
            assert not any(
                p.text.startswith(f"Paint {a} {c} ") for a in ("a", "an") for c in COLORS
            )


def test_other_seeds_and_sizes_work():
    train, test = build_prompt_splits(n_train=40, n_test=8, seed=7)
    assert len(train) == 40 and len(test) == 8
    assert not {p.id for p in train} & {p.id for p in test}


def test_system_prompt_mentions_canvas_size():
    text = system_prompt(512)
    assert "createCanvas(512, 512, WEBGL)" in text
    assert "-256" in text and "{size}" not in text


def test_system_prompt_does_not_ask_for_noloop():
    # The page stops the draw loop itself; a noLoop() rule invites brush.noLoop().
    assert "noLoop" not in system_prompt(512)
