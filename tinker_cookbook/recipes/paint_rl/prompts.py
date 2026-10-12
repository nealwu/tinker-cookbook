"""Policy system prompt and the painting-prompt dataset.

The system prompt follows the lesson from the "paint with code" project:
a short, strict allowlist of brush calls constrains the model better than a
long API reference (which mostly produced hallucinated APIs).

The prompt pool is ``subject x colour x style``. It is built so the verifier
has something to resolve and the held-out set measures generalisation rather
than recombination:

* **Subjects are what watercolorists commonly paint**: flowers, animals,
  still-life objects and scenes, twelve of each, in three difficulty tiers:
  tier 1 a single subject, tier 2 a subject in a setting, tier 3 two named
  elements in a spatial relation. Flowers get an equal share; a vague flower
  is the easiest blob to pass off, so the prompt-adherence criterion asks
  for an unmistakable, specific depiction.
* **Colours and styles are visually separable** at 448 px (no golden/ochre,
  no rose-pink/coral/crimson, no loose/splashy/wet-on-wet near-synonyms), so
  the colour- and style-fidelity criteria score signal rather than noise.
* **Two held-out tiers.** ``novel_subject`` prompts use subjects that never
  appear in training in any form; ``novel_combo`` prompts use seen subjects,
  colours and styles in an unseen triple. Training prompts are balanced over
  subjects so no subject is memorised more than another.

The split is deterministic in the seed and pinned by ``prompts_test.py``: a
change to the pool that moves the held-out set fails the test on purpose.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar

CANVAS_SIZE_PLACEHOLDER = "{size}"

SYSTEM_PROMPT_TEMPLATE = """You are a generative artist who paints watercolors with code, using p5.js and the p5.brush library.

Write ONE complete JavaScript sketch that paints the requested subject. Reply with a single ```javascript code block and nothing else.

Hard rules:
- Define `function setup()` only. No draw() loop, no async code, no images, fonts, fetch, DOM, or external files.
- setup() must start with exactly: createCanvas({size}, {size}, WEBGL); angleMode(DEGREES); background(<paper colour>); brush.load();
- setup() must end with noLoop();
- WEBGL coordinates: (0, 0) is the CENTRE of the canvas. x and y run from -{half} to {half}; y grows downwards.
- Use ONLY these brush functions (any other brush.* call throws and the painting is lost):
    brush.pick(name)                 name in "2B","HB","2H","cpencil","pen","rotring","spray","marker","marker2","charcoal"
    brush.stroke(css_colour)  brush.noStroke()  brush.strokeWeight(multiplier)
    brush.fill(css_colour, opacity_0_to_255)  brush.noFill()
    brush.bleed(strength_0_to_0.5, "out" or "in")  brush.fillTexture(texture_0_to_1, border_0_to_1)
    brush.line(x1, y1, x2, y2)  brush.circle(x, y, radius)  brush.rect(x, y, w, h)
    brush.polygon([[x, y], ...])  brush.spline([[x, y], ...], curvature_0_to_1)
    brush.beginShape(curvature_0_to_1); brush.vertex(x, y); ...; brush.endShape(CLOSE)
    brush.hatch(spacing, angle_degrees)  brush.noHatch()
    brush.field(name)                name in "curved","truncated","zigzag","seabed","waves"   brush.noField()
    brush.push()  brush.pop()  brush.seed(number)
- List the vertices of every filled shape CLOCKWISE, otherwise the watercolor bleed inverts.
- Plain p5 helpers are fine for maths and colour: random, noise, sin, cos, map, lerp, constrain, color, lerpColor, PI.

Painting advice: build the image from several translucent washes (fill opacity 40-120 with bleed 0.1-0.4), let colours overlap and pool, vary edge softness, keep pencil or marker outlines sparse and broken, and fill most of the canvas with a clear, recognisable subject.
"""


def system_prompt(canvas_size: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.replace("{size}", str(canvas_size)).replace(
        "{half}", str(canvas_size // 2)
    )


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

Family = Literal["flower", "animal", "object", "scene"]
PromptKind = Literal["train", "novel_subject", "novel_combo", "novel_direction"]


@dataclass(frozen=True)
class Subject:
    """``name`` is the noun the colour word attaches to; ``context`` is the rest."""

    name: str
    context: str
    tier: int
    family: Family

    @property
    def phrase(self) -> str:
        return f"{self.name} {self.context}".strip()


# The four families watercolorists paint most: flowers, animals, still-life
# objects and scenes, twelve each, every family spanning all three tiers
# (four single subjects, five in a setting, three pairs).
SUBJECTS: tuple[Subject, ...] = (
    # flowers
    Subject("sunflower", "", 1, "flower"),
    Subject("rose", "", 1, "flower"),
    Subject("tulip", "", 1, "flower"),
    Subject("daisy", "", 1, "flower"),
    Subject("lotus blossom", "floating on a still pond", 2, "flower"),
    Subject("cherry blossom branch", "against a pale sky", 2, "flower"),
    Subject("poppy field", "under a summer sky", 2, "flower"),
    Subject("magnolia branch", "in a glass vase", 2, "flower"),
    Subject("bouquet of peonies", "in a ceramic jug", 2, "flower"),
    Subject("wisteria vine", "trailing over a garden gate", 3, "flower"),
    Subject("hydrangea bush", "beside a white picket fence", 3, "flower"),
    Subject("clump of daffodils", "growing beside a stone birdbath", 3, "flower"),
    # animals
    Subject("fox", "", 1, "animal"),
    Subject("owl", "", 1, "animal"),
    Subject("koi fish", "", 1, "animal"),
    Subject("robin", "", 1, "animal"),
    Subject("heron", "standing in shallow water", 2, "animal"),
    Subject("cat", "asleep on a windowsill", 2, "animal"),
    Subject("deer", "grazing in a misty meadow", 2, "animal"),
    Subject("rabbit", "sitting in tall grass", 2, "animal"),
    Subject("swan", "gliding across a river", 2, "animal"),
    Subject("whale", "breaching next to a tiny rowing boat", 3, "animal"),
    Subject("hummingbird", "hovering beside a trumpet flower", 3, "animal"),
    Subject("butterfly", "perched on a thistle", 3, "animal"),
    # still-life objects
    Subject("teapot", "", 1, "object"),
    Subject("pear", "", 1, "object"),
    Subject("hot air balloon", "", 1, "object"),
    Subject("watering can", "", 1, "object"),
    Subject("cup of coffee", "on a wooden table", 2, "object"),
    Subject("lantern", "hanging in a stone doorway", 2, "object"),
    Subject("bicycle", "leaning against a brick wall", 2, "object"),
    Subject("stack of old books", "on a windowsill", 2, "object"),
    Subject("basket of apples", "on a checked tablecloth", 2, "object"),
    Subject("teapot", "pouring tea into a cup", 3, "object"),
    Subject("bowl of lemons", "beside a glass bottle", 3, "object"),
    Subject("straw hat", "hanging on a chair beside an open window", 3, "object"),
    # scenes
    Subject("lighthouse", "", 1, "scene"),
    Subject("windmill", "", 1, "scene"),
    Subject("cottage", "", 1, "scene"),
    Subject("barn", "", 1, "scene"),
    Subject("mountain lake", "at dawn", 2, "scene"),
    Subject("old oak tree", "alone in a meadow", 2, "scene"),
    Subject("sailboat", "on a calm sea", 2, "scene"),
    Subject("fishing village", "at sunset", 2, "scene"),
    Subject("birch forest", "in autumn", 2, "scene"),
    Subject("lighthouse", "with a sailboat passing in front of it", 3, "scene"),
    Subject("village street", "leading to a church under a crescent moon", 3, "scene"),
    Subject("stone bridge", "arching over a quiet river", 3, "scene"),
)

# Held out of training entirely: one subject per family, spanning the tiers,
# each chosen so its noun appears in no other subject.
NOVEL_SUBJECTS: frozenset[str] = frozenset(
    {
        "hydrangea bush beside a white picket fence",
        "cat asleep on a windowsill",
        "pear",
        "mountain lake at dawn",
    }
)

# Eight hues that stay distinct in a 448 px watercolor.
COLOURS: tuple[str, ...] = (
    "crimson",
    "coral",
    "golden",
    "emerald",
    "teal",
    "indigo",
    "violet",
    "sepia",
)

# Four styles, each with a visual signature a reviewer can check.
STYLES: tuple[str, ...] = (
    "wet-on-wet watercolor wash",
    "minimal watercolor with lots of white paper",
    "watercolor with soft ink outlines",
    "layered watercolor glazes",
)


@dataclass(frozen=True)
class PaintPrompt:
    """One painting request. ``id`` names the artifact folder."""

    id: str
    text: str
    subject: str
    tier: int
    family: str
    colour: str
    style: str
    kind: PromptKind = "train"

    def with_kind(self, kind: PromptKind) -> PaintPrompt:
        return PaintPrompt(**{**asdict(self), "kind": kind})


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def build_prompt_pool() -> list[PaintPrompt]:
    """Every (subject, colour, style) combination, in a fixed order."""
    prompts: list[PaintPrompt] = []
    for si, subject in enumerate(SUBJECTS):
        for ci, colour in enumerate(COLOURS):
            for ti, style in enumerate(STYLES):
                text = f"Paint {_article(colour)} {colour} {subject.phrase} in {style}."
                prompts.append(
                    PaintPrompt(
                        id=f"s{si:02d}c{ci:02d}t{ti}",
                        text=text,
                        subject=subject.phrase,
                        tier=subject.tier,
                        family=subject.family,
                        colour=colour,
                        style=style,
                    )
                )
    return prompts


T = TypeVar("T")


def _stratified_pick(
    items: Iterable[T],
    n: int,
    key: Callable[[T], object],
    rng: random.Random,
    balance: Sequence[Callable[[T], object]] = (),
) -> list[T]:
    """``n`` items spread as evenly as possible over the values of ``key``.

    Each stratum is shuffled, the strata are shuffled, and items are taken
    round-robin, so counts per stratum differ by at most one. Within a
    stratum the item whose ``balance`` attributes (e.g. colour, style) have
    been used least so far is taken, so a small draw still covers every
    colour and style rather than landing three sepias and no ink outlines.
    """
    strata: dict[object, list[T]] = defaultdict(list)
    for item in items:
        strata[key(item)].append(item)
    order = list(strata)
    rng.shuffle(order)
    for k in order:
        rng.shuffle(strata[k])
    used: list[dict[object, int]] = [defaultdict(int) for _ in balance]
    picked: list[T] = []
    while len(picked) < n and any(strata[k] for k in order):
        for k in order:
            if not strata[k] or len(picked) >= n:
                continue
            # Least-used balance attributes first; the shuffle breaks ties.
            best = min(
                range(len(strata[k])),
                key=lambda i: tuple(u[f(strata[k][i])] for u, f in zip(used, balance, strict=True)),
            )
            item = strata[k].pop(best)
            for u, f in zip(used, balance, strict=True):
                u[f(item)] += 1
            picked.append(item)
    if len(picked) < n:
        raise ValueError(f"asked for {n} prompts but only {len(picked)} are available")
    return picked


def build_prompt_splits(
    n_train: int = 256,
    n_test_novel_subject: int = 8,
    n_test_novel_combo: int = 8,
    seed: int = 0,
) -> tuple[list[PaintPrompt], list[PaintPrompt]]:
    """(train, test) prompt lists; test = novel-subject prompts then novel-combo prompts.

    * novel-subject: drawn from :data:`NOVEL_SUBJECTS`, spread over those subjects;
      none of these subjects appears in training.
    * novel-combo: drawn from the remaining subjects, spread over families (then
      colours, styles and tiers); the exact
      triple is removed from the training pool.
    * train: balanced over the remaining subjects.
    """
    rng = random.Random(seed)
    pool = build_prompt_pool()
    novel = [p for p in pool if p.subject in NOVEL_SUBJECTS]
    known = [p for p in pool if p.subject not in NOVEL_SUBJECTS]

    by_look = (lambda p: p.colour, lambda p: p.style)
    test_novel = [
        p.with_kind("novel_subject")
        for p in _stratified_pick(
            novel, n_test_novel_subject, lambda p: p.subject, rng, balance=by_look
        )
    ]
    test_combo = [
        p.with_kind("novel_combo")
        for p in _stratified_pick(
            known,
            n_test_novel_combo,
            lambda p: p.family,
            rng,
            balance=(*by_look, lambda p: p.tier),
        )
    ]
    held = {p.id for p in test_combo}
    train = _stratified_pick(
        [p for p in known if p.id not in held], n_train, lambda p: p.subject, rng, balance=by_look
    )
    return train, test_novel + test_combo


def split_summary(prompts: Sequence[PaintPrompt]) -> dict[str, int]:
    """Counts by kind and tier, for logging."""
    counts: dict[str, int] = defaultdict(int)
    for p in prompts:
        counts[p.kind] += 1
        counts[f"{p.kind}/tier{p.tier}"] += 1
    return dict(sorted(counts.items()))
