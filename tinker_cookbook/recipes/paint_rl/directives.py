"""Directed painting prompts: the base prompt plus a watercolourist's directions.

The base pool (``prompts.py``) asks for a subject, a colour and a style. A
directed prompt keeps that sentence and adds one or more short directions of
the kind a watercolour teacher gives, in five tiers that continue the subject
tiers 1-3:

* **tier 4, composition**: placement, cropping and viewpoint;
* **tier 5, palette and value**: a limited palette that includes the prompt's
  colour, high- or low-key, a single warm or cool accent;
* **tier 6, technique**, using only what the brush allowlist can do: washes
  and bleed, ``fillTexture`` granulation, ``hatch``, ``field`` flows, and
  pencil, pen, marker, charcoal and spray marks;
* **tier 7, light and mood**: time of day, weather, direction of the light;
* **tier 8, stacked**: two or three directions from different tiers at once.

Every direction can be checked from the image alone, so the verifier's
prompt-adherence criterion scores it with no change to the verifier. The
phrasings are curated by hand, adapted from the BrushArena / BLOOM directive
work; nothing is generated, and the prompts are deterministic in the seed.

Held-out hygiene follows ``prompts.py``: no subject in ``NOVEL_SUBJECTS`` is
used, no word that only a novel subject uses appears in any direction, and the
(subject, colour, style) triples of the held-out prompts are never reused. A
separate set of phrasings is reserved for :func:`build_directed_heldout`
(kind ``novel_direction``) and never appears in training.
``directives_test.py`` checks all of this.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from tinker_cookbook.recipes.paint_rl.prompts import (
    NOVEL_SUBJECTS,
    STYLES,
    SUBJECTS,
    PaintPrompt,
    PromptKind,
    _stratified_pick,
    build_prompt_pool,
    build_prompt_splits,
)

DIRECTED_TIERS: tuple[int, ...] = (4, 5, 6, 7, 8)

WET, MINIMAL, INK, GLAZES = STYLES

# ---------------------------------------------------------------------------
# What a direction may assume about the base prompt
# ---------------------------------------------------------------------------

# Subjects painted indoors or under water: no sky to place or light.
_NO_SKY: frozenset[str] = frozenset(
    {
        "magnolia branch in a glass vase",
        "bouquet of peonies in a ceramic jug",
        "koi fish",
        "teapot",
        "cup of coffee on a wooden table",
        "lantern hanging in a stone doorway",
        "stack of old books on a windowsill",
        "basket of apples on a checked tablecloth",
        "teapot pouring tea into a cup",
        "bowl of lemons beside a glass bottle",
        "straw hat hanging on a chair beside an open window",
    }
)
# Subjects already beside open water, so a reflection belongs in the picture.
_BY_WATER: frozenset[str] = frozenset(
    {
        "lotus blossom floating on a still pond",
        "heron standing in shallow water",
        "swan gliding across a river",
        "whale breaching next to a tiny rowing boat",
        "sailboat on a calm sea",
        "lighthouse with a sailboat passing in front of it",
        "stone bridge arching over a quiet river",
    }
)
# Subjects that read naturally from directly overhead.
_FROM_ABOVE: frozenset[str] = frozenset(
    {
        "sunflower",
        "daisy",
        "lotus blossom floating on a still pond",
        "koi fish",
        "swan gliding across a river",
        "cup of coffee on a wooden table",
        "basket of apples on a checked tablecloth",
        "teapot pouring tea into a cup",
        "bowl of lemons beside a glass bottle",
        "sailboat on a calm sea",
    }
)
# Subjects whose own wording already sets the light or the sky; tier-7
# directions would contradict it.
_LIGHT_SET: frozenset[str] = frozenset(
    {
        "cherry blossom branch against a pale sky",
        "poppy field under a summer sky",
        "deer grazing in a misty meadow",
        "fishing village at sunset",
        "village street leading to a church under a crescent moon",
    }
)


def _traits(phrase: str) -> frozenset[str]:
    traits: set[str] = set()
    if phrase not in _NO_SKY:
        traits.add("sky")
    if phrase not in _LIGHT_SET:
        traits.add("light")
    if phrase in _BY_WATER:
        traits.add("water")
    if phrase in _FROM_ABOVE:
        traits.add("above")
    return frozenset(traits)


SUBJECT_TRAITS: dict[str, frozenset[str]] = {s.phrase: _traits(s.phrase) for s in SUBJECTS}

WARM_COLOURS: frozenset[str] = frozenset({"crimson", "coral", "golden", "sepia"})

# The second colour of a limited palette: always the other temperature, so a
# two-colour palette has both warm and cool to mix from.
PALETTE_PARTNER: dict[str, str] = {
    "crimson": "indigo",
    "coral": "teal",
    "golden": "violet",
    "emerald": "sepia",
    "teal": "coral",
    "indigo": "sepia",
    "violet": "golden",
    "sepia": "indigo",
}


@dataclass(frozen=True)
class Direction:
    """One direction appended to a base prompt.

    ``text`` may name the prompt's colour as ``{colour}`` and its palette
    partner as ``{other}``. The other fields keep a direction off base prompts
    it would contradict.
    """

    text: str
    tier: int
    needs: frozenset[str]
    """Subject traits it needs: "sky", "water", "above", "light" (see SUBJECT_TRAITS)."""
    hue: Literal["warm", "cool"] | None
    """Only for prompt colours of this temperature (single-accent palettes)."""
    clashes: frozenset[str]
    """Styles it contradicts, e.g. a full-bleed crop and "lots of white paper"."""
    parts: tuple[int, ...]
    """Tier 8 only: the tiers it combines."""

    def fits(self, prompt: PaintPrompt) -> bool:
        warmth = "warm" if prompt.colour in WARM_COLOURS else "cool"
        return (
            self.needs <= SUBJECT_TRAITS[prompt.subject]
            and self.hue in (None, warmth)
            and prompt.style not in self.clashes
        )

    def render(self, colour: str) -> str:
        return self.text.format(colour=colour, other=PALETTE_PARTNER[colour])


def _d(
    tier: int,
    text: str,
    needs: str = "",
    hue: Literal["warm", "cool"] | None = None,
    clashes: Sequence[str] = (),
    parts: tuple[int, ...] = (),
) -> Direction:
    # Anything that sets the light needs a subject that leaves it open.
    light = {"light"} if 7 in (tier, *parts) else set()
    return Direction(text, tier, frozenset(needs.split()) | light, hue, frozenset(clashes), parts)


# ---------------------------------------------------------------------------
# Training directions, about 18 per tier
# ---------------------------------------------------------------------------

TRAIN_DIRECTIONS: tuple[Direction, ...] = (
    # tier 4: composition, placement, viewpoint
    _d(4, "Place it small in the lower third, under a wide, empty sky laid in as one soft graded wash.", "sky"),
    _d(4, "Crop in tight for a close-up: it fills the frame and runs off at least two edges of the page.", clashes=[MINIMAL]),
    _d(4, "Seen from a low viewpoint looking up, so it rises tall against the sky with the horizon near the bottom edge.", "sky"),
    _d(4, "Set it off-centre to the right, with generous untouched white paper on the left as breathing room."),
    _d(4, "Set it off-centre to the left, and leave the right third of the page as quiet, barely tinted paper."),
    _d(4, "Seen from high above, looking straight down on it, so its top shape reads clearly against the surface below.", "above"),
    _d(4, "Place it dead centre in a calm, symmetrical layout, with equal open space on either side of it."),
    _d(4, "Paint it as a vignette: the washes fade out softly towards the edges and the four corners stay bare white paper."),
    _d(4, "Lay it along a strong diagonal that runs from the bottom left corner up towards the top right of the page."),
    _d(4, "Put it far off in the middle distance, small and pale, with a broad, loosely painted foreground leading the eye to it.", "sky"),
    _d(4, "Fill the page edge to edge with it and its surroundings, leaving no white margin anywhere.", clashes=[MINIMAL]),
    _d(4, "Show it reflected upside down in still water across the lower half of the page, the reflection softer than the original.", "water"),
    _d(4, "Place a high horizon near the top edge, so the ground or water in front of it fills most of the page.", "sky"),
    _d(4, "Frame it with darker, loosely painted foliage in the two bottom corners, the subject lighter and clearer in the middle.", "sky"),
    _d(4, "Keep it at eye level and side-on, stretched across the middle of the page with calm bands of paper above and below."),
    _d(4, "Place it in the upper half of the page, with a long, empty stretch of ground or table running towards us below it."),
    _d(4, "Make it large and close in the bottom right, with a small, faint distant view opening up behind it on the upper left.", "sky"),
    _d(4, "Frame it through an opening, a doorway, a window or a gap in branches, painted as a dark border around the edges."),
    # tier 5: palette and value
    _d(5, "Use a limited palette of just {colour} and {other}, mixing the two for the darks, with the white paper as the only light."),
    _d(5, "Keep to three colours only: {colour}, {other} and a warm grey, mixed on the page rather than added as new hues."),
    _d(5, "Paint it entirely in tones of {colour}, a single-colour value study running from the palest tint to the deepest shade."),
    _d(5, "Keep it high-key: pale, airy washes almost everywhere, with only a few small touches of darker value."),
    _d(5, "Keep it low-key: deep, dark washes over most of the page, with the subject picked out by a few light passages.", clashes=[MINIMAL]),
    _d(5, "Let the {colour} be the only warm note in an otherwise cool picture of soft, quiet blue-greys.", hue="warm"),
    _d(5, "Keep the whole picture cool, with one small {other} accent as the only warm note in it.", hue="cool"),
    _d(5, "Use strong value contrast: the darkest darks sit right against the lightest lights at the centre of interest."),
    _d(5, "Keep every value in a narrow, muted middle range, soft and close in tone, with no pure darks and no bright whites.", clashes=[MINIMAL]),
    _d(5, "Keep the colours clean and unmixed, with {colour} and {other} side by side in separate, fresh washes."),
    _d(5, "Set the {colour} against a quiet {other} background wash, so the two colours push against each other."),
    _d(5, "Grey down everything around the subject so only the subject keeps a clear, saturated {colour}; the rest stays muted."),
    _d(5, "Surround the {colour} with warm, earthy browns and rusts, like a page from an old travel sketchbook.", hue="cool"),
    _d(5, "Surround the {colour} with cool slate blues and greys, so it glows against them like a lamp.", hue="warm"),
    _d(5, "Paint it only in pale, chalky pastel tints of {colour} and {other}, softened with plenty of water."),
    _d(5, "Build the darks from mixed {colour} and {other}, never black, so even the deepest shadows carry colour."),
    _d(5, "Keep the {colour} for the subject alone, and paint everything else in neutral greys and the bare paper."),
    _d(5, "Let the background go nearly black with deep washes, so the {colour} subject shines out of the dark.", clashes=[MINIMAL]),
    # tier 6: technique, only what the brush allowlist can do
    _d(6, "Contrast soft, bleeding wet-on-wet washes for the background with crisp, fine pencil details on the subject itself."),
    _d(6, "Shade the shadows with fine hatched lines over the washes, the hatching closer together where the shadow is darkest."),
    _d(6, "Let flowing brushstrokes follow a gentle curved current across the background, like wind moving through the whole picture."),
    _d(6, "Paint the background with rippling, wave-like strokes that run in parallel bands across the page behind it."),
    _d(6, "Add energy with quick, angular zigzag strokes in the background, against a calmer, smoother subject."),
    _d(6, "Keep the outlines broken and sparse: a few short, interrupted lines suggest the edges, and the washes do the rest."),
    _d(6, "Give the washes a grainy, granulated texture, as if the pigment settled into the paper in tiny speckles."),
    _d(6, "Paint the sky wet-on-wet so its colours run softly into each other, and keep the foreground dry, sharp and hard-edged.", "sky"),
    _d(6, "Leave hard, dark tide-lines at the edges of each wash, where the pigment has pooled as it dried."),
    _d(6, "Draw it first with a loose charcoal line, and let the line show through thin, transparent washes laid over the top."),
    _d(6, "Build it up in three or more transparent glazes, each one smaller and darker than the last, with every layer visible."),
    _d(6, "Use only big, loose wash shapes with no line work at all; the edges come from where one wash meets another.", clashes=[INK]),
    _d(6, "Add crisp pen-and-ink detail on the subject only, while the setting stays as soft washes with no lines."),
    _d(6, "Soften every edge so the washes bleed outward into the paper, with not a single hard line in the whole painting.", clashes=[INK]),
    _d(6, "Cross-hatch the darkest areas in two directions with a fine pencil, and keep the lit side as clean, open wash."),
    _d(6, "Lay a fine, misty spray of colour behind the subject, fading out softly into the bare paper around it."),
    _d(6, "Let the colours bleed into each other wherever they touch, so the edges inside the subject are soft and blooming."),
    _d(6, "Use dry, scratchy coloured-pencil strokes over a single pale wash, with the pencil doing most of the describing."),
    # tier 7: light, mood, time of day
    _d(7, "Light it with golden-hour backlight: a warm glow behind it, a bright rim along its edges and long shadows falling towards us."),
    _d(7, "Make it an overcast, quiet day: soft grey light, muted colours, and no hard shadows anywhere in the picture."),
    _d(7, "Paint it moonlit at night: deep blue darks, cold silver highlights, and one soft pool of pale light on the subject."),
    _d(7, "Set it on a misty morning, with distant shapes fading into pale, milky mist and only the nearest edges clear.", "sky"),
    _d(7, "Make it stormy: heavy, dark clouds pile up behind it and a cold, uneasy light falls across the whole scene.", "sky"),
    _d(7, "Light it with strong midday sun from straight above: short, crisp shadows directly underneath and bleached highlights on top."),
    _d(7, "Light it from one low side, so one half glows warmly and the other half falls into long, cool shadow."),
    _d(7, "Set it at dusk, in the last blue light after sunset, with the sky fading from soft peach to deep blue.", "sky"),
    _d(7, "Make it a rainy day, with wet, reflective ground, blurred shapes and slanting streaks of rain across the picture.", "sky"),
    _d(7, "Give it the hush of falling snow: a pale grey sky, soft white flakes and cold blue shadows on the ground.", "sky"),
    _d(7, "Make it a warm, lamplit evening: a soft orange glow on the subject and deep, warm brown shadows around it."),
    _d(7, "Set it on a bright, breezy spring morning, with clean fresh light, small sharp shadows and a clear pale sky behind.", "sky"),
    _d(7, "Give it a hazy, heat-soaked summer afternoon: washed-out colours, shimmering air and soft, short shadows."),
    _d(7, "Make the mood calm and melancholy, with dim, cool light and long, soft shadows stretching away from it."),
    _d(7, "Light it with dappled sunlight falling through leaves, scattering small bright patches across it and the ground.", "sky"),
    _d(7, "Catch the last light of a winter afternoon: a low, pale sun, long blue shadows and a cold, clear sky.", "sky"),
    _d(7, "Make it feel bright and joyful, in clear morning light with saturated colour and short, crisp, cheerful shadows."),
    _d(7, "Light it from behind with a strong glare, so it becomes a dark, simple silhouette with a glowing edge."),
    # tier 8: two or three directions from different tiers
    _d(8, "Place it small in the lower third under a wide sky, lit by golden-hour backlight with long shadows stretching towards us.", "sky", parts=(4, 7)),
    _d(8, "Use only {colour} and {other}, with soft, bleeding wet-on-wet washes for the background and crisp pencil detail on the subject.", parts=(5, 6)),
    _d(8, "Seen from below against a stormy sky, in a limited palette of {colour} and {other}, with the clouds dark and heavy behind it.", "sky", parts=(4, 5, 7)),
    _d(8, "Make it a misty morning: paint the sky wet-on-wet so it melts into the mist, and keep the nearest edges dry and sharp.", "sky", parts=(6, 7)),
    _d(8, "Crop in close so it fills the frame, and shade its shadows with fine hatched lines over transparent washes.", clashes=[MINIMAL], parts=(4, 6)),
    _d(8, "Make it moonlit and low-key: deep, dark blue washes almost everywhere, with the {colour} catching the only pale light.", clashes=[MINIMAL], parts=(5, 7)),
    _d(8, "Set it off-centre to the right with open white paper on the left, and keep the whole picture high-key and pale.", parts=(4, 5)),
    _d(8, "Make it an overcast, quiet day with soft grey light and no hard shadows, and keep the outlines broken and sparse.", parts=(6, 7)),
    _d(8, "Seen from high above on a bright, still afternoon, with grainy, granulated washes and short, crisp shadows underneath.", "above", parts=(4, 6, 7)),
    _d(8, "Paint it entirely in tones of {colour}, with flowing strokes that follow a gentle curved current across the background.", parts=(5, 6)),
    _d(8, "Put it far off in the middle distance, small and pale, on a rainy day with wet, reflective ground in front of it.", "sky", parts=(4, 7)),
    _d(8, "Let the {colour} be the only warm note in a cool, overcast scene of soft blue-greys and quiet, shadowless light.", hue="warm", parts=(5, 7)),
    _d(8, "Keep the picture cool and moonlit, with one small {other} glint as the only warm light anywhere in it.", hue="cool", parts=(5, 7)),
    _d(8, "Paint it as a vignette that fades to bare paper at the corners, with wave-like strokes rippling through the background.", parts=(4, 6)),
    _d(8, "Lay it along a strong diagonal, use only {colour} and {other}, and leave hard, dark tide-lines at the edges of each wash.", parts=(4, 5, 6)),
    _d(8, "Light it from one low side, and hatch the shadow side in fine pencil lines while the lit side stays clean wash.", parts=(6, 7)),
    _d(8, "Show it reflected in still water across the lower half of the page, in the soft, fading blue light of dusk.", "water", parts=(4, 7)),
    _d(8, "Make it a golden-hour scene in a limited palette of {colour} and {other}, with soft wet-on-wet washes and a few broken pencil lines.", parts=(5, 6, 7)),
)  # fmt: skip

# ---------------------------------------------------------------------------
# Held-out directions: never used in training, four per tier
# ---------------------------------------------------------------------------

HELDOUT_DIRECTIONS: tuple[Direction, ...] = (
    _d(4, "Tuck it into the top left quarter of the page, and leave the rest as open paper touched by a single pale wash."),
    _d(4, "Seen from a worm's-eye view at ground level, with blades of grass or pebbles looming large and close in the foreground.", "sky"),
    _d(4, "Lead the eye in along a gentle S-curve, a path, a stream or a shadow, winding from the bottom edge up to it.", "sky"),
    _d(4, "Show it from a three-quarter angle, turned slightly away from us, and place it on the right third of the page."),
    _d(5, "Restrict yourself to {colour}, {other} and nothing else, and let the overlaps of the two make every in-between tone."),
    _d(5, "Make it a two-value picture: one flat pale tone and one flat dark tone, with almost no middle values between them."),
    _d(5, "Bleach it out like a sun-faded print, with every colour, the {colour} included, washed thin and pale."),
    _d(5, "Let the {colour} be the only saturated colour, and paint the rest in dusty, greyed-down {other} tones."),
    _d(6, "Model its form with parallel hatching laid at a single steep angle, over flat, pale underlying washes."),
    _d(6, "Give the background a mottled, seabed-like flow of short, irregular strokes, while the subject stays smooth and calm."),
    _d(6, "Outline it with a single unbroken marker line, then flood loose colour inside it that strays past the line."),
    _d(6, "Give only the shadows a speckled, granular texture, and keep the lit areas as smooth, clean washes."),
    _d(7, "Set it just before a thunderstorm, under an eerie greenish light, with everything still and the sky turning dark.", "sky"),
    _d(7, "Light it with flickering firelight from below, warm and orange, throwing tall, wavering shadows up behind it."),
    _d(7, "Make it a foggy night under a single streetlamp, with a halo of light around the lamp and soft darkness beyond.", "sky"),
    _d(7, "Let a single hard shaft of sunlight cut across it at an angle, leaving everything outside the beam in deep shadow."),
    _d(8, "Tuck it into the top left quarter of the page under a pale, clearing sky just after rain, with shining wet ground below.", "sky", parts=(4, 7)),
    _d(8, "Restrict it to {colour} and {other}, outline it once with an unbroken marker line, and flood colour inside that strays past it.", parts=(5, 6)),
    _d(8, "Seen from a worm's-eye view at ground level, as a two-value picture of flat pale and flat dark, lit by firelight from below.", "sky", parts=(4, 5, 7)),
    _d(8, "Under an eerie greenish light before a thunderstorm, give only the shadows a speckled, granular texture and keep the lit areas smooth.", "sky", parts=(6, 7)),
)  # fmt: skip


# ---------------------------------------------------------------------------
# Building the prompts
# ---------------------------------------------------------------------------


def _triple(p: PaintPrompt) -> tuple[str, str, str]:
    return (p.subject, p.colour, p.style)


def _build(
    n: int,
    directions: Sequence[Direction],
    exclude: Iterable[PaintPrompt],
    kind: PromptKind,
    id_letter: str,
    rng: random.Random,
) -> list[PaintPrompt]:
    if n <= 0:
        return []
    by_tier = {t: [d for d in directions if d.tier == t] for t in DIRECTED_TIERS}
    held = {_triple(p) for p in exclude}
    bases = [
        p for p in build_prompt_pool() if p.subject not in NOVEL_SUBJECTS and _triple(p) not in held
    ]
    # Every (tier, base prompt) pair that some direction of the tier fits.
    pairs = [(t, b) for t in DIRECTED_TIERS for b in bases if any(d.fits(b) for d in by_tier[t])]
    # Even over tiers; then families, subjects, colours and styles overall,
    # and families, colours and styles within each tier.
    picked = _stratified_pick(
        pairs,
        n,
        lambda tb: tb[0],
        rng,
        balance=(
            lambda tb: tb[1].family,
            lambda tb: tb[1].subject,
            lambda tb: tb[1].colour,
            lambda tb: tb[1].style,
            lambda tb: (tb[0], tb[1].family),
            lambda tb: (tb[0], tb[1].colour),
            lambda tb: (tb[0], tb[1].style),
        ),
    )
    # Within a tier each pair takes the least-used direction that fits it,
    # pairs with the fewest options choosing first; the shuffle breaks ties.
    order = {t: rng.sample(range(len(ds)), len(ds)) for t, ds in by_tier.items()}
    options = [[i for i in order[t] if by_tier[t][i].fits(base)] for t, base in picked]
    used: dict[tuple[int, int], int] = dict.fromkeys(
        ((t, i) for t, ds in by_tier.items() for i in range(len(ds))), 0
    )
    chosen: dict[int, int] = {}
    for k in sorted(range(len(picked)), key=lambda k: len(options[k])):
        t = picked[k][0]
        chosen[k] = min(options[k], key=lambda i: used[t, i])
        used[t, chosen[k]] += 1
    prompts: list[PaintPrompt] = []
    for k, (t, base) in enumerate(picked):
        i = chosen[k]
        prompts.append(
            PaintPrompt(
                id=f"d{t}{id_letter}{i:02d}{base.id}",
                text=f"{base.text} {by_tier[t][i].render(base.colour)}",
                subject=base.subject,
                tier=t,
                family=base.family,
                colour=base.colour,
                style=base.style,
                kind=kind,
            )
        )
    return prompts


def build_directed_prompts(
    n: int = 256, seed: int = 0, exclude: Iterable[PaintPrompt] | None = None
) -> list[PaintPrompt]:
    """``n`` directed training prompts, balanced over tiers 4-8, families, subjects, colours and styles.

    ``exclude``: prompts whose (subject, colour, style) triple must not be
    reused, normally the run's held-out prompts; None takes the default
    held-out split for ``seed``. Novel subjects are always left out.
    """
    if exclude is None:
        exclude = build_prompt_splits(seed=seed)[1]
    return _build(n, TRAIN_DIRECTIONS, exclude, "train", "v", random.Random(f"directed:{seed}"))


def build_directed_heldout(
    n: int = 16, seed: int = 0, exclude: Iterable[PaintPrompt] | None = None
) -> list[PaintPrompt]:
    """``n`` held-out directed prompts (kind ``novel_direction``), balanced like the training ones.

    Subjects, colours and styles are ones seen in training; the direction
    phrasings are reserved for testing and never appear in
    :func:`build_directed_prompts`. ``exclude`` is as there.
    """
    if exclude is None:
        exclude = build_prompt_splits(seed=seed)[1]
    return _build(
        n,
        HELDOUT_DIRECTIONS,
        exclude,
        "novel_direction",
        "h",
        random.Random(f"directed-heldout:{seed}"),
    )
