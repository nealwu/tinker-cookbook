"""Multimodal LLM-as-a-Verifier reward, and the held-out Gemini evaluator.

The policy writes code; the *reward* comes from a multimodal model that looks
at the rendered paintings.

* :class:`PairwiseVerifier` — LLM-as-a-Verifier, the training reward, built on
  ``llm_verifier.compare`` from the `llm-verifier` package
  (https://github.com/llm-as-a-verifier/llm-as-a-verifier). Every comparison
  shows the verifier two paintings and ONE criterion and asks for a letter
  grade A-T per painting; the grade is read as the *expectation over the
  verifier's token distribution* at each ``<score_X>`` tag, not as the sampled
  letter, so a 5% chance of "A" still moves the reward. Scores are averaged
  over criteria and ``n_evaluations`` repeats, odd repeats swapping the two
  paintings so positional bias cancels. A rollout group is scored with a full
  **round-robin tournament**: every unordered pair of compiled paintings is
  compared, so a painting's reward is its mean grade against every other
  painting in its group.

  The verifier model (``thinkingmachines/Inkling-Small`` by default) is reached
  through Tinker's OpenAI-compatible endpoint, which returns token logprobs.
  That endpoint does not support assistant prefill, so the client is set up to
  read the letter distribution from the model's own sampled score tags (the
  ``llm_verifier`` path for hosted APIs) rather than by prefilling each tag.

* :class:`GeminiEvaluator` — the held-out yardstick: ``gemini-3.8-flash``
  scores each held-out painting on its own, 0-10 per criterion. It never
  trains anything; it gives every checkpoint a score on one fixed scale that
  is independent of the training reward.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import chz
from PIL import Image

try:
    import llm_verifier
except ImportError as e:  # pragma: no cover - environment specific
    raise ImportError("paint_rl needs the llm-verifier package: pip install llm-verifier") from e

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_EVALUATOR_MODEL = "gemini-3.8-flash"
TINKER_OPENAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Tinker's named reasoning-effort presets (the OpenAI-compatible endpoint
# takes the name, the CLI takes the float).
EFFORT_PRESETS: dict[float, str] = {
    0.0: "none",
    0.1: "minimal",
    0.2: "low",
    0.7: "medium",
    0.9: "high",
    0.99: "xhigh",
}


# ---------------------------------------------------------------------------
# Criteria (narrow ones, decidable from the image alone)
# ---------------------------------------------------------------------------

GROUND_TRUTH_NOTE = (
    "Judge ONLY what is visible in the rendered painting(s). The code that "
    "produced them is not shown and does not matter. A blank, near-uniform, "
    "or noise-filled canvas is a complete failure regardless of intent. Do "
    "not assume effort or hidden detail: if you cannot see it, it does not count."
)

DEFAULT_CRITERIA: dict[str, str] = {
    "Prompt Adherence": (
        "Is the requested subject immediately recognisable to a casual viewer, "
        "and are the colour and style words in the prompt honoured? A viewer "
        "should be able to name the subject without reading the prompt. Score "
        "HIGH for an unmistakable, specific depiction; score LOW for generic "
        "blobs, an unrelated subject, or a canvas where the subject must be "
        "guessed. Ignore painterly polish and composition here."
    ),
    "Watercolor Technique": (
        "Does it look like real watercolor on paper: soft bleeding edges, "
        "layered translucent washes that pool and overlap, visible paper "
        "texture or granulation, varied stroke pressure, restrained broken "
        "outlines? Score LOW for flat clip-art fills, hard vector edges, "
        "uniform opaque shapes, or random scribbles. Ignore whether the "
        "subject matches the prompt."
    ),
    "Composition and Aesthetics": (
        "Is the picture pleasant to look at: a clear focal point, balanced use "
        "of the canvas, deliberate negative space, harmonious colours? Score "
        "LOW for tiny or cut-off subjects, cluttered chaos, muddy colour, or "
        "large empty regions that look unfinished. Ignore subject correctness."
    ),
}

# What the verifier is told the two "trajectories" are. llm_verifier attaches
# the images in order after the task, so slot A is the first image.
PAIRWISE_NOTE = (
    "The candidates are watercolor paintings made by code for the painting "
    "request in the task, attached as images. A letter grade of A means "
    "outstanding on the criterion and T means a complete failure, e.g. a blank "
    "canvas or pure noise. " + GROUND_TRUTH_NOTE
)
SLOT_A = "Painting A: the FIRST attached image."
SLOT_B = "Painting B: the SECOND attached image."


def resolve_criteria(names: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(name, description)`` for ``names`` (keys of DEFAULT_CRITERIA); empty = all."""
    selected = list(names) or list(DEFAULT_CRITERIA)
    missing = [n for n in selected if n not in DEFAULT_CRITERIA]
    if missing:
        raise ValueError(f"unknown criteria {missing}; known: {list(DEFAULT_CRITERIA)}")
    return [(n, DEFAULT_CRITERIA[n]) for n in selected]


@chz.chz
class VerifierConfig:
    """The training verifier, called through Tinker's OpenAI-compatible endpoint."""

    model_name: str = DEFAULT_VERIFIER_MODEL
    # Thinking effort; one of EFFORT_PRESETS. 0.2 is "low": enough for a
    # two-image comparison and keeps the reward cheap.
    effort: float = 0.2
    # Output budget, shared by the reasoning trace and the verdict.
    max_tokens: int = 8192
    # Repeated verifications K per criterion. Odd repeats swap the prompt slots.
    n_evaluations: int = 2
    # Criteria names (keys of DEFAULT_CRITERIA); empty = all of them.
    criteria: tuple[str, ...] = ()
    # Longest image side sent to the model.
    image_size: int = 448
    max_concurrency: int = 480
    base_url: str | None = None


@chz.chz
class EvaluatorConfig:
    """The held-out evaluator (Gemini), which never enters the reward."""

    model_name: str = DEFAULT_EVALUATOR_MODEL
    max_tokens: int = 8192
    temperature: float = 1.0
    n_evaluations: int = 1
    criteria: tuple[str, ...] = ()
    image_size: int = 448
    max_concurrency: int = 50


@dataclass
class Verdict:
    """One verifier call, kept for logging/artifacts."""

    criterion: str
    rep: int
    slot_a: int
    """Group index of the painting shown in slot A (evaluator: the only painting)."""
    slot_b: int | None
    score_a: float
    score_b: float | None
    text: str
    truncated: bool


@dataclass
class GroupScores:
    """Per-painting rewards in [0, 1] plus the verdict log for a group."""

    scores: list[float]
    verdicts: list[Verdict] = field(default_factory=list)
    n_calls: int = 0
    n_truncated: int = 0


def _fit(image: Image.Image, size: int) -> Image.Image:
    im = image.convert("RGB")
    im.thumbnail((size, size))
    return im


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# LLM-as-a-Verifier: pairwise, fine-grained, round-robin tournament
# ---------------------------------------------------------------------------


def round_robin_pairs(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Every unordered pair of ``n`` candidates as a directed (slot A, slot B).

    Which member of a pair starts in slot A is randomised, and the pairs are
    shuffled, so a group's comparisons are not systematically ordered by index;
    ``n_evaluations = 2`` then re-runs each pair with the slots swapped, which
    is what actually cancels the positional bias.
    """
    pairs = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in combinations(range(n), 2)]
    rng.shuffle(pairs)
    return pairs


def effort_preset(effort: float) -> str:
    """The named preset for ``effort``, as Tinker's OpenAI endpoint expects it."""
    for value, name in EFFORT_PRESETS.items():
        if abs(effort - value) < 1e-9:
            return name
    raise ValueError(f"effort must be one of {sorted(EFFORT_PRESETS)}, got {effort}")


# Extra retries for transient endpoint errors (502s, timeouts), on top of the
# openai client's own quick ones: an outage of a minute or two then costs
# latency instead of turning a whole step's comparisons into ties.
RETRY_DELAYS_S: tuple[float, ...] = (10.0, 30.0, 60.0)


def _is_transient(error: Exception) -> bool:
    import openai

    if isinstance(error, openai.APIConnectionError):  # includes timeouts
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def make_tinker_verifier_client(config: VerifierConfig) -> Any:
    """An ``openai`` client on Tinker's OpenAI-compatible endpoint, set up for llm_verifier.

    The endpoint returns token logprobs but has no assistant prefill, so the
    client is tagged for llm_verifier's hosted-API path, which reads the
    letter distribution at the score tags the model samples itself. That path
    takes its reasoning effort and token budget from the environment.
    """
    from openai import OpenAI

    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise llm_verifier.MissingAPIKeyError("set TINKER_API_KEY to use the Tinker verifier")
    os.environ["DEEPSEEK_EFFORT"] = effort_preset(config.effort)
    os.environ["DEEPSEEK_MAX_TOKENS"] = str(config.max_tokens)
    client = OpenAI(
        base_url=config.base_url or TINKER_OPENAI_BASE_URL,
        api_key=key,
        max_retries=5,
        # A comparison takes 5-16 s; the occasional request that hangs on the
        # endpoint is retried after 2 min instead of stalling the step for 10.
        timeout=120.0,
    )
    client._llm_verifier_model = config.model_name  # pyright: ignore[reportAttributeAccessIssue]
    client._llm_verifier_deepseek = True  # pyright: ignore[reportAttributeAccessIssue]
    return client


class PairwiseVerifier:
    """LLM-as-a-Verifier reward for a group of paintings (round-robin tournament)."""

    def __init__(self, config: VerifierConfig):
        self.config = config
        self._client = make_tinker_verifier_client(config)
        # llm_verifier.compare is synchronous; run calls on a dedicated pool
        # sized to the concurrency this loop wants.
        self._pool = ThreadPoolExecutor(
            max_workers=config.max_concurrency, thread_name_prefix="paint-verifier"
        )

    def _compare_sync(
        self, problem: str, image_a: bytes, image_b: bytes, criterion: tuple[str, str]
    ) -> tuple[float, float, bool]:
        name, description = criterion
        for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
            try:
                ra, rb = llm_verifier.compare(
                    problem,
                    SLOT_A,
                    SLOT_B,
                    criteria={name: description},
                    images=[image_a, image_b],
                    ground_truth_note=PAIRWISE_NOTE,
                    model=self.config.model_name,
                    client=self._client,
                )
                return ra, rb, False
            except Exception as e:
                if delay is not None and _is_transient(e):
                    logger.info(
                        "verifier call failed (%s), retry %d in %.0f s", e, attempt + 1, delay
                    )
                    time.sleep(delay)
                    continue
                # A call that still fails is a tie, as llm_verifier's on_error="tie".
                logger.warning("verifier call failed (%s: %s); scoring a tie", type(e).__name__, e)
                return 0.5, 0.5, True
        raise AssertionError("unreachable")

    async def compare(
        self, problem: str, image_a: bytes, image_b: bytes, criterion: tuple[str, str]
    ) -> tuple[float, float, bool]:
        """Directed comparison: A in slot A, B in slot B. Returns (R_A, R_B, failed)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, self._compare_sync, problem, image_a, image_b, criterion
        )

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        """Round-robin rewards for ``images`` (all of them must be valid paintings).

        Every unordered pair of paintings is compared, for every criterion and
        repeat; odd repeats swap the two slots, so with the default
        ``n_evaluations = 2`` each pair is judged once in each order and the
        verifier's positional bias cancels pair by pair. A painting's reward is
        the mean of its fine-grained scores over the ``n - 1`` opponents it met.
        """
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        if n == 1:
            return GroupScores(scores=[0.5])
        criteria = resolve_criteria(self.config.criteria)
        pngs = [_png_bytes(_fit(im, self.config.image_size)) for im in images]
        jobs = [
            (a, b, crit, rep, rep % 2 == 1)
            for a, b in round_robin_pairs(n, random.Random(seed))
            for crit in criteria
            for rep in range(self.config.n_evaluations)
        ]

        async def run(job: tuple[int, int, tuple[str, str], int, bool]) -> Verdict:
            a, b, crit, rep, swap = job
            first, second = (b, a) if swap else (a, b)
            r_first, r_second, failed = await self.compare(problem, pngs[first], pngs[second], crit)
            return Verdict(
                criterion=crit[0],
                rep=rep,
                slot_a=first,
                slot_b=second,
                score_a=r_first,
                score_b=r_second,
                text="",
                truncated=failed,
            )

        verdicts = await asyncio.gather(*(run(job) for job in jobs))
        total = [0.0] * n
        count = [0] * n
        for v in verdicts:
            assert v.slot_b is not None and v.score_b is not None
            total[v.slot_a] += v.score_a
            count[v.slot_a] += 1
            total[v.slot_b] += v.score_b
            count[v.slot_b] += 1
        return GroupScores(
            scores=[total[i] / count[i] if count[i] else 0.5 for i in range(n)],
            verdicts=list(verdicts),
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# Held-out evaluator: Gemini, one painting at a time, 0-10
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    truncated: bool


class _GeminiClient:
    """Gemini via the google-genai async client."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._client: Any = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            from google import genai

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise llm_verifier.MissingAPIKeyError(
                    f"set GEMINI_API_KEY to use {self.config.model_name!r} as the held-out "
                    "evaluator, or pass eval_model=None to turn it off"
                )
            from google.genai.types import HttpOptions, HttpRetryOptions

            # Gemini sheds load with 429 / 503 ("high demand"); retry those with
            # exponential backoff instead of failing the evaluation.
            retry = HttpRetryOptions(
                attempts=8,
                initial_delay=2.0,
                max_delay=60.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            )
            self._client = genai.Client(api_key=key, http_options=HttpOptions(retry_options=retry))

    async def ask(self, prompt: str, image: Image.Image) -> Reply:
        await self._ensure_started()
        from google.genai.types import Content, GenerateContentConfig, Part

        parts = [Part(text=prompt), Part.from_bytes(data=_png_bytes(image), mime_type="image/png")]
        try:
            async with self._semaphore:
                response = await self._client.aio.models.generate_content(
                    model=self.config.model_name,
                    contents=[Content(role="user", parts=parts)],
                    config=GenerateContentConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    ),
                )
        except Exception as e:  # still failing after the client's retries
            logger.warning("Gemini call failed (%s: %s)", type(e).__name__, e)
            return Reply(text="", truncated=True)
        text = response.text or ""
        candidate = (response.candidates or [None])[0]
        finish = getattr(candidate, "finish_reason", None)
        return Reply(
            text=text,
            truncated=str(finish or "").upper().endswith("MAX_TOKENS") or not text,
        )


_EVALUATOR_PREAMBLE = (
    "You are an expert reviewer of watercolor paintings. You will be given a "
    "painting request and one painting made for it, and you score it on ONE "
    "specific criterion.\n\n"
    f"{GROUND_TRUTH_NOTE}\n"
)


def evaluator_prompt(problem: str, criterion: tuple[str, str]) -> str:
    name, description = criterion
    return (
        _EVALUATOR_PREAMBLE + f"\n**Evaluation Guideline — {name}:**\n{description}\n\n"
        f'Score the painting ONLY on this specific criterion ("{name}") from 0 '
        "(complete failure) to 10 (outstanding).\n\n"
        "Reason it through first, then END your reply with exactly this line and "
        "nothing after it, replacing the placeholder with an integer 0-10:\n"
        "<score> INTEGER_0_TO_10 </score>\n"
        f"\n**Task (the painting request):**\n{problem}\n"
        "\nThe painting is attached below.\n\nBegin your analysis now."
    )


_SCORE_TAG = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


def extract_score(text: str) -> float | None:
    """Last ``<score> n </score>`` in the reply, normalised to [0, 1]."""
    matches = _SCORE_TAG.findall(text or "")
    if not matches:
        return None
    return max(0.0, min(10.0, float(matches[-1]))) / 10.0


class GeminiEvaluator:
    """Independent absolute scores per painting, no pairing."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._client = _GeminiClient(config)

    async def score_one(
        self, problem: str, image: Image.Image, criterion: tuple[str, str]
    ) -> tuple[float | None, Reply]:
        """Score in [0, 1], or None when the call failed or the verdict is unparseable."""
        reply = await self._client.ask(evaluator_prompt(problem, criterion), image)
        return extract_score(reply.text), reply

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        del seed  # independent scores; nothing to randomise
        if not images:
            return GroupScores(scores=[])
        criteria = resolve_criteria(self.config.criteria)
        fitted = [_fit(im, self.config.image_size) for im in images]
        jobs = [
            (i, crit, rep)
            for i in range(len(images))
            for crit in criteria
            for rep in range(self.config.n_evaluations)
        ]

        async def run(job: tuple[int, tuple[str, str], int]) -> tuple[Verdict, Reply, bool]:
            i, crit, rep = job
            score, reply = await self.score_one(problem, fitted[i], crit)
            verdict = Verdict(
                criterion=crit[0],
                rep=rep,
                slot_a=i,
                slot_b=None,
                score_a=0.5 if score is None else score,
                score_b=None,
                text=reply.text,
                truncated=reply.truncated or score is None,
            )
            return verdict, reply, score is not None

        triples = await asyncio.gather(*(run(job) for job in jobs))
        results = [(v, r) for v, r, _ in triples]
        # Failed or unparseable verdicts are left out of the mean; a painting
        # with none at all falls back to a 0.5 tie.
        total = [0.0] * len(images)
        count = [0] * len(images)
        for v, _, ok in triples:
            if ok:
                total[v.slot_a] += v.score_a
                count[v.slot_a] += 1
        return GroupScores(
            scores=[total[i] / count[i] if count[i] else 0.5 for i in range(len(images))],
            verdicts=[v for v, _ in results],
            n_calls=len(results),
            n_truncated=sum(v.truncated for v, _ in results),
        )


_VERIFIERS: dict[VerifierConfig, PairwiseVerifier] = {}
_EVALUATORS: dict[EvaluatorConfig, GeminiEvaluator] = {}


def get_shared_verifier(config: VerifierConfig) -> PairwiseVerifier:
    """One verifier per config per process, so the client and its pool are reused."""
    if config not in _VERIFIERS:
        _VERIFIERS[config] = PairwiseVerifier(config)
    return _VERIFIERS[config]


def get_shared_evaluator(config: EvaluatorConfig) -> GeminiEvaluator:
    """One evaluator per config per process."""
    if config not in _EVALUATORS:
        _EVALUATORS[config] = GeminiEvaluator(config)
    return _EVALUATORS[config]
