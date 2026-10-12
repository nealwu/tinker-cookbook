"""Multimodal LLM-as-a-Verifier reward, and the held-out evaluator.

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

* :class:`TinkerEvaluator` — the held-out yardstick: a stronger vision model
  on Tinker (``moonshotai/Kimi-K2.6`` by default) scores each held-out
  painting on its own, 0-10 per criterion. It never trains anything; it gives
  every checkpoint a score on one fixed scale that is independent of the
  training reward.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal

import chz
import tinker
from PIL import Image

from tinker_cookbook import model_info
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import (
    ImagePart,
    Message,
    Renderer,
    TextPart,
    get_renderer,
    get_text_content,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

try:
    import llm_verifier
    from llm_verifier import fine_grained_reward as llm_verifier_fgr
except ImportError as e:  # pragma: no cover - environment specific
    raise ImportError("paint_rl needs the llm-verifier package: pip install llm-verifier") from e

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_EVALUATOR_MODEL = "moonshotai/Kimi-K2.6"
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

# The same for the absolute-grading ablation, which shows one painting per call.
ABSOLUTE_NOTE = (
    "The candidate is a watercolor painting made by code for the painting "
    "request in the task, attached as an image. A letter grade of A means "
    "outstanding on the criterion and T means a complete failure, e.g. a blank "
    "canvas or pure noise. " + GROUND_TRUTH_NOTE
)
SLOT_SINGLE = "Painting A: the attached image."


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
    # Ablations of the reward. "absolute" grades each painting on its own with
    # the same prompt framing, scale and score reading as a pairwise call (one
    # painting instead of two; repeats are plain repeats). "judge" is the
    # LLM-as-a-Judge baseline: each painting on its own, an integer 0-10 per
    # criterion read from the reply (the held-out evaluator's prompt), same
    # model, endpoint and effort. "letter" reads the sampled grade instead of its
    # expectation under the token distribution (pairwise and absolute only).
    mode: Literal["pairwise", "absolute", "judge"] = "pairwise"
    score_from: Literal["expectation", "letter"] = "expectation"


@chz.chz
class EvaluatorConfig:
    """The held-out evaluator, a vision model on Tinker that never enters the reward."""

    model_name: str = DEFAULT_EVALUATOR_MODEL
    # Default: the model's recommended renderer (thinking on, for Kimi-K2.6).
    renderer_name: str | None = None
    # Output budget, shared by the reasoning trace and the verdict.
    max_tokens: int = 8192
    temperature: float = 1.0
    n_evaluations: int = 1
    criteria: tuple[str, ...] = ()
    image_size: int = 448
    # One checkpoint's held-out evaluation is 16 prompts x 5 paintings x 3
    # criteria = 240 calls; this keeps all of them in flight.
    max_concurrency: int = 256
    base_url: str | None = None


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


def absolute_prompt(problem: str, criterion: tuple[str, str]) -> str:
    """Single-painting counterpart of ``llm_verifier``'s pairwise prompt.

    Same framing, rating scale and ``<score_A>`` tag as the pairwise call, so
    the absolute-grading ablation changes only how many paintings are shown.
    """
    name, description = criterion
    scale = llm_verifier_fgr.SCALE
    return (
        "You are an expert evaluator of AI coding agents. "
        "You will see a task description and one agent trajectory, then "
        "evaluate it on ONE specific criterion, stated at the end.\n\n"
        f"{ABSOLUTE_NOTE}\n\n"
        f"**Task:**\n{problem}\n\n"
        "**Attached images:** 1 image(s) are attached to this message, in order; "
        "they are part of the task context.\n\n"
        f"**Trajectory A:**\n{SLOT_SINGLE}\n\n"
        f"**Rating Scale:**\n{scale['scale_description']}\n\n"
        f"**Evaluation Guideline — {name}:**\n{description}\n\n"
        f'Score the trajectory ONLY on this specific criterion ("{name}"). Ignore '
        "other aspects of the trajectory that are not relevant to it.\n\n"
        "Reason it through first, then END your reply with exactly this line and "
        "nothing after it. Replace the placeholder with a single letter A-T, "
        "keeping the spaces around the letter exactly as shown:\n"
        f"<score_A> {scale['score_format']} </score_A>\n\n"
        "Begin your analysis now."
    )


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
        # No short client-side timeout: under load the endpoint is slow but still
        # progressing, and timing a request out only re-sends it, which adds load
        # and makes every later call slower. The bound only catches a truly hung
        # connection; the group-scoring cap (grader_timeout_s) bounds the step.
        timeout=1800.0,
    )
    client._llm_verifier_model = config.model_name  # pyright: ignore[reportAttributeAccessIssue]
    client._llm_verifier_deepseek = True  # pyright: ignore[reportAttributeAccessIssue]
    return client


class PairwiseVerifier:
    """LLM-as-a-Verifier reward for a group of paintings (round-robin tournament).

    With ``mode="absolute"`` it grades each painting on its own instead (an
    ablation of the pairwise reward).
    """

    def __init__(self, config: VerifierConfig):
        self.config = config
        self._client = make_tinker_verifier_client(config)
        # llm_verifier.compare is synchronous; run calls on a dedicated pool
        # sized to the concurrency this loop wants.
        self._pool = ThreadPoolExecutor(
            max_workers=config.max_concurrency, thread_name_prefix="paint-verifier"
        )

    def _with_retries(
        self, call: Callable[[], list[float]], n_scores: int
    ) -> tuple[list[float], bool]:
        """Run one verifier call; returns (scores, failed)."""
        for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
            try:
                return call(), False
            except Exception as e:
                if delay is not None and _is_transient(e):
                    logger.info(
                        "verifier call failed (%s), retry %d in %.0f s", e, attempt + 1, delay
                    )
                    time.sleep(delay)
                    continue
                # A call that still fails is a tie, as llm_verifier's on_error="tie".
                logger.warning("verifier call failed (%s: %s); scoring a tie", type(e).__name__, e)
                return [0.5] * n_scores, True
        raise AssertionError("unreachable")

    def _read_scores(self, prompt: str, images: list[bytes], tags: list[str]) -> list[float]:
        """One verifier call through llm_verifier's internals, scored per ``score_from``."""
        text, tokens, position_logprobs = llm_verifier_fgr.call_verifier(
            self._client,
            prompt,
            self.config.model_name,
            images=llm_verifier_fgr.as_image_list(images),
        )
        if self.config.score_from == "letter":
            tokens, position_logprobs = None, None  # extract_score falls back to the letter
        return [
            llm_verifier_fgr.extract_score(text, tokens, position_logprobs, tag) for tag in tags
        ]

    def _compare_sync(
        self, problem: str, image_a: bytes, image_b: bytes, criterion: tuple[str, str]
    ) -> tuple[float, float, bool]:
        name, description = criterion
        if self.config.score_from == "expectation":

            def call() -> list[float]:
                return list(
                    llm_verifier.compare(
                        problem,
                        SLOT_A,
                        SLOT_B,
                        criteria={name: description},
                        images=[image_a, image_b],
                        ground_truth_note=PAIRWISE_NOTE,
                        model=self.config.model_name,
                        client=self._client,
                    )
                )

        else:
            # What llm_verifier.compare does for one criterion, minus the logprobs.
            prompt = llm_verifier_fgr.build_prompt(
                problem,
                SLOT_A,
                SLOT_B,
                {"name": name, "description": description},
                PAIRWISE_NOTE,
                n_images=2,
            )

            def call() -> list[float]:
                return self._read_scores(prompt, [image_a, image_b], ["<score_A>", "<score_B>"])

        (ra, rb), failed = self._with_retries(call, 2)
        return ra, rb, failed

    def _grade_sync(
        self, problem: str, image: bytes, criterion: tuple[str, str]
    ) -> tuple[float, bool]:
        if self.config.mode == "judge":
            prompt = evaluator_prompt(problem, criterion)

            def call() -> list[float]:
                text, _, _ = llm_verifier_fgr.call_verifier(
                    self._client,
                    prompt,
                    self.config.model_name,
                    images=llm_verifier_fgr.as_image_list([image]),
                )
                score = extract_score(text)
                if score is None:
                    raise ValueError("judge reply has no <score> tag")
                return [score]

            (r,), failed = self._with_retries(call, 1)
            return r, failed
        prompt = absolute_prompt(problem, criterion)
        (r,), failed = self._with_retries(
            lambda: self._read_scores(prompt, [image], ["<score_A>"]), 1
        )
        return r, failed

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
        if self.config.mode in ("absolute", "judge"):
            return await self._score_group_absolute(problem, images)
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

    async def _score_group_absolute(self, problem: str, images: list[Image.Image]) -> GroupScores:
        """Each painting graded on its own, for every criterion and repeat."""
        criteria = resolve_criteria(self.config.criteria)
        pngs = [_png_bytes(_fit(im, self.config.image_size)) for im in images]
        loop = asyncio.get_running_loop()

        async def run(i: int, crit: tuple[str, str], rep: int) -> Verdict:
            r, failed = await loop.run_in_executor(
                self._pool, self._grade_sync, problem, pngs[i], crit
            )
            return Verdict(
                criterion=crit[0],
                rep=rep,
                slot_a=i,
                slot_b=None,
                score_a=r,
                score_b=None,
                text="",
                truncated=failed,
            )

        verdicts = await asyncio.gather(
            *(
                run(i, crit, rep)
                for i in range(len(images))
                for crit in criteria
                for rep in range(self.config.n_evaluations)
            )
        )
        total = [0.0] * len(images)
        count = [0] * len(images)
        for v in verdicts:
            total[v.slot_a] += v.score_a
            count[v.slot_a] += 1
        return GroupScores(
            scores=[total[i] / count[i] if count[i] else 0.5 for i in range(len(images))],
            verdicts=list(verdicts),
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# Held-out evaluator: a vision model on Tinker, one painting at a time, 0-10
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    truncated: bool


class _TinkerEvaluatorClient:
    """A vision model on Tinker, sampled through the cookbook renderer for that model."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._renderer: Renderer | None = None
        self._sampling_client: tinker.SamplingClient | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._sampling_client is not None:
                return
            name = self.config.model_name
            if not model_info.get_model_attributes(name).is_vl:
                raise ValueError(f"{name!r} does not accept image input; pick a vision model")
            self._renderer = get_renderer(
                self.config.renderer_name or model_info.get_recommended_renderer_name(name),
                get_tokenizer(name),
                get_image_processor(name),
                model_name=name,
            )
            service_client = tinker.ServiceClient(
                base_url=self.config.base_url,
                user_metadata=recipe_user_metadata("recipe_paint_rl_evaluator"),
            )
            self._sampling_client = await service_client.create_sampling_client_async(
                base_model=name
            )

    async def ask(self, prompt: str, image: Image.Image) -> Reply:
        await self._ensure_started()
        assert self._renderer is not None and self._sampling_client is not None
        messages = [
            Message(
                role="user",
                content=[TextPart(type="text", text=prompt), ImagePart(type="image", image=image)],
            )
        ]
        model_input = self._renderer.build_generation_prompt(messages)
        try:
            async with self._semaphore:
                response = await self._sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stop=self._renderer.get_stop_sequences(),
                    ),
                )
        except Exception as e:  # still failing after the SDK's own retries
            logger.warning("evaluator call failed (%s: %s)", type(e).__name__, e)
            return Reply(text="", truncated=True)
        message, termination = self._renderer.parse_response(response.sequences[0].tokens)
        text = get_text_content(message)
        return Reply(text=text, truncated=not termination.is_stop_sequence or not text)


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


class TinkerEvaluator:
    """Independent absolute scores per painting, no pairing."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._client = _TinkerEvaluatorClient(config)

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
_EVALUATORS: dict[EvaluatorConfig, TinkerEvaluator] = {}


def get_shared_verifier(config: VerifierConfig) -> PairwiseVerifier:
    """One verifier per config per process, so the client and its pool are reused."""
    if config not in _VERIFIERS:
        _VERIFIERS[config] = PairwiseVerifier(config)
    return _VERIFIERS[config]


def get_shared_evaluator(config: EvaluatorConfig) -> TinkerEvaluator:
    """One evaluator per config per process."""
    if config not in _EVALUATORS:
        _EVALUATORS[config] = TinkerEvaluator(config)
    return _EVALUATORS[config]
