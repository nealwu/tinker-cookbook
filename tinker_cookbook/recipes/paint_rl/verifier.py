"""Multimodal rewards (LLM-as-a-Verifier, LLM-as-a-Judge) and the held-out evaluator.

The policy writes code; the *reward* comes from a multimodal model that looks
at the rendered paintings. Two interchangeable training rewards share the
same model, thinking effort, token budget, client and concurrency, so a run
with one is directly comparable to a run with the other:

* :class:`PairwiseVerifier` — LLM-as-a-Verifier, built on
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

* :class:`AbsoluteJudge` — the LLM-as-a-Judge baseline: each painting is
  scored on its own, once, on a single overall criterion, as an integer 1-10
  read from the judge's reply. Paintings with the same integer get the same
  reward (and so the same advantage); how often the top of a group is tied is
  reported as ``scorer/top_tie``.

  Both rewards use ``thinkingmachines/Inkling-Small`` by default, reached
  through Tinker's OpenAI-compatible endpoint. That endpoint returns token
  logprobs but does not support assistant prefill, so the verifier's client
  is set up to read the letter distribution from the model's own sampled
  score tags (the ``llm_verifier`` path for hosted APIs).

* :class:`HeldoutEvaluator` — the held-out yardstick: ``moonshotai/Kimi-K2.6``
  on Tinker's native sampler scores each held-out painting on its own, 1-10,
  with the judge's prompt and single overall criterion. It never trains anything; it gives every checkpoint of every
  run, verifier or judge, a score on one fixed scale.
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
from typing import Any, Literal, TypeVar

import chz
from PIL import Image

try:
    import llm_verifier
except ImportError as e:  # pragma: no cover - environment specific
    raise ImportError("paint_rl needs the llm-verifier package: pip install llm-verifier") from e

logger = logging.getLogger(__name__)

RewardMode = Literal["verifier", "judge"]

# The training reward model and its settings, shared by the verifier and the
# judge so the two arms differ only in how they score.
DEFAULT_REWARD_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_REWARD_EFFORT = 0.2  # the "low" preset
DEFAULT_REWARD_MAX_TOKENS = 8192
# Concurrent calls to the reward model. A verifier step is 8 groups x 10 pairs
# x 3 criteria x 2 repeats = 480 calls, so the whole step is in flight at once,
# with headroom for larger batches or more repeats.
DEFAULT_REWARD_CONCURRENCY = 1200

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
        "Is the requested subject immediately recognizable to a casual viewer, "
        "and are the color and style words in the prompt honored? A viewer "
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
        "of the canvas, deliberate negative space, harmonious colors? Score "
        "LOW for tiny or cut-off subjects, cluttered chaos, muddy color, or "
        "large empty regions that look unfinished. Ignore subject correctness."
    ),
}

# The judge's single overall criterion: the three above, rolled into one.
OVERALL_CRITERION: tuple[str, str] = (
    "Overall Quality",
    "How good is this as a watercolor painting of the request, all things "
    "considered? Weigh together whether the requested subject is immediately "
    "recognizable and the color and style words are honored, whether it looks "
    "like real watercolor on paper (soft bleeding edges, translucent layered "
    "washes, varied strokes, no flat clip-art fills), and whether the "
    "composition is pleasant (clear focal point, balanced use of the canvas, "
    "harmonious color).",
)

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


# Every criterion a scorer can be asked for, by name.
KNOWN_CRITERIA: dict[str, str] = {**DEFAULT_CRITERIA, OVERALL_CRITERION[0]: OVERALL_CRITERION[1]}


def resolve_criteria(names: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(name, description)`` for ``names`` (keys of KNOWN_CRITERIA); empty = the
    three DEFAULT_CRITERIA."""
    selected = list(names) or list(DEFAULT_CRITERIA)
    missing = [n for n in selected if n not in KNOWN_CRITERIA]
    if missing:
        raise ValueError(f"unknown criteria {missing}; known: {list(KNOWN_CRITERIA)}")
    return [(n, KNOWN_CRITERIA[n]) for n in selected]


@chz.chz
class VerifierConfig:
    """LLM-as-a-Verifier, called through Tinker's OpenAI-compatible endpoint."""

    model_name: str = DEFAULT_REWARD_MODEL
    # Thinking effort; one of EFFORT_PRESETS.
    effort: float = DEFAULT_REWARD_EFFORT
    # Output budget, shared by the reasoning trace and the verdict.
    max_tokens: int = DEFAULT_REWARD_MAX_TOKENS
    # Repeated verifications K per criterion. Odd repeats swap the prompt
    # slots, so an even K judges every pair equally often in each order.
    n_evaluations: int = 2
    # Criteria names (keys of DEFAULT_CRITERIA); empty = all of them.
    criteria: tuple[str, ...] = ()
    # Longest image side sent to the model.
    image_size: int = 448
    max_concurrency: int = DEFAULT_REWARD_CONCURRENCY
    base_url: str | None = None


@chz.chz
class JudgeConfig:
    """LLM-as-a-Judge baseline: same model, effort and client as the verifier."""

    model_name: str = DEFAULT_REWARD_MODEL
    effort: float = DEFAULT_REWARD_EFFORT
    max_tokens: int = DEFAULT_REWARD_MAX_TOKENS
    # Independent judgments per painting, averaged. 1 is the classic judge.
    n_evaluations: int = 1
    image_size: int = 448
    max_concurrency: int = DEFAULT_REWARD_CONCURRENCY
    base_url: str | None = None


RewardConfig = VerifierConfig | JudgeConfig


@chz.chz
class EvaluatorConfig:
    """The held-out evaluator, on Tinker's native sampler; never enters the reward."""

    model_name: str = DEFAULT_EVALUATOR_MODEL
    # None: model_info's recommended renderer for model_name.
    renderer_name: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0
    n_evaluations: int = 1
    # The judge's single overall criterion, scored 1-10 in one call per
    # painting; () scores the three DEFAULT_CRITERIA in separate calls instead.
    criteria: tuple[str, ...] = (OVERALL_CRITERION[0],)
    image_size: int = 448
    # One checkpoint is 50 prompts x 5 paintings = 250 calls (750 with the
    # three criteria); 2000 keeps every one of them in flight.
    max_concurrency: int = 2000
    base_url: str | None = None


@dataclass
class Verdict:
    """One scoring call, kept for logging/artifacts."""

    criterion: str
    rep: int
    slot_a: int
    """Group index of the painting shown in slot A (judge/evaluator: the only painting)."""
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


def top_is_tied(scores: list[float]) -> bool:
    """True when more than one painting shares the group's highest score."""
    if len(scores) < 2:
        return False
    top = max(scores)
    return sum(abs(s - top) < 1e-9 for s in scores) > 1


def _fit(image: Image.Image, size: int) -> Image.Image:
    im = image.convert("RGB")
    im.thumbnail((size, size))
    return im


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _mean_per_painting(verdicts: list[Verdict], n: int, ok: list[bool]) -> list[float]:
    """Mean score per painting over its usable verdicts; 0.5 when it has none."""
    total = [0.0] * n
    count = [0] * n
    for v, usable in zip(verdicts, ok, strict=True):
        if not usable:
            continue
        total[v.slot_a] += v.score_a
        count[v.slot_a] += 1
        if v.slot_b is not None and v.score_b is not None:
            total[v.slot_b] += v.score_b
            count[v.slot_b] += 1
    return [total[i] / count[i] if count[i] else 0.5 for i in range(n)]


_SCORE_TAG = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


# The absolute scale the judge and the held-out evaluator score on, the
# standard LLM-as-a-Judge 1-10.
SCORE_MIN, SCORE_MAX = 1, 10


def extract_score(text: str, lo: float = SCORE_MIN, hi: float = SCORE_MAX) -> float | None:
    """Last ``<score> n </score>`` in the reply, clamped to [lo, hi] and mapped to [0, 1]."""
    matches = _SCORE_TAG.findall(text or "")
    if not matches:
        return None
    return (max(lo, min(hi, float(matches[-1]))) - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Tinker's OpenAI-compatible endpoint, shared by the verifier and the judge
# ---------------------------------------------------------------------------


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

T = TypeVar("T")


def _is_transient(error: Exception) -> bool:
    import openai

    if isinstance(error, openai.APIConnectionError):  # includes timeouts
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def _with_retries(call: Callable[[], T], what: str) -> T | None:
    """``call()``, retrying transient errors; None when it still fails."""
    for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
        try:
            return call()
        except Exception as e:
            if delay is not None and _is_transient(e):
                logger.info("%s call failed (%s), retry %d in %.0f s", what, e, attempt + 1, delay)
                time.sleep(delay)
                continue
            logger.warning("%s call failed (%s: %s)", what, type(e).__name__, e)
            return None
    raise AssertionError("unreachable")


def make_tinker_openai_client(base_url: str | None, max_concurrency: int) -> Any:
    """An ``openai`` client on Tinker's OpenAI-compatible endpoint."""
    from openai import DefaultHttpxClient, OpenAI

    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise llm_verifier.MissingAPIKeyError("set TINKER_API_KEY to use the Tinker reward model")
    import httpx

    return OpenAI(
        base_url=base_url or TINKER_OPENAI_BASE_URL,
        api_key=key,
        max_retries=5,
        # A call takes 5-60 s; the occasional request that hangs on the
        # endpoint is retried after 5 min instead of stalling the step.
        timeout=300.0,
        # The default pool (1000 connections) would queue part of a step.
        http_client=DefaultHttpxClient(
            limits=httpx.Limits(
                max_connections=max_concurrency, max_keepalive_connections=max_concurrency
            )
        ),
    )


class _TinkerOpenAIScorer:
    """A Tinker OpenAI client plus a thread pool sized to the wanted concurrency."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._client = make_tinker_openai_client(config.base_url, config.max_concurrency)
        # The openai client (and llm_verifier.compare) are synchronous; run
        # calls on a dedicated pool sized to the concurrency this loop wants.
        self._pool = ThreadPoolExecutor(
            max_workers=config.max_concurrency, thread_name_prefix=f"paint-{type(self).__name__}"
        )

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)


# ---------------------------------------------------------------------------
# LLM-as-a-Verifier: pairwise, fine-grained, round-robin tournament
# ---------------------------------------------------------------------------


def round_robin_pairs(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Every unordered pair of ``n`` candidates as a directed (slot A, slot B).

    Which member of a pair starts in slot A is randomized, and the pairs are
    shuffled, so a group's comparisons are not systematically ordered by index;
    odd repeats then re-run each pair with the slots swapped, which is what
    actually cancels the positional bias.
    """
    pairs = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in combinations(range(n), 2)]
    rng.shuffle(pairs)
    return pairs


class PairwiseVerifier(_TinkerOpenAIScorer):
    """LLM-as-a-Verifier reward for a group of paintings (round-robin tournament)."""

    config: VerifierConfig

    def __init__(self, config: VerifierConfig):
        super().__init__(config)
        # llm_verifier's hosted-API path reads the letter distribution at the
        # score tags the model samples itself (Tinker has no assistant
        # prefill), and takes its reasoning effort and budget from the env.
        os.environ["DEEPSEEK_EFFORT"] = effort_preset(config.effort)
        os.environ["DEEPSEEK_MAX_TOKENS"] = str(config.max_tokens)
        self._client._llm_verifier_model = config.model_name
        self._client._llm_verifier_deepseek = True

    def _compare_sync(
        self, problem: str, image_a: bytes, image_b: bytes, criterion: tuple[str, str]
    ) -> tuple[float, float, bool]:
        name, description = criterion
        result = _with_retries(
            lambda: llm_verifier.compare(
                problem,
                SLOT_A,
                SLOT_B,
                criteria={name: description},
                images=[image_a, image_b],
                ground_truth_note=PAIRWISE_NOTE,
                model=self.config.model_name,
                client=self._client,
            ),
            "verifier",
        )
        if result is None:
            # A call that still fails is a tie, as llm_verifier's on_error="tie".
            return 0.5, 0.5, True
        ra, rb = result
        return ra, rb, False

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        """Round-robin rewards for ``images`` (all of them must be valid paintings).

        Every unordered pair of paintings is compared, for every criterion and
        repeat; odd repeats swap the two slots, so with an even
        ``n_evaluations`` each pair is judged equally often in each order and
        the verifier's positional bias cancels pair by pair. A painting's
        reward is the mean of its fine-grained scores over the ``n - 1``
        opponents it met.
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
            r_first, r_second, failed = await self._run(
                self._compare_sync, problem, pngs[first], pngs[second], crit
            )
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

        verdicts = list(await asyncio.gather(*(run(job) for job in jobs)))
        # Failed calls count as the 0.5 tie they were scored as.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, [True] * len(verdicts)),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: one painting at a time, one overall criterion, 1-10
# ---------------------------------------------------------------------------

_ABSOLUTE_PREAMBLE = (
    "You are an expert reviewer of watercolor paintings. You will be given a "
    "painting request and one painting made for it, and you score it on ONE "
    "criterion.\n\n"
    f"{GROUND_TRUTH_NOTE}\n"
)


def absolute_prompt(problem: str, criterion: tuple[str, str] = OVERALL_CRITERION) -> str:
    """The 1-10 single-painting prompt, shared by the judge and the held-out evaluator."""
    name, description = criterion
    return (
        _ABSOLUTE_PREAMBLE + f"\n**Evaluation Guideline — {name}:**\n{description}\n\n"
        f'Score the painting on this criterion ("{name}") from {SCORE_MIN} '
        f"(complete failure) to {SCORE_MAX} (outstanding).\n\n"
        "Reason it through first, then END your reply with exactly this line and "
        f"nothing after it, replacing the placeholder with an integer {SCORE_MIN}-{SCORE_MAX}:\n"
        f"<score> INTEGER_{SCORE_MIN}_TO_{SCORE_MAX} </score>\n"
        f"\n**Task (the painting request):**\n{problem}\n"
        "\nThe painting is attached below.\n\nBegin your analysis now."
    )


class AbsoluteJudge(_TinkerOpenAIScorer):
    """LLM-as-a-Judge reward: each painting scored independently, integer 1-10."""

    config: JudgeConfig

    def _ask_sync(self, prompt: str, png: bytes) -> tuple[str, bool]:
        import base64

        b64 = base64.b64encode(png).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        response = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.config.max_tokens,
                temperature=1.0,
                # The same thinking settings llm_verifier sends for the verifier.
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort_preset(self.config.effort),
                },
            ),
            "judge",
        )
        if response is None:
            return "", True
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason == "length"

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        """Absolute rewards for ``images``; equal integer scores give equal rewards."""
        del seed  # independent scores; nothing to randomize
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        prompt = absolute_prompt(problem)
        pngs = [_png_bytes(_fit(im, self.config.image_size)) for im in images]
        jobs = [(i, rep) for i in range(n) for rep in range(self.config.n_evaluations)]

        async def run(job: tuple[int, int]) -> tuple[Verdict, bool]:
            i, rep = job
            text, truncated = await self._run(self._ask_sync, prompt, pngs[i])
            score = extract_score(text)
            verdict = Verdict(
                criterion=OVERALL_CRITERION[0],
                rep=rep,
                slot_a=i,
                slot_b=None,
                score_a=0.5 if score is None else score,
                score_b=None,
                text=text,
                truncated=truncated or score is None,
            )
            return verdict, score is not None

        pairs = await asyncio.gather(*(run(job) for job in jobs))
        verdicts = [v for v, _ in pairs]
        # A failed or unparseable verdict is left out of the mean; a painting
        # with none at all falls back to 0.5 and is counted in failed_frac.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, [ok for _, ok in pairs]),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# Held-out evaluator: Kimi on Tinker's native sampler, one call per painting and
# criterion, 0-10
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    truncated: bool


class _TinkerVisionClient:
    """A vision model on Tinker's native sampler, prompted through its cookbook renderer."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._sampler: Any = None
        self._renderer: Any = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._sampler is not None:
                return
            import tinker

            from tinker_cookbook import model_info
            from tinker_cookbook.image_processing_utils import get_image_processor
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            name = self.config.model_name
            renderer_name = self.config.renderer_name or model_info.get_recommended_renderer_name(
                name
            )
            self._renderer = get_renderer(
                renderer_name, get_tokenizer(name), image_processor=get_image_processor(name)
            )
            service = tinker.ServiceClient(base_url=self.config.base_url)
            self._sampler = await service.create_sampling_client_async(base_model=name)

    async def ask(self, prompt: str, image: Image.Image, num_samples: int) -> list[Reply]:
        """``num_samples`` replies to one prompt, from a single sampling request."""
        await self._ensure_started()
        import tinker

        from tinker_cookbook.renderers import Message, get_text_content

        message = Message(
            role="user",
            content=[{"type": "text", "text": prompt}, {"type": "image", "image": image}],
        )
        model_input = self._renderer.build_generation_prompt([message])
        params = tinker.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        try:
            async with self._semaphore:
                result = await self._sampler.sample_async(
                    model_input, num_samples=num_samples, sampling_params=params
                )
        except Exception as e:  # still failing after the SDK's own retries
            logger.warning("held-out evaluator call failed (%s: %s)", type(e).__name__, e)
            return [Reply(text="", truncated=True)] * num_samples
        replies = []
        for sequence in result.sequences:
            reply, _ = self._renderer.parse_response(sequence.tokens)
            text = get_text_content(reply)
            replies.append(Reply(text=text, truncated=sequence.stop_reason == "length" or not text))
        return replies


class HeldoutEvaluator:
    """Independent absolute scores per painting and criterion, no pairing.

    By default there is one criterion, the judge's overall one, so each
    painting is one request; with several criteria each is its own request,
    so one criterion's reasoning never colors another's. All of a group's requests are submitted at once (and
    every group's, since groups are evaluated concurrently), up to
    ``max_concurrency`` in flight; ``n_evaluations`` repeats come from
    ``num_samples`` on the same request.
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._client = _TinkerVisionClient(config)

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        del seed  # independent scores; nothing to randomize
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        criteria = resolve_criteria(self.config.criteria)
        fitted = [_fit(im, self.config.image_size) for im in images]
        jobs = [(i, crit) for i in range(n) for crit in criteria]
        per_job = await asyncio.gather(
            *(
                self._client.ask(
                    absolute_prompt(problem, crit), fitted[i], self.config.n_evaluations
                )
                for i, crit in jobs
            )
        )
        verdicts: list[Verdict] = []
        usable: list[bool] = []
        for (i, (name, _)), replies in zip(jobs, per_job, strict=True):
            for rep, reply in enumerate(replies):
                score = extract_score(reply.text)
                verdicts.append(
                    Verdict(
                        criterion=name,
                        rep=rep,
                        slot_a=i,
                        slot_b=None,
                        score_a=0.5 if score is None else score,
                        score_b=None,
                        text=reply.text,
                        truncated=reply.truncated or score is None,
                    )
                )
                usable.append(score is not None)
        # Failed or unparseable verdicts are left out of the mean; a painting
        # with none at all falls back to a 0.5 tie.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, usable),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


Scorer = PairwiseVerifier | AbsoluteJudge
_SCORERS: dict[RewardConfig, Scorer] = {}
_EVALUATORS: dict[EvaluatorConfig, HeldoutEvaluator] = {}


def get_shared_scorer(config: RewardConfig) -> Scorer:
    """One reward scorer per config per process, so the client and its pool are reused."""
    if config not in _SCORERS:
        _SCORERS[config] = (
            PairwiseVerifier(config)
            if isinstance(config, VerifierConfig)
            else AbsoluteJudge(config)
        )
    return _SCORERS[config]


def get_shared_evaluator(config: EvaluatorConfig) -> HeldoutEvaluator:
    """One evaluator per config per process."""
    if config not in _EVALUATORS:
        _EVALUATORS[config] = HeldoutEvaluator(config)
    return _EVALUATORS[config]
