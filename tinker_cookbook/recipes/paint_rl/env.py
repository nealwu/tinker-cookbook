"""RL environment: the policy writes a p5.brush sketch, the group is verified on the rendered images.

Reward for one rollout (all terms in [0, 1]):

    reward = w_compile * compiled + w_length * length_ok + w_score * score

* ``compiled`` — the sketch ran in the headless renderer, used the brush
  library, and painted a non-blank canvas (the compile gate).
* ``length_ok`` — the code length sits inside ``[min_code_chars, max_code_chars]``.
* ``score`` — the pairwise verifier's score of the rendered painting; 0 for
  rollouts that failed the compile gate (they are excluded from the
  tournament).

Held-out groups are additionally scored by an independent strong evaluator
(``gemini-3.8-flash``) that never feeds back into training; it reports
``eval_strong/score``, a measure of quality that does not depend on the
training reward.

Every rollout's JavaScript, PNG, raw response, and verifier scores are
written under ``artifact_dir/<split>/step_XXXX/<prompt id>/`` and indexed in
``artifact_dir/index.jsonl`` so the paintings can be browsed as training goes
(see ``gallery.py``).
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chz
import tinker
from PIL import Image

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.paint_rl.directives import (
    DIRECTED_TIERS,
    build_directed_heldout,
    build_directed_prompts,
)
from tinker_cookbook.recipes.paint_rl.prompts import (
    PaintPrompt,
    build_prompt_splits,
    split_summary,
    system_prompt,
)
from tinker_cookbook.recipes.paint_rl.render import (
    RenderBackend,
    RenderResult,
    extract_code,
    get_shared_renderer,
)
from tinker_cookbook.recipes.paint_rl.verifier import (
    EvaluatorConfig,
    GroupScores,
    VerifierConfig,
    get_shared_evaluator,
    get_shared_verifier,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


@chz.chz
class RewardWeights:
    compile_gate: float = 0.05
    length_gate: float = 0.05
    score: float = 0.90
    min_code_chars: int = 300
    max_code_chars: int = 8000


class PaintEnv(Env):
    """Single-turn: prompt in, one sketch out. Reward is assigned at group level."""

    def __init__(self, prompt: PaintPrompt, renderer: Renderer, system: str, effort: float):
        self.prompt = prompt
        self.renderer = renderer
        self.system = system
        self.effort = effort
        self.response_text: str = ""
        self.code: str | None = None
        self.truncated: bool = False

    @property
    def messages(self) -> list[Message]:
        return [
            Message(role="system", content=self.system),
            Message(role="user", content=self.prompt.text),
        ]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if isinstance(self.renderer, TmlV0Renderer):
            prompt = self.renderer.build_generation_prompt(self.messages, effort=self.effort)
        else:
            prompt = self.renderer.build_generation_prompt(self.messages)
        return prompt, self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        self.response_text = get_text_content(message)
        self.truncated = (extra or {}).get(
            "stop_reason"
        ) == "length" or not termination.is_stop_sequence
        self.code = None if self.truncated else extract_code(self.response_text)
        metrics: Metrics = {"policy/truncated": float(self.truncated)}
        if self.truncated:
            metrics["stop/max_tokens"] = 1.0
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
            logs={
                "truncated": int(self.truncated),
                "has_code": int(self.code is not None),
            },
        )


@dataclass
class RolloutRecord:
    """What gets written to the artifact index for one rollout."""

    split: str
    iteration: int
    prompt_id: str
    prompt: str
    tier: int
    kind: str
    index: int
    reward: float
    score: float
    eval_score: float | None
    """Held-out evaluator (Gemini) score; None on the train split or when off."""
    compiled: bool
    length_ok: bool
    truncated: bool
    render_error: str | None
    code_chars: int
    brush_calls: int
    js_path: str | None
    png_path: str | None


def _image_html(png: bytes, size: int = 256) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" style="border:1px solid #ccc"/>'


@dataclass(frozen=True)
class PaintEnvGroupBuilder(EnvGroupBuilder):
    prompt: PaintPrompt
    renderer: Renderer
    system: str
    # Policy thinking effort; only Inkling (tml_v0) models take one.
    policy_effort: float
    group_size: int
    canvas_size: int
    verifier_config: VerifierConfig
    # Independent evaluator for held-out groups; scores nothing on the train
    # split and never enters the reward. None turns it off.
    evaluator_config: EvaluatorConfig | None
    weights: RewardWeights
    artifact_dir: str | None
    split: str
    iteration: int
    seed: int = 0
    render_concurrency: int = 16
    render_backend: RenderBackend = "local"
    # Per-sketch render timeout. A sketch that times out fails the compile gate.
    render_timeout_s: float = 300.0
    # False skips the training verifier entirely (score 0): used by the
    # held-out evaluator, which only needs the independent evaluator's score.
    score_with_training_reward: bool = True
    # Step to file artifacts under; None infers it (see _artifact_iteration).
    artifact_step: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            PaintEnv(self.prompt, self.renderer, self.system, self.policy_effort)
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        # The cookbook aggregates metrics per tag whenever a tag selects a
        # strict subset of the batch, so held-out metrics come out per tier
        # (test/env/tier3/...) and per kind (test/env/novel_subject/...).
        # Directed prompts (tiers 4-8) are tagged "directed" in place of the
        # split, so env/train/... and test/env/test/... keep covering only the
        # base prompts once directed ones are mixed in.
        directed = self.prompt.tier in DIRECTED_TIERS
        tags = ["paint", "directed" if directed else self.split, f"tier{self.prompt.tier}"]
        if self.prompt.kind != self.split:
            tags.append(self.prompt.kind)
        return tags

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        envs = [e for e in env_group if isinstance(e, PaintEnv)]
        assert len(envs) == len(trajectory_group)
        renderer = get_shared_renderer(
            self.canvas_size, self.render_concurrency, self.render_backend, self.render_timeout_s
        )

        # 1) render every sketch (compile gate)
        t_render = time.monotonic()
        renders: list[RenderResult] = await asyncio.gather(
            *(
                renderer.render(env.code)
                if env.code is not None
                else _failed_render(
                    "no code block in response" if not env.truncated else "truncated"
                )
                for env in envs
            )
        )
        render_seconds = time.monotonic() - t_render

        # 2) verify the valid paintings as a group
        valid = [i for i, r in enumerate(renders) if r.ok]
        images: list[Image.Image] = []
        for i in valid:
            image = renders[i].image()
            assert image is not None
            images.append(image)
        t_verify = time.monotonic()
        group_seed = random.Random(f"{self.seed}:{self.iteration}:{self.prompt.id}").randrange(
            1 << 30
        )
        if images and self.score_with_training_reward:
            verifier = get_shared_verifier(self.verifier_config)
            group_scores = await verifier.score_group(self.prompt.text, images, seed=group_seed)
        else:
            group_scores = GroupScores(scores=[])
        verify_seconds = time.monotonic() - t_verify
        # Empty when the training reward is skipped (held-out evaluation only).
        score_by_index = (
            dict(zip(valid, group_scores.scores, strict=True)) if group_scores.scores else {}
        )

        # 2b) held-out only: an independent evaluator that never touches the
        # reward, so progress is measured on a scale the policy cannot game.
        eval_scores = GroupScores(scores=[])
        if images and self.evaluator_config is not None and self.split != "train":
            evaluator = get_shared_evaluator(self.evaluator_config)
            eval_scores = await evaluator.score_group(self.prompt.text, images, seed=group_seed)
        # Empty on the train split, where the evaluator does not run at all.
        eval_by_index = (
            dict(zip(valid, eval_scores.scores, strict=True)) if eval_scores.scores else {}
        )

        # 3) compose rewards
        results: list[tuple[float, Metrics]] = []
        records: list[RolloutRecord] = []
        w = self.weights
        for i, (env, render) in enumerate(zip(envs, renders, strict=True)):
            code_chars = len(env.code or "")
            compiled = render.ok
            length_ok = compiled and w.min_code_chars <= code_chars <= w.max_code_chars
            score = score_by_index.get(i, 0.0)
            reward = w.compile_gate * compiled + w.length_gate * length_ok + w.score * score
            metrics: Metrics = {
                "reward/total": reward,
                "reward/compile_ok": float(compiled),
                "reward/length_ok": float(length_ok),
                "reward/score": score,
                "code/chars": code_chars,
                "code/brush_calls": render.n_brush_calls,
                "code/missing": float(env.code is None),
                "render/blank": float(render.blank),
                "time/render_group_s": render_seconds,
                "time/verify_group_s": verify_seconds,
                "verifier/calls_per_rollout": group_scores.n_calls / len(envs),
                "verifier/failed_frac": (
                    group_scores.n_truncated / group_scores.n_calls if group_scores.n_calls else 0.0
                ),
            }
            if compiled:
                metrics["reward/score_if_compiled"] = score
            if eval_scores.n_calls:
                metrics["eval_strong/score"] = eval_by_index.get(i, 0.0)
                metrics["eval_strong/failed_frac"] = eval_scores.n_truncated / eval_scores.n_calls
                if compiled:
                    metrics["eval_strong/score_if_compiled"] = eval_by_index[i]
            results.append((reward, metrics))
            records.append(
                RolloutRecord(
                    split=self.split,
                    iteration=self.iteration,
                    prompt_id=self.prompt.id,
                    prompt=self.prompt.text,
                    tier=self.prompt.tier,
                    kind=self.prompt.kind,
                    index=i,
                    reward=reward,
                    score=score,
                    eval_score=eval_by_index.get(i, 0.0) if eval_scores.n_calls else None,
                    compiled=compiled,
                    length_ok=length_ok,
                    truncated=env.truncated,
                    render_error=render.error,
                    code_chars=code_chars,
                    brush_calls=render.n_brush_calls,
                    js_path=None,
                    png_path=None,
                )
            )

        # 4) persist artifacts + logtree report
        if self.artifact_dir is not None:
            self._save_artifacts(envs, renders, records, group_scores, eval_scores)
        self._log(envs, renders, records, group_scores)
        return results

    def _artifact_iteration(self, root: Path) -> int:
        """Training step to file this group under.

        The train split knows its batch index. Held-out evaluation reuses
        one set of builders (always ``iteration=0``), so for other splits the
        step is inferred from the newest ``train/step_XXXX`` directory: an
        evaluation that runs after training step ``n`` is filed as ``n + 1``.
        """
        if self.artifact_step is not None:
            return self.artifact_step
        if self.split == "train":
            return self.iteration
        steps = [
            int(p.name.split("_", 1)[1])
            for p in (root / "train").glob("step_*")
            if p.name.split("_", 1)[1].isdigit()
        ]
        return max(steps) + 1 if steps else 0

    def _save_artifacts(
        self,
        envs: list[PaintEnv],
        renders: list[RenderResult],
        records: list[RolloutRecord],
        group_scores: GroupScores,
        eval_scores: GroupScores,
    ) -> None:
        root = Path(self.artifact_dir or ".")
        iteration = self._artifact_iteration(root)
        for rec in records:
            rec.iteration = iteration
        group_dir = root / self.split / f"step_{iteration:04d}" / self.prompt.id
        group_dir.mkdir(parents=True, exist_ok=True)
        for env, render, rec in zip(envs, renders, records, strict=True):
            (group_dir / f"g{rec.index}.response.txt").write_text(
                env.response_text, encoding="utf-8"
            )
            if env.code is not None:
                js_path = group_dir / f"g{rec.index}.js"
                js_path.write_text(env.code, encoding="utf-8")
                rec.js_path = str(js_path.relative_to(root))
            if render.png is not None:
                png_path = group_dir / f"g{rec.index}.png"
                png_path.write_bytes(render.png)
                rec.png_path = str(png_path.relative_to(root))
        summary = {
            "prompt": asdict(self.prompt),
            "split": self.split,
            "iteration": iteration,
            "rollouts": [asdict(r) for r in records],
            "verdicts": [asdict(v) for v in group_scores.verdicts],
            "eval_verdicts": [asdict(v) for v in eval_scores.verdicts],
        }
        (group_dir / "group.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        with (root / "index.jsonl").open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec)) + "\n")

    def _log(
        self,
        envs: list[PaintEnv],
        renders: list[RenderResult],
        records: list[RolloutRecord],
        group_scores: GroupScores,
    ) -> None:
        with logtree.scope_header(f"Paint group: {self.prompt.text}"):
            cells: list[str] = []
            for render, rec in zip(renders, records, strict=True):
                img = _image_html(render.png) if render.png is not None else "<i>(no image)</i>"
                caption = (
                    f"#{rec.index} reward={rec.reward:.3f} score={rec.score:.3f} "
                    + (f"gemini={rec.eval_score:.2f} " if rec.eval_score is not None else "")
                    + f"chars={rec.code_chars}"
                    + (
                        f"<br/><span style='color:#b00'>{html.escape(rec.render_error)}</span>"
                        if rec.render_error
                        else ""
                    )
                )
                cells.append(
                    f"<div style='display:inline-block;margin:6px;text-align:center;vertical-align:top;max-width:270px'>"
                    f"{img}<br/><small>{caption}</small></div>"
                )
            logtree.log_html("".join(cells))
            for env, rec in zip(envs, records, strict=True):
                if env.code is not None:
                    logtree.details(env.code, summary=f"Sketch #{rec.index} (JavaScript)")
            for v in group_scores.verdicts:
                label = (
                    f"{v.criterion} rep{v.rep}: #{v.slot_a} vs #{v.slot_b} -> {v.score_a:.2f}/{v.score_b:.2f}"
                    if v.slot_b is not None
                    else f"{v.criterion} rep{v.rep}: #{v.slot_a} -> {v.score_a:.2f}"
                )
                logtree.details(v.text, summary=label)


async def _failed_render(reason: str) -> RenderResult:
    return RenderResult(
        ok=False, png=None, error=reason, n_brush_calls=0, blank=True, duration_s=0.0
    )


class PaintRLDataset(RLDataset):
    """Cycles through ``prompts`` in a seeded, per-epoch shuffled order."""

    def __init__(
        self,
        prompts: Sequence[PaintPrompt],
        batch_size: int,
        n_batches: int,
        make_builder: Callable[[PaintPrompt, int], PaintEnvGroupBuilder],
        seed: int,
    ):
        if not prompts:
            raise ValueError("no prompts")
        self.prompts = list(prompts)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.make_builder = make_builder
        self.seed = seed

    def _prompt_for(self, flat_index: int) -> PaintPrompt:
        n = len(self.prompts)
        epoch, offset = divmod(flat_index, n)
        order = list(range(n))
        random.Random(f"{self.seed}:{epoch}").shuffle(order)
        return self.prompts[order[offset]]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self.make_builder(self._prompt_for(index * self.batch_size + j), index)
            for j in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.n_batches


def build_run_prompts(
    n_train: int,
    n_test: int,
    seed: int,
    n_directed: int = 0,
    n_directed_test: int = 0,
) -> tuple[list[PaintPrompt], list[PaintPrompt]]:
    """A run's (train, test) prompts: the base split, then any directed prompts appended.

    With no directed prompts this is exactly :func:`build_prompt_splits`.
    Both directed sets are built against the base held-out prompts, so
    neither reuses a held-out (subject, colour, style) triple.
    """
    train, test = build_prompt_splits(
        n_train=n_train,
        n_test_novel_subject=n_test // 2,
        n_test_novel_combo=n_test - n_test // 2,
        seed=seed,
    )
    directed = build_directed_prompts(n_directed, seed=seed, exclude=test) if n_directed else []
    directed_test = (
        build_directed_heldout(n_directed_test, seed=seed, exclude=test) if n_directed_test else []
    )
    return train + directed, test + directed_test


@chz.chz
class PaintRLDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    n_batches: int
    policy_effort: float = 0.7
    verifier_config: VerifierConfig = chz.field(default_factory=VerifierConfig)
    # Held-out evaluator. ``None`` turns the extra pass off.
    evaluator_config: EvaluatorConfig | None = chz.field(default_factory=EvaluatorConfig)
    weights: RewardWeights = chz.field(default_factory=RewardWeights)
    canvas_size: int = 512
    render_concurrency: int = 16
    render_backend: RenderBackend = "local"
    render_timeout_s: float = 300.0
    n_train_prompts: int = 256
    # Held-out prompts, split evenly: half novel subjects (never trained on in
    # any form), half novel combinations of seen subject / colour / style.
    n_test_prompts: int = 16
    # Directed prompts (directives.py): a base prompt plus watercolour
    # directions in tiers 4-8, appended to the training prompts. 0 = off.
    n_directed_prompts: int = 0
    # Held-out directed prompts (kind novel_direction, phrasings never used in
    # training), appended to the held-out prompts. 0 = off.
    n_directed_test: int = 0
    test_group_size: int = 5
    seed: int = 0
    artifact_dir: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name_for_tokenizer))
        train_prompts, test_prompts = build_run_prompts(
            n_train=self.n_train_prompts,
            n_test=self.n_test_prompts,
            seed=self.seed,
            n_directed=self.n_directed_prompts,
            n_directed_test=self.n_directed_test,
        )
        logger.info(
            "Prompt split: train %s | test %s",
            split_summary(train_prompts),
            split_summary(test_prompts),
        )
        if self.artifact_dir is not None:
            # The exact prompts this run trained and evaluated on, for the record.
            root = Path(self.artifact_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "prompts.json").write_text(
                json.dumps(
                    {
                        "train": [asdict(p) for p in train_prompts],
                        "test": [asdict(p) for p in test_prompts],
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )

        def make_builder(
            split: str, group_size: int
        ) -> Callable[[PaintPrompt, int], PaintEnvGroupBuilder]:
            def build(prompt: PaintPrompt, iteration: int) -> PaintEnvGroupBuilder:
                return PaintEnvGroupBuilder(
                    prompt=prompt,
                    renderer=renderer,
                    system=system,
                    policy_effort=self.policy_effort,
                    group_size=group_size,
                    canvas_size=self.canvas_size,
                    verifier_config=self.verifier_config,
                    evaluator_config=self.evaluator_config,
                    weights=self.weights,
                    artifact_dir=self.artifact_dir,
                    split=split,
                    iteration=iteration,
                    seed=self.seed,
                    render_concurrency=self.render_concurrency,
                    render_backend=self.render_backend,
                    render_timeout_s=self.render_timeout_s,
                )

            return build

        system = system_prompt(self.canvas_size)
        train = PaintRLDataset(
            train_prompts,
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            make_builder=make_builder("train", self.group_size),
            seed=self.seed,
        )
        test = None
        if test_prompts:
            test = PaintRLDataset(
                test_prompts,
                batch_size=len(test_prompts),
                n_batches=1,
                make_builder=make_builder("test", self.test_group_size),
                seed=self.seed,
            )
        return train, test
