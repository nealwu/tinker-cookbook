"""Held-out evaluation of a paint_rl run's checkpoints, in a separate process.

    # alongside training (follows checkpoints.jsonl until the final checkpoint)
    python -m tinker_cookbook.recipes.paint_rl.eval_checkpoints log_path=/tmp/paint_rl/verifier

    # or afterwards, on every 10th checkpoint only
    python -m tinker_cookbook.recipes.paint_rl.eval_checkpoints \
        log_path=/tmp/paint_rl/verifier every=10 follow=False

Training saves sampler weights every ``save_every`` steps and lists them in
``<log_path>/checkpoints.jsonl``. This script evaluates the base model (step
0) and then each listed checkpoint whose step is a multiple of ``every``, plus
the final one, on the run's held-out prompts: the policy samples
``test_group_size`` paintings per prompt, they are rendered, and the held-out
evaluator (Gemini) scores them. The training verifier is skipped unless
``with_verifier=True``. Training is never blocked, stays on-policy, and a
crashed or lagging evaluation can be re-run later on any checkpoint.

Everything goes under ``<log_path>/heldout_eval/``: ``metrics.jsonl`` (one row
per step, the same ``test/env/...`` keys training would log, headline
``test/env/all/eval_strong/score``), ``step_XXXXXX.html`` rollout reports and
``paintings/test/step_XXXX/`` artifacts. Steps already in ``metrics.jsonl``
are skipped, so the script can be stopped and restarted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import tinker

from tinker_cookbook.recipes.paint_rl.env import (
    PaintEnvGroupBuilder,
    PaintRLDataset,
    RewardWeights,
)
from tinker_cookbook.recipes.paint_rl.prompts import (
    PaintPrompt,
    build_prompt_splits,
    system_prompt,
)
from tinker_cookbook.recipes.paint_rl.render import get_shared_renderer
from tinker_cookbook.recipes.paint_rl.verifier import EvaluatorConfig, VerifierConfig
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.git_rev import recipe_user_metadata

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # The training run to evaluate (its config.json and checkpoints.jsonl).
    log_path: str
    # Evaluate checkpoints whose step is a multiple of this (the final one always).
    every: int = 5
    include_base: bool = True
    # Keep polling checkpoints.jsonl until the final checkpoint has been evaluated.
    follow: bool = True
    poll_seconds: float = 60.0
    # Also score held-out groups with the training verifier (for verifier vs
    # evaluator agreement). Off by default: it is 60 extra calls per group.
    with_verifier: bool = False
    # Default: <log_path>/heldout_eval
    out_dir: str | None = None
    # Concurrent calls to the held-out evaluator. Policy sampling on Tinker is
    # not throttled: all held-out rollouts are submitted at once.
    eval_concurrency: int = 50
    # Rendering here is not time-critical and shares the CPU with training:
    # few pages at once and a long timeout, so a slow render is never scored
    # as a failed sketch.
    render_concurrency: int = 4
    render_timeout_s: float = 600.0
    base_url: str | None = None


@dataclass(frozen=True)
class Target:
    step: int
    name: str
    sampler_path: str | None
    """None means the base model."""


def _rebuild(cls: type[Any], raw: dict[str, Any], **overrides: Any) -> Any:
    """Rebuild a VerifierConfig / EvaluatorConfig from its config.json form."""
    known = set(chz.chz_fields(cls))
    fields = {k: v for k, v in raw.items() if k in known}
    fields["criteria"] = tuple(fields.get("criteria") or ())
    fields.update(overrides)
    return cls(**fields)


def _read_checkpoints(log_path: Path) -> list[dict[str, Any]]:
    path = log_path / "checkpoints.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("sampler_path")]


def _done_steps(out_dir: Path) -> set[int]:
    path = out_dir / "metrics.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["step"] for line in path.read_text().splitlines() if line.strip()}


def _targets(checkpoints: list[dict[str, Any]], every: int, include_base: bool) -> list[Target]:
    targets = [Target(0, "base", None)] if include_base else []
    for row in checkpoints:
        step = int(row.get("batch", 0))
        if row["name"] == "final" or (every > 0 and step % every == 0):
            targets.append(Target(step, row["name"], row["sampler_path"]))
    by_step: dict[int, Target] = {}
    for t in targets:  # a periodic and the final checkpoint can share a step
        by_step.setdefault(t.step, t)
    return [by_step[s] for s in sorted(by_step)]


class HeldoutEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log_path = Path(cfg.log_path)
        self.out_dir = Path(cfg.out_dir or self.log_path / "heldout_eval")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        run = json.loads((self.log_path / "config.json").read_text())

        data = run["dataset_builder"]
        self.data = data
        self.model_name: str = run["model_name"]
        self.max_tokens: int = run["max_tokens"]
        self.renderer = get_renderer(data["renderer_name"], get_tokenizer(self.model_name))
        _, self.test_prompts = build_prompt_splits(
            n_train=data["n_train_prompts"],
            n_test_novel_subject=data["n_test_prompts"] // 2,
            n_test_novel_combo=data["n_test_prompts"] - data["n_test_prompts"] // 2,
            seed=data["seed"],
        )
        # Runs logged before the rename used judge_config / eval_judge_config.
        verifier_raw = data.get("verifier_config") or data["judge_config"]
        evaluator_raw = data.get("evaluator_config") or data.get("eval_judge_config")
        if not evaluator_raw:
            raise ValueError("this run has no held-out evaluator configured (eval_model=None)")
        self.verifier_config: VerifierConfig = _rebuild(VerifierConfig, verifier_raw)
        self.evaluator_config: EvaluatorConfig = _rebuild(
            EvaluatorConfig, evaluator_raw, max_concurrency=cfg.eval_concurrency
        )
        self.weights = RewardWeights(**data["weights"])
        self.service = tinker.ServiceClient(
            base_url=cfg.base_url, user_metadata=recipe_user_metadata("recipe_paint_rl_eval")
        )

    def _dataset(self, step: int) -> PaintRLDataset:
        system = system_prompt(self.data["canvas_size"])

        def build(prompt: PaintPrompt, iteration: int) -> PaintEnvGroupBuilder:
            return PaintEnvGroupBuilder(
                prompt=prompt,
                renderer=self.renderer,
                system=system,
                policy_effort=self.data["policy_effort"],
                group_size=self.data["test_group_size"],
                canvas_size=self.data["canvas_size"],
                verifier_config=self.verifier_config,
                evaluator_config=self.evaluator_config,
                weights=self.weights,
                artifact_dir=str(self.out_dir / "paintings"),
                split="test",
                iteration=iteration,
                seed=self.data["seed"],
                render_concurrency=self.cfg.render_concurrency,
                render_timeout_s=self.cfg.render_timeout_s,
                render_backend=self.data["render_backend"],
                score_with_training_reward=self.cfg.with_verifier,
                artifact_step=step,
            )

        return PaintRLDataset(
            self.test_prompts,
            batch_size=len(self.test_prompts),
            n_batches=1,
            make_builder=build,
            seed=self.data["seed"],
        )

    async def evaluate(self, target: Target) -> dict[str, float]:
        if target.sampler_path is None:
            client = await self.service.create_sampling_client_async(base_model=self.model_name)
        else:
            client = await self.service.create_sampling_client_async(model_path=target.sampler_path)
        evaluator = RLTestSetEvaluator(self._dataset(target.step), max_tokens=self.max_tokens)
        with logtree.init_trace(
            f"Held-out evaluation, step {target.step} ({target.name})",
            path=str(self.out_dir / f"step_{target.step:06d}.html"),
        ):
            metrics = await evaluator(client)
        row = {"step": target.step, "checkpoint": target.name, **metrics}
        with (self.out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            "step %d (%s): eval_strong/score=%s compile_ok=%s",
            target.step,
            target.name,
            metrics.get("test/env/all/eval_strong/score"),
            metrics.get("test/env/all/reward/compile_ok"),
        )
        return metrics

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            # Close the headless browsers, or interpreter shutdown waits on them.
            await get_shared_renderer(
                self.data["canvas_size"],
                self.cfg.render_concurrency,
                self.data["render_backend"],
                self.cfg.render_timeout_s,
            ).close()

    async def _run(self) -> None:
        while True:
            checkpoints = _read_checkpoints(self.log_path)
            done = _done_steps(self.out_dir)
            todo = [
                t
                for t in _targets(checkpoints, self.cfg.every, self.cfg.include_base)
                if t.step not in done
            ]
            for target in todo:
                await self.evaluate(target)
            # A run extended past its first max_steps has an earlier "final"
            # checkpoint, so finishing means a final one at the current target.
            max_steps = json.loads((self.log_path / "config.json").read_text()).get("max_steps")
            finished = any(
                c["name"] == "final" and (max_steps is None or int(c.get("batch", 0)) >= max_steps)
                for c in checkpoints
            )
            if finished or not self.cfg.follow:
                return
            await asyncio.sleep(self.cfg.poll_seconds)


async def main(cfg: Config) -> None:
    await HeldoutEvaluator(cfg).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(chz.entrypoint(Config)))
