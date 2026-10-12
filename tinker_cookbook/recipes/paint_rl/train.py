"""RL launcher: teach Inkling-Small to paint watercolors with p5.brush code.

    python -m tinker_cookbook.recipes.paint_rl.train log_path=/tmp/paint_rl/verifier

The policy is ``thinkingmachines/Inkling-Small`` on Tinker. It is rewarded by
the pairwise, fine-grained LLM-as-a-Verifier round-robin tournament, run by
Inkling-Small itself at low thinking effort. Held-out groups are additionally
scored by ``gemini-3.8-flash``, which never feeds back into training.

Needs ``TINKER_API_KEY``, plus ``GEMINI_API_KEY`` for the held-out evaluator
(``eval_model=None`` turns it off).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.paint_rl.env import PaintRLDatasetBuilder, RewardWeights
from tinker_cookbook.recipes.paint_rl.render import RenderBackend
from tinker_cookbook.recipes.paint_rl.verifier import (
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_VERIFIER_MODEL,
    EvaluatorConfig,
    VerifierConfig,
)
from tinker_cookbook.rl.train import AsyncConfig, Config, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Policy
    model_name: str = "thinkingmachines/Inkling-Small"
    lora_rank: int = 32
    renderer_name: str | None = None  # default: tml_v0
    load_checkpoint_path: str | None = None
    # Thinking effort in [0, 1); 0.7 is the "medium" preset. Sketches land at
    # ~1.1k tokens at this effort, so 8192 leaves plenty of headroom.
    policy_effort: float = 0.7
    max_tokens: int = 8192
    temperature: float = 1.0

    # Optimisation. Inkling has no calibrated get_lr(); sweep around this.
    learning_rate: float = 4e-5
    group_size: int = 5
    groups_per_batch: int = 8
    max_steps: int = 300
    num_substeps: int = 1
    kl_penalty_coef: float = 0.0
    loss_fn: LossFnType = "importance_sampling"
    max_steps_off_policy: int | None = None

    # Reward: the training verifier.
    verifier_model: str = DEFAULT_VERIFIER_MODEL
    verifier_effort: float = 0.2  # the "low" preset
    # A call that never reaches its score tags has no distribution to read and
    # falls back to a 0.5 tie, so the budget is cheap insurance.
    verifier_max_tokens: int = 8192
    verifier_n_evaluations: int = 2
    criteria: tuple[str, ...] = ()
    # Concurrent verifier calls. One step's tournament is 8 groups x 60 = 480
    # calls; measured on Tinker's endpoint, 480 in flight finished a step's worth
    # in 18 s vs 25 s at 128, with no errors.
    verifier_concurrency: int = 480
    weights: RewardWeights = chz.field(default_factory=RewardWeights)

    # Held-out evaluator: independent of the reward, never trained on.
    # eval_model=None turns it off.
    eval_model: str | None = DEFAULT_EVALUATOR_MODEL
    eval_max_tokens: int = 8192
    eval_n_evaluations: int = 1
    eval_concurrency: int = 50

    # Data / rendering
    # Training prompts are balanced over subjects, colours and styles. Held-out
    # prompts split evenly into novel-subject and novel-combination halves.
    n_train_prompts: int = 256
    n_test_prompts: int = 16
    test_group_size: int = 5
    canvas_size: int = 512
    render_concurrency: int = 16
    # Sketches grow longer over training (about 2x by step 150), and software
    # WebGL rendering of a long one can pass 2 min when the CPU is shared; a
    # timeout scores 0, so leave plenty of headroom.
    render_timeout_s: float = 300.0
    render_backend: RenderBackend = "local"  # 'modal' after `modal deploy .../render_modal.py`
    seed: int = 0

    # Logging
    log_path: str | None = None
    artifact_dir: str | None = None  # default: <log_path>/paintings
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Held-out evaluation inside the training loop blocks training, so it is
    # off by default; run eval_checkpoints.py alongside instead (it evaluates
    # every saved checkpoint in a separate process). Set >0 to evaluate inline.
    eval_every: int = 0
    # Checkpoint (training state + sampler weights) every 5 steps: 60 over a
    # 300-step run, plus the final one.
    save_every: int = 5
    # Lifetime of the periodic checkpoints on Tinker, in seconds. None keeps
    # them all indefinitely (the cookbook default expires them after 7 days);
    # the final checkpoint is always kept.
    checkpoint_ttl_seconds: int | None = None
    compute_post_kl: bool = False
    remove_constant_reward_groups: bool = False
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    base_url: str | None = None


def build_scorer_configs(cli: CLIConfig) -> tuple[VerifierConfig, EvaluatorConfig | None]:
    """(training verifier, held-out evaluator); the evaluator is None if off."""
    verifier_config = VerifierConfig(
        model_name=cli.verifier_model,
        effort=cli.verifier_effort,
        max_tokens=cli.verifier_max_tokens,
        n_evaluations=cli.verifier_n_evaluations,
        criteria=cli.criteria,
        max_concurrency=cli.verifier_concurrency,
    )
    evaluator_config = (
        EvaluatorConfig(
            model_name=cli.eval_model,
            max_tokens=cli.eval_max_tokens,
            n_evaluations=cli.eval_n_evaluations,
            criteria=cli.criteria,
            max_concurrency=cli.eval_concurrency,
        )
        if cli.eval_model is not None
        else None
    )
    return verifier_config, evaluator_config


async def cli_main(cli: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli.model_name,
        explicit_renderer_name=cli.renderer_name,
        load_checkpoint_path=cli.load_checkpoint_path,
        base_url=cli.base_url,
    )
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"paint-{cli.model_name.split('/')[-1]}-r{cli.lora_rank}"
        f"-lr{cli.learning_rate}-g{cli.group_size}x{cli.groups_per_batch}-{stamp}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/paint_rl/{run_name}"
    artifact_dir = cli.artifact_dir or str(Path(log_path) / "paintings")

    verifier_config, evaluator_config = build_scorer_configs(cli)
    dataset_builder = PaintRLDatasetBuilder(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        batch_size=cli.groups_per_batch,
        group_size=cli.group_size,
        n_batches=cli.max_steps,
        policy_effort=cli.policy_effort,
        verifier_config=verifier_config,
        evaluator_config=evaluator_config,
        weights=cli.weights,
        canvas_size=cli.canvas_size,
        render_concurrency=cli.render_concurrency,
        render_timeout_s=cli.render_timeout_s,
        render_backend=cli.render_backend,
        n_train_prompts=cli.n_train_prompts,
        n_test_prompts=cli.n_test_prompts,
        test_group_size=cli.test_group_size,
        seed=cli.seed,
        artifact_dir=artifact_dir,
    )
    config = Config(
        recipe_name="recipe_paint_rl",
        learning_rate=cli.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli.model_name,
        renderer_name=renderer_name,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or run_name,
        log_path=log_path,
        base_url=cli.base_url,
        load_checkpoint_path=cli.load_checkpoint_path,
        compute_post_kl=cli.compute_post_kl,
        kl_penalty_coef=cli.kl_penalty_coef,
        num_substeps=cli.num_substeps,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        ttl_seconds=cli.checkpoint_ttl_seconds,
        loss_fn=cli.loss_fn,
        max_steps=cli.max_steps,
        async_config=AsyncConfig(
            max_steps_off_policy=cli.max_steps_off_policy,
            groups_per_batch=cli.groups_per_batch,
        )
        if cli.max_steps_off_policy is not None
        else None,
        # A prompt whose whole group scores the same teaches nothing once the
        # advantages are centred; drop it instead of paying for the backward pass.
        remove_constant_reward_groups=cli.remove_constant_reward_groups,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    logger.info("Paintings and sketches will be saved under %s", artifact_dir)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
