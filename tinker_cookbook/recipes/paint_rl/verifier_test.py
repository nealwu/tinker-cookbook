import random

import pytest

from tinker_cookbook.recipes.paint_rl.verifier import (
    effort_preset,
    extract_score,
    round_robin_pairs,
)


def test_round_robin_covers_every_pair_once():
    pairs = round_robin_pairs(5, random.Random(0))
    assert len(pairs) == 10
    assert {frozenset(p) for p in pairs} == {
        frozenset((a, b)) for a in range(5) for b in range(a + 1, 5)
    }


def test_effort_preset():
    assert effort_preset(0.2) == "low"
    assert effort_preset(0.99) == "xhigh"
    with pytest.raises(ValueError):
        effort_preset(0.3)


def test_extract_score_uses_last_tag_and_clamps():
    assert extract_score("<score> 4 </score> ... <score> 10 </score>") == 1.0
    assert extract_score("<score>12</score>") == 1.0  # clamped to 10
    assert extract_score("no verdict") is None


def test_transient_errors_are_retried():
    import httpx
    import openai

    from tinker_cookbook.recipes.paint_rl.verifier import _is_transient

    request = httpx.Request("POST", "https://example.invalid")

    def status(code: int) -> openai.APIStatusError:
        return openai.APIStatusError("x", response=httpx.Response(code, request=request), body=None)

    assert _is_transient(status(502)) and _is_transient(status(429))
    assert not _is_transient(status(400))
    assert _is_transient(openai.APIConnectionError(request=request))
    assert not _is_transient(RuntimeError("no answer logprobs"))


def test_judge_score_maps_1_to_10_onto_unit_interval():
    assert extract_score("<score> 1 </score>") == 0.0
    assert extract_score("<score> 5 </score>") == 4 / 9
    assert extract_score("<score>10</score>") == 1.0
    assert extract_score("<score> 0 </score>") == 0.0  # clamped to 1
    assert extract_score("seven") is None


def test_top_is_tied():
    from tinker_cookbook.recipes.paint_rl.verifier import top_is_tied

    assert top_is_tied([8 / 9, 8 / 9, 0.5])
    assert not top_is_tied([8 / 9, 7 / 9, 7 / 9])  # a tie below the top is not counted
    assert not top_is_tied([0.7])


def test_mean_per_painting_skips_unusable_verdicts():
    from tinker_cookbook.recipes.paint_rl.verifier import Verdict, _mean_per_painting

    def v(i: int, score: float) -> Verdict:
        return Verdict("c", 0, i, None, score, None, "", False)

    verdicts = [v(0, 1.0), v(0, 0.0), v(1, 0.2), v(2, 0.9)]
    assert _mean_per_painting(verdicts, 3, [True, False, True, False]) == [1.0, 0.2, 0.5]


def test_reward_config_follows_reward_mode():
    from tinker_cookbook.recipes.paint_rl.env import PaintRLDatasetBuilder, RewardWeights
    from tinker_cookbook.recipes.paint_rl.verifier import JudgeConfig, RewardMode, VerifierConfig

    def builder(mode: RewardMode) -> PaintRLDatasetBuilder:
        return PaintRLDatasetBuilder(
            model_name_for_tokenizer="m", renderer_name="r", batch_size=1, group_size=2,
            n_batches=1, policy_effort=0.7, reward_mode=mode, verifier_config=VerifierConfig(),
            judge_config=JudgeConfig(), evaluator_config=None, weights=RewardWeights(),
            canvas_size=512, render_concurrency=1, render_backend="local",
            render_timeout_s=1.0, n_train_prompts=1, n_test_prompts=0, test_group_size=1, seed=0,
        )  # fmt: skip

    assert isinstance(builder("verifier").reward_config, VerifierConfig)
    assert isinstance(builder("judge").reward_config, JudgeConfig)
    # The two arms are matched on everything but the scoring.
    v, j = VerifierConfig(), JudgeConfig()
    assert (v.model_name, v.effort, v.max_tokens, v.max_concurrency) == (
        j.model_name, j.effort, j.max_tokens, j.max_concurrency,
    )  # fmt: skip
    assert effort_preset(v.effort) == "low" and v.n_evaluations == 2


def test_absolute_prompt_asks_for_one_criterion_on_1_to_10():
    from tinker_cookbook.recipes.paint_rl.verifier import (
        OVERALL_CRITERION,
        EvaluatorConfig,
        VerifierConfig,
        absolute_prompt,
        resolve_criteria,
    )

    for criterion in resolve_criteria(()):
        prompt = absolute_prompt("Paint a teal mailbox.", criterion)
        assert criterion[0] in prompt and "<score> INTEGER_1_TO_10 </score>" in prompt
        others = [name for name, _ in resolve_criteria(()) if name != criterion[0]]
        assert not any(name in prompt for name in others)
    # The held-out evaluator asks exactly what the judge asks.
    assert resolve_criteria(EvaluatorConfig().criteria) == [OVERALL_CRITERION]
    assert absolute_prompt("p", OVERALL_CRITERION) == absolute_prompt("p")
    assert len(resolve_criteria(VerifierConfig().criteria)) == 3
