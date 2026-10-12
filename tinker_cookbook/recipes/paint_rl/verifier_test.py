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
    assert extract_score("<score> 4 </score> ... <score> 9 </score>") == 0.9
    assert extract_score("<score>12</score>") == 1.0
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
