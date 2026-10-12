# Paint with code: RL with a multimodal LLM-as-a-Verifier

Teach `thinkingmachines/Inkling-Small` to paint watercolors by writing
[p5.brush](https://github.com/acamposuribe/p5.brush) sketches. The policy writes
a JavaScript sketch, the sketch is rendered headlessly to a PNG, a multimodal
model scores the *image*, and group-centred rewards update the policy. The
task follows ["Training AI to Paint with Code"](https://surya.website/rling-qwen-to-paint-with-code).

The reward is where this recipe differs. That project scored each image on an
absolute 0-10 scale and found the scores compressed and training stalled.
Here, rewards come from
[LLM-as-a-Verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier):
pairwise comparisons on a fine-grained 20-letter scale, one narrow criterion
per call, repeated with the two image slots swapped, and aggregated over a
round-robin tournament within each rollout group. Each comparison is one
`llm_verifier.compare` call from the
[`llm-verifier`](https://pypi.org/project/llm-verifier/) package. Inkling-Small
is its own verifier: the same base model, sampled at low thinking effort, looks
at the two rendered paintings through its native image input.

| Role | Model | Notes |
|---|---|---|
| Policy | `thinkingmachines/Inkling-Small` | LoRA rank 32, `tml_v0` renderer, thinking effort 0.7, 8k token budget |
| Training reward | `thinkingmachines/Inkling-Small` | `llm_verifier.compare` over Tinker's OpenAI-compatible endpoint, thinking effort 0.2, token logprobs, pairwise round-robin tournament |
| Held-out evaluator | `gemini-3.8-flash` | Separate from the reward and never trained against |

## Running

```bash
pip install -e ".[paint-rl]"     # llm-verifier, openai, google-genai, playwright
python -m playwright install --with-deps chromium

export TINKER_API_KEY=...
export GEMINI_API_KEY=...   # held-out evaluator; or pass eval_model=None

python -m tinker_cookbook.recipes.paint_rl.train log_path=/tmp/paint_rl/verifier

# In a second shell: held-out evaluation of every checkpoint as it is saved
python -m tinker_cookbook.recipes.paint_rl.eval_checkpoints log_path=/tmp/paint_rl/verifier

# Look at the results
python -m tinker_cookbook.recipes.paint_rl.gallery artifact_dir=/tmp/paint_rl/verifier/paintings
python -m tinker_cookbook.recipes.paint_rl.progression \
    artifact_dir=/tmp/paint_rl/verifier/paintings out_dir=/tmp/paint_rl/verifier/progression
python -m tinker_cookbook.recipes.paint_rl.compare_runs \
    runs='{"lr4e-5": "/tmp/paint_rl/verifier", "lr1e-4": "/tmp/paint_rl/lr1e-4"}' out=compare.png
```

Defaults: 8 groups of 5 rollouts per step, 300 steps, LoRA rank 32, learning
rate 4e-5. A checkpoint is saved every 5 steps and kept on Tinker indefinitely
(`checkpoint_ttl_seconds=None`; the cookbook default would expire periodic
checkpoints after 7 days); they are listed in `<log_path>/checkpoints.jsonl`.
`hyperparam_utils.get_lr` has no calibrated value for Inkling; 4e-5 worked
here (see the top-level README for the run), but sweep the learning rate and
the two effort settings (`policy_effort`, `verifier_effort`) if you change
the model. Every knob is in `train.py`.

## The loop

1. **Prompt.** For example, `Paint a teal heron standing in shallow water in
   wet-on-wet watercolor wash.` The pool is 48 subjects x 8 colours x 4 styles
   (`prompts.py`), drawn from what watercolorists commonly paint: twelve
   flowers, twelve animals, twelve still-life objects and twelve scenes. Each
   family spans three difficulty tiers: a single subject, a subject in a
   setting, and two elements in a spatial relation. Colours and styles stay
   visually distinct at 448 px. The 256 training prompts are balanced over
   subjects, colours and styles. The 16 held-out prompts come in two halves:
   **novel subjects** (one per family, never trained on in any form) and
   **novel combinations** of subjects, colours and styles seen in training.
   `prompts_test.py` pins the split. The system prompt allows a fixed list of
   about 20 brush calls and gives no API documentation; the original project
   found that a long API reference made the model call functions that don't
   exist.
2. **Sketch.** The policy replies with one `javascript` block that defines
   `setup()` on a 512x512 WEBGL canvas.
3. **Render.** `render.py` runs the sketch in a sandboxed headless Chromium
   (Playwright with SwiftShader WebGL), with p5 1.11.3 and p5.brush 1.1.4
   inlined, and reads the canvas back as a PNG. A sketch passes the compile
   gate if it throws no error, makes at least three distinct `brush.*`
   drawing calls, and paints a non-blank canvas.
4. **Verify.** Every unordered pair of the group's compiled paintings is
   compared once per criterion and repeat. A group of 5 gives 10 pairs x 3
   criteria x 2 repeats = 60 comparisons. A painting's score is its mean grade
   across every comparison it appears in. Sketches that fail the compile gate
   score 0 and are left out of the tournament.
5. **Update.** `reward = 0.05 * compiled + 0.05 * length_ok + 0.90 * score`.
   The reward is centred within each group and fed to an importance-sampling
   policy gradient (`tinker_cookbook/rl/train.py`).

Every rollout's JavaScript, PNG and raw response, plus every verifier score,
are saved under
`<log_path>/paintings/<split>/step_XXXX/<prompt id>/` and indexed in
`paintings/index.jsonl`.

## The reward is a distribution, not a letter

`llm_verifier.compare` reads the verifier's top 20 alternatives at the token
after each `<score_A>` and `<score_B>` tag, and the score is the
*expectation* of the grade under that distribution, not the letter that
happened to be sampled. A painting the verifier is torn on between "E" and
"G" scores between them, and a 5% chance of "A" still raises the reward.
Fine-grained scores like these are the reason the scale has 20 levels.

The verifier is called through Tinker's
[OpenAI-compatible endpoint](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/),
which returns token logprobs but has no assistant prefill. `verifier.py`
therefore sets the client up for `llm_verifier`'s hosted-API path, which
reads the distribution from the score tags the model samples itself, rather
than its prefill path, which would score every comparison as a tie on this
endpoint.

With the default two repeats per criterion, the second repeat swaps the two
image slots, so the verifier's bias toward one position cancels out within
each pair. The three criteria are scored separately: prompt adherence,
watercolor technique, and composition.

## Held-out evaluation

Held-out evaluation runs in its own process, so it never blocks training and
training stays on-policy. A checkpoint is saved every 5 steps (kept on Tinker
indefinitely); `eval_checkpoints.py` follows `checkpoints.jsonl` and
evaluates the base model and each checkpoint on the 16 held-out prompts:
the policy samples 5 paintings per prompt, and Gemini scores them
(`eval_strong/score`). The training verifier is skipped there unless
`with_verifier=True`. Results go to `<log_path>/heldout_eval/`; already
evaluated steps are skipped, so the evaluator can be restarted. `eval_every`
in `train.py` (default 0) still runs the same evaluation inline if you prefer.

Gemini's scores never enter the reward. The verifier's reward is relative to
the other paintings in a group, so it can't be compared across steps or runs;
the Gemini score is the fixed yardstick. Gemini returns no logprobs, so it
scores from the number it writes in its reply: an integer 0-10 per criterion,
averaged over the three criteria and divided by 10. Calls that fail with 429/5xx are retried with backoff; a call that still
fails is left out (`eval_strong/failed_frac`).

Useful metrics in `metrics.jsonl`:

- `env/all/reward/compile_ok`: compile gate pass rate.
- `env/all/reward/score_if_compiled`: verifier score of compiled paintings.
- `env/all/verifier/failed_frac`: verifier calls that failed or ran out of tokens and were scored as ties; should stay near 0.
- `test/env/all/eval_strong/score` (in `heldout_eval/metrics.jsonl`): Gemini's held-out score, the headline number.
- `test/env/tier3/...`, `test/env/novel_subject/...`: held-out metrics broken out by tier and by held-out kind.

## Results

With the defaults, the held-out Gemini score rose from 3.24 to 5.35 out of 10
by step 470 (4.25 to 5.35 on compiled paintings only), and the held-out
compile rate from 76% to 100%. Curves, example paintings and the run's
metrics are in the
[top-level README](https://github.com/jackyk02/paint_tinker#results).

## Rendering on Modal

`render_backend=modal` sends each page to a Modal function instead of a local
browser (`render_modal.py`), for machines without Chromium. Deploy it once with
`modal deploy tinker_cookbook/recipes/paint_rl/render_modal.py`. On a machine
with many cores, local rendering is faster.

## Files

| File | Purpose |
|---|---|
| `prompts.py` | System prompt (brush allowlist), prompt pool, and the pinned held-out split |
| `assets.py` | Pinned p5 / p5.brush sources, cached under `~/.cache/tinker-cookbook/paint_rl` |
| `render.py` | Headless WebGL renderer and the compile / brush-use / blank gates |
| `render_modal.py` | Optional Modal backend for the renderer |
| `verifier.py` | `PairwiseVerifier` (round-robin tournament over `llm_verifier.compare`) and the held-out `GeminiEvaluator` |
| `env.py` | `PaintEnv`, group reward, held-out evaluation, artifact saving, dataset builder |
| `train.py` | CLI launcher |
| `eval_checkpoints.py` | Held-out evaluation of saved checkpoints, in a separate process |
| `gallery.py`, `progression.py`, `compare_runs.py` | HTML gallery, per-step best-painting sheets, run comparison curves |
