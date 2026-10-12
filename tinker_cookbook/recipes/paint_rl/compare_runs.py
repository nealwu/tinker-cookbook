"""Compare paint_rl runs (e.g. a learning-rate sweep) from their metrics.jsonl.

    python -m tinker_cookbook.recipes.paint_rl.compare_runs \
        runs='{"lr4e-5": "/tmp/paint_rl/lr4e-5", "lr1e-4": "/tmp/paint_rl/lr1e-4"}' \
        out=/tmp/paint_rl/compare.png

Prints a per-run summary table and writes a figure with the training curves
of the compile-gate pass rate, verifier score of compiled paintings, total
reward, and code length. Runs with different verifier settings optimise
different rewards; compare those on the held-out Gemini score instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import chz

KEYS: tuple[tuple[str, str], ...] = (
    ("env/all/reward/compile_ok", "compile gate pass rate"),
    ("env/all/reward/score_if_compiled", "verifier score (compiled paintings)"),
    ("env/all/reward/total", "total reward"),
    ("env/all/code/chars", "code length (chars)"),
)


@chz.chz
class Config:
    runs: dict[str, str]
    out: str = "compare.png"
    smooth: int = 3


def load_metrics(log_dir: Path) -> list[dict[str, float]]:
    path = log_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _series(rows: list[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        if key in row and "step" in row:
            xs.append(int(row["step"]))
            ys.append(float(row[key]))
    return xs, ys


def _smooth(ys: list[float], k: int) -> list[float]:
    if k <= 1:
        return ys
    out = []
    for i in range(len(ys)):
        window = ys[max(0, i - k + 1) : i + 1]
        out.append(sum(window) / len(window))
    return out


def main(config: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(KEYS), figsize=(4.5 * len(KEYS), 3.6))
    print(f"{'run':<12s} {'steps':>5s} " + " ".join(f"{label[:22]:>24s}" for _, label in KEYS))
    for name, log_dir in config.runs.items():
        rows = load_metrics(Path(log_dir))
        cells = []
        for ax, (key, label) in zip(axes, KEYS, strict=True):
            xs, ys = _series(rows, key)
            if xs:
                ax.plot(xs, _smooth(ys, config.smooth), label=name)
                first = sum(ys[: config.smooth]) / min(len(ys), config.smooth)
                last = sum(ys[-config.smooth :]) / min(len(ys), config.smooth)
                cells.append(f"{first:>10.3f} -> {last:<10.3f}")
            else:
                cells.append(f"{'n/a':>24s}")
            ax.set_title(label)
            ax.set_xlabel("step")
        print(f"{name:<12s} {len(rows):>5d} " + " ".join(f"{c:>24s}" for c in cells))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(config.out, dpi=120)
    print(f"Wrote {config.out}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
