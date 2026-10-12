"""Optional Modal backend for sketch rendering.

Local rendering (``render.py``) is the default and is faster on a machine
that has Chromium: a sketch renders in well under a second. This backend
moves the browser to Modal for machines that cannot run Chromium, or when
rendering must scale beyond one box.

Deploy once (needs ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``):

    modal deploy tinker_cookbook/recipes/paint_rl/render_modal.py

then train with ``render_backend=modal``. The page HTML (with p5 + p5.brush
inlined) is built locally by :class:`~tinker_cookbook.recipes.paint_rl.render.SketchRenderer`
and shipped to the function, which returns the PNG data URL and any error.
"""

from __future__ import annotations

import modal

APP_NAME = "tinker-paint-rl"
FUNCTION_NAME = "render_sketch"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("playwright==1.58.0")
    .run_commands("playwright install --with-deps chromium")
)

_CHROMIUM_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
]


@app.function(image=image, timeout=300, max_containers=64)
def render_sketch(page_html: str, canvas_size: int, timeout_s: float) -> dict[str, str | None]:
    """Render one page and return ``{"data_url": ..., "error": ...}``."""
    from playwright.sync_api import sync_playwright

    error: str | None = None
    data_url: str | None = None
    with sync_playwright() as p:
        browser = p.chromium.launch(args=_CHROMIUM_ARGS)
        context = browser.new_context(
            viewport={"width": canvas_size, "height": canvas_size}, offline=True
        )
        page = context.new_page()
        try:
            page.set_content(page_html, timeout=timeout_s * 1000)
            page.wait_for_function("window.__done === true", timeout=timeout_s * 1000)
            error = page.evaluate("window.__error")
            data_url = page.evaluate(
                "(() => { const c = document.querySelector('canvas'); "
                "return c ? c.toDataURL('image/png') : null; })()"
            )
        except Exception as e:
            error = f"render failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}"
        finally:
            browser.close()
    return {"data_url": data_url, "error": error}


def lookup_render_function() -> modal.Function:
    """The deployed ``render_sketch`` function."""
    return modal.Function.from_name(APP_NAME, FUNCTION_NAME)
