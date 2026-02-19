"""Take screenshots of the SoftMatch-DB web app for README.

Usage::

    uv run python docs/take_screenshots.py

Requires the web app to be running at http://localhost:8000.
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).resolve().parent / "screenshots"


def search_and_capture(
    page: Page,
    query: str,
    filename: str,
    expand_first: bool = False,
    label: str = "",
) -> None:
    """Execute a search query and capture the screenshot.

    Args:
        page: Playwright page object.
        query: Search query string.
        filename: Output filename (without directory).
        expand_first: Whether to click the first result row to expand KWIC.
        label: Label for console output.
    """
    page.fill("#query", query)
    page.click("#btn")
    page.wait_for_function(
        "document.querySelectorAll('tr.result-row').length > 0",
        timeout=15000,
    )
    time.sleep(0.5)

    if expand_first:
        first_row = page.locator("tr.result-row").first
        first_row.click()
        page.wait_for_function(
            "document.querySelector('.kwic-item') !== null",
            timeout=10000,
        )
        time.sleep(0.5)

    page.screenshot(path=str(OUT_DIR / filename))
    print(f"  {label}: {filename}")


def main() -> None:
    """Capture screenshots of key UI states."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})

        # --- 1. Initial state ---
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "01_initial.png"))
        print("  Initial state: 01_initial.png")

        # --- 2. Single-token: "京都" with KWIC ---
        search_and_capture(
            page, "京都", "02_search_kyoto.png",
            expand_first=True,
            label="京都 + KWIC",
        )

        # --- 3. 2-token: "世界遺産" → soft matches ---
        search_and_capture(
            page, "世界遺産", "03_search_world_heritage.png",
            expand_first=True,
            label="世界遺産 (2-tok) + KWIC",
        )

        # --- 4. 3-token: "自然言語処理" → soft matches ---
        search_and_capture(
            page, "自然言語処理", "04_search_nlp.png",
            expand_first=True,
            label="自然言語処理 (3-tok) + KWIC",
        )

        # --- 5. 3-token: "プロ野球選手" → semantic variants ---
        search_and_capture(
            page, "プロ野球選手", "05_search_baseball.png",
            expand_first=False,
            label="プロ野球選手 (3-tok)",
        )

        # --- 6. 2-token: "人工知能" with KWIC ---
        search_and_capture(
            page, "人工知能", "06_search_ai.png",
            expand_first=True,
            label="人工知能 (2-tok) + KWIC",
        )

        browser.close()

    print(f"\nAll screenshots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
