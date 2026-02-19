"""Take screenshots of the SoftMatch-DB web app for README.

Usage::

    uv run python docs/take_screenshots.py

Requires the web app to be running at http://localhost:8000.
"""

from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8000"
OUT_DIR = Path(__file__).resolve().parent / "screenshots"


def main() -> None:
    """Capture screenshots of key UI states."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})

        # --- 1. Initial state (empty, with suggestions) ---
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "01_initial.png"))
        print("1/4 Initial state captured")

        # --- 2. Search results for "京都" ---
        page.click("a.chip >> text=京都")
        page.wait_for_function(
            "document.querySelector('table') !== null",
            timeout=10000,
        )
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "02_search_kyoto.png"))
        print("2/4 Search results captured")

        # --- 3. KWIC expanded for first result ---
        first_row = page.locator("tr.result-row").first
        first_row.click()
        # Wait for KWIC content to load
        page.wait_for_function(
            "document.querySelector('.kwic-item') !== null",
            timeout=10000,
        )
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "03_kwic_expanded.png"))
        print("3/4 KWIC expanded captured")

        # --- 4. Search results for "人工知能" ---
        page.fill("#query", "人工知能")
        page.click("#btn")
        page.wait_for_function(
            "document.querySelectorAll('tr.result-row').length > 0",
            timeout=10000,
        )
        time.sleep(0.5)
        # Expand first KWIC
        first_row = page.locator("tr.result-row").first
        first_row.click()
        page.wait_for_function(
            "document.querySelector('.kwic-item') !== null",
            timeout=10000,
        )
        time.sleep(0.5)
        page.screenshot(path=str(OUT_DIR / "04_search_ai.png"))
        print("4/4 AI search + KWIC captured")

        browser.close()

    print(f"\nAll screenshots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
