#!/usr/bin/env python3
"""
update_readme_metrics.py

Updates README.md with the latest codebase metrics:
- Lines of code (using cloc or tokei)
- Test count and test coverage (using pytest and pytest-cov)

Usage:
    python scripts/update_readme_metrics.py

Intended for use in CI or local development. Requires cloc or tokei, pytest, and pytest-cov.
"""
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"

# Badge regexes
LOC_BADGE_RE = re.compile(r"(\[!\[Lines of Code\]\().*?(\)\])")
COV_BADGE_RE = re.compile(r"(\[!\[Test Coverage\]\().*?(\)\])")
TESTS_BADGE_RE = re.compile(r"(\[!\[Test Count\]\().*?(\)\])")

# Badge templates
LOC_BADGE = "[![Lines of Code](https://img.shields.io/badge/lines%20of%20code-{loc}-blue?logo=github)](https://github.com/krljakob/RAGnificent)"
COV_BADGE = "[![Test Coverage](https://img.shields.io/badge/coverage-{cov}-brightgreen?logo=pytest)](https://github.com/krljakob/RAGnificent)"
TESTS_BADGE = "[![Test Count](https://img.shields.io/badge/tests-{tests}-yellow?logo=pytest)](https://github.com/krljakob/RAGnificent)"


def get_loc():
    """Get total lines of code using cloc or tokei."""
    try:
        # Try cloc first
        result = subprocess.run(
            ["cloc", "--json", "RAGnificent", "src", "tests"],
            capture_output=True,
            text=True,
            check=True,
        )
        import json

        stats = json.loads(result.stdout)
        total = sum(v["code"] for k, v in stats.items() if k not in ("header", "SUM"))
        return f"{total:,}"
    except Exception:
        # Fallback to tokei
        try:
            result = subprocess.run(
                ["tokei", "RAGnificent", "src", "tests", "--output", "json"],
                capture_output=True,
                text=True,
                check=True,
            )
            import json

            stats = json.loads(result.stdout)
            total = sum(
                v["code"] for v in stats.values() if isinstance(v, dict) and "code" in v
            )
            return f"{total:,}"
        except Exception:
            return "unknown"


def get_pytest_metrics():
    """Run pytest with coverage and parse test count and coverage percent."""
    try:
        result = subprocess.run(
            [
                "pytest",
                "--maxfail=1",
                "--disable-warnings",
                "--cov=RAGnificent",
                "--cov-report=term-missing",
                "-q",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        # Parse test count
        match = re.search(r"(\d+) passed", output)
        test_count = match[1] if match else "unknown"
        # Parse coverage
        cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+(\d+(?:\.\d+)?%)", output)
        coverage = cov_match[1] if cov_match else "unknown"
        return test_count, coverage
    except Exception:
        return "unknown", "unknown"


def update_readme(loc, test_count, coverage):
    """Update README.md badges in place."""
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace badges
    content, loc_n = LOC_BADGE_RE.subn(LOC_BADGE.format(loc=loc), content, count=1)
    content, cov_n = COV_BADGE_RE.subn(COV_BADGE.format(cov=coverage), content, count=1)
    content, tests_n = TESTS_BADGE_RE.subn(
        TESTS_BADGE.format(tests=test_count), content, count=1
    )

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)



def main():
    loc = get_loc()
    test_count, coverage = get_pytest_metrics()
    update_readme(loc, test_count, coverage)


if __name__ == "__main__":
    main()
