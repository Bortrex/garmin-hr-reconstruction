# Repository conventions

- Write clear, simple, maintainable Python following PEP 8, with useful PEP 257 docstrings.
- Avoid Python type annotations, unnecessary abstractions, packages, and boilerplate. Reuse existing dependencies where practical.
- Keep the repository structure minimal: scripts in `scripts/`, source data in `data/`, derived data in `data/processed/`, and reports and figures in `docs/`. Document basic setup and usage in `README.md`.
- Preserve original Garmin files. Keep source provenance and timezone-aware timestamps explicit; distinguish recorded observations from estimates.
- For small, low-risk plotting or data-processing tasks, do not create dedicated test suites, CI workflows, or extra planning documents by default.
- Instead, run the actual script, confirm outputs exist and are non-empty, and perform basic data and numerical sanity checks. Inspect generated plots when relevant.
- Use focused automated checks only when scientific correctness, regressions, or consequential failures justify them.
- Never invent results or claim a check passed unless it was performed. Report limitations and checks that could not be run.
- Never push, publish, delete important data, or modify unrelated work without approval.
