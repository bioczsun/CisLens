# Contributing

Install `.[prepare,dev]` using the reference constraints. Read AGENTS.md and the
scientific invariants before changing model or data behavior. Run pytest, the
synthetic smoke workflow and `ruff check src tests`; build a wheel when changing
packaging. Add a regression test for a numerical or compatibility bug rather
than copying implementation details into tests.

Describe the concrete behavior changed, the validation performed, and whether
existing checkpoint predictions, preprocessing or published experiment settings
are affected. Never commit raw datasets, checkpoints, credentials or generated
run outputs. Contributions remain subject to the original authors selecting
and publishing the repository's license; see docs/publication.md.
