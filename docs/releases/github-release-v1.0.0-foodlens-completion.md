# v1.0.0-foodlens-completion

FoodLens completion handoff release.

## Summary
- Finalized FoodLens completion docs, runbook, evidence bundle, and runtime contract.
- Fixed backend Python 3.9 typing compatibility.
- Added Kaggle recalibration helpers and watcher scripts.
- Added operational deployment, rollback, and health-check runbook.
- Published release changelog and delivery notes.

## Validation
- `python3 -m pytest tests/backend/test_decision.py -q` (6 passed)
- `python3 -m pytest tests/backend/test_api_contract.py -q` (12 passed)
- `python3 -m pytest tests/backend/test_url_ingestion.py -q` (9 passed)
- `python3 -m pytest tests/backend -q` (27 passed)
- `cd app/frontend && npm run build`

## Release Artifacts
- `CHANGELOG.md`
- `docs/releases/2026-06-11-foodlens-v1-release-notes.md`
- `docs/ops/foodlens-ops-deployment-runbook.md`
- `docs/superpowers/foodlens-completion-runbook.md`
- `docs/superpowers/foodlens-runtime-contract.md`
- `docs/superpowers/foodlens-subagent-evidence-bundle.md`

## Git References
- Tag: `v1.0.0-foodlens-completion`
- Main release commit: `047e929`
- Delivery marker merge: `8f9b73ed166151585057c210c6601091d76afccf`
- Completion commits: `8174d6f`, `e227ef4`

