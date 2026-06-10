# Changelog

## [1.0.0] - 2026-06-11

### Added
- Subagent completion execution board and runbook artifacts under `docs/superpowers/`.
- Completion handoff evidence bundle and runtime contract documentation.
- Kaggle recalibration watcher scripts under `scripts/`:
  - `watch_kaggle_kernel_status.py`
  - `watch_kaggle_kernel_and_recalibrate.py`
  - `watch_kaggle_run_outputs.sh`
- Decision-layer recalibration helper scripts under `kaggle/accuracy_phase1*`.

### Fixed
- Python 3.9 compatibility for backend typing annotations in:
  - `app/backend/decision.py`
  - `app/backend/inference.py`

### Verified
- `python3 -m pytest tests/backend/test_decision.py -q` (6 passed)
- `python3 -m pytest tests/backend/test_api_contract.py -q` (12 passed)
- `python3 -m pytest tests/backend/test_url_ingestion.py -q` (9 passed)
- `python3 -m pytest tests/backend -q` (27 passed)
- `cd app/frontend && npm run build`

### Links
- PR handoff marker: `foodlens-completion-delivery-note` (merged)
- Delivery marker merge SHA: `8f9b73ed166151585057c210c6601091d76afccf`
- Completion commits on main: `8174d6f`, `e227ef4`

