# FoodLens Ops Runbook

## Deploy
1. Pull latest `main`.
2. Confirm environment versions:
   - Python runtime supports backend requirements.
   - Node runtime supports frontend build.
3. Build backend dependencies and frontend assets.
4. Run release sanity checks:
   - backend tests (all backend suite pass)
   - frontend build
5. Deploy backend service and frontend bundle with zero-downtime strategy.

## Rollback
1. Identify last known-good commit/tag.
2. Roll back backend container/image to previous version.
3. Roll back frontend static bundle to previous production build.
4. Re-run API health check and a single sample inference after rollback.

## Runtime health checks
- API reachability: `/health` endpoint responds.
- Inference latency: p95 stays within agreed SLO.
- Error rate: no sustained spike > baseline + threshold.
- Disk usage: logs/temporary artifacts monitored.
- Python traceback frequency: no new recurring backend exceptions.

## Incident checklist
1. Capture failed request payloads and trace IDs.
2. Verify model files are present and readable.
3. Confirm decision policy file is loaded correctly.
4. Confirm object detection role selection logic is functioning.
5. Run fallback path manually for:
   - URL mode
   - video mode
   - image mode

