# FoodLens Backend

This backend provides the FoodLens API contract while the real PyTorch inference
service is being prepared.

## Run Locally

Install runtime dependencies in your preferred environment:

```bash
pip install fastapi uvicorn python-multipart
```

Start the API:

```bash
uvicorn app.backend.api:app --reload --port 8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Endpoints

```text
POST /predict/image
POST /predict/video
```

Both endpoints currently return deterministic mock predictions that match the
final response shape expected by the frontend.

## Real Inference Integration

To move from mock inference to real inference, place artifacts outside git under
`app/artifacts/` and update `app/backend/inference.py`.

Required artifacts:

- `resnet50_ft_v2_best.pth`
- ordered class names
- calibration temperature
- decision policy
- hard-class list
- confusion-pair list
