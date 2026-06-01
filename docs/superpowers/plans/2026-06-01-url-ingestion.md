# URL Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pasted direct image URL and YouTube URL analysis to FoodLens while preserving the existing file upload, review, crop, and model metadata workflows.

**Architecture:** Put remote media ingestion in the FastAPI backend, not the browser. The backend validates public HTTP(S) URLs, downloads direct images with strict limits, uses `yt-dlp` plus `ffmpeg` for YouTube video frame extraction, then reuses the existing `predict_multi_food_image_bytes` pipeline. The React frontend adds a compact URL form to `UploadControls` and routes image/video URL submits through new analyzer methods and API client calls.

**Tech Stack:** FastAPI, Pydantic, Python standard library networking/subprocess/tempfile, Pillow-backed existing inference, `yt-dlp`, React/Vite/TypeScript, Vitest, pytest.

---

## File Map

- Create `app/backend/url_security.py`: parse and validate public `http`/`https` URLs, reject private/local hosts.
- Create `app/backend/media_download.py`: bounded direct image download with timeout, redirect validation, and max bytes.
- Create `app/backend/youtube_ingestion.py`: YouTube URL download through `yt-dlp`, frame extraction through `ffmpeg`, and frame response combination.
- Modify `app/backend/schemas.py`: add `UrlPredictionRequest`.
- Modify `app/backend/api.py`: add `/predict/multi-food/image-url` and `/predict/multi-food/youtube-url`.
- Modify `app/backend/requirements.txt`: add `yt-dlp`.
- Create `tests/backend/test_url_ingestion.py`: backend validation/download/YouTube helper and endpoint tests.
- Modify `app/frontend/src/api/foodlensClient.ts`: add URL endpoint client functions and user-input error class.
- Modify `app/frontend/src/state/useAnalyzer.ts`: add `analyzeImageUrl` and `analyzeYoutubeUrl`.
- Modify `app/frontend/src/components/UploadControls.tsx`: add URL input form.
- Modify `app/frontend/src/components/AnalyzerWorkbench.tsx`: pass URL handlers to upload controls.
- Modify `app/frontend/src/components/AnalyzerWorkbench.test.tsx`: add frontend URL behavior tests and update mocks.
- Modify `app/frontend/src/styles.css`: style the URL input form inside the existing controls.

---

## Task 1: Backend URL Safety And Direct Image Download

**Files:**
- Create: `app/backend/url_security.py`
- Create: `app/backend/media_download.py`
- Test: `tests/backend/test_url_ingestion.py`

- [ ] **Step 1: Write failing URL safety tests**

Add this file:

```python
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image
import pytest

from app.backend.media_download import DownloadError, download_image_url
from app.backend.url_security import UrlValidationError, validate_public_http_url


def make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(220, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_validate_public_http_url_rejects_private_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [(None, None, None, None, ("127.0.0.1", port or 443))],
    )

    with pytest.raises(UrlValidationError, match="public media URL"):
        validate_public_http_url("https://example.test/plate.jpg")


def test_validate_public_http_url_allows_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.backend.url_security.socket.getaddrinfo",
        lambda host, port, type=0: [(None, None, None, None, ("93.184.216.34", port or 443))],
    )

    assert validate_public_http_url("https://example.com/plate.jpg") == "https://example.com/plate.jpg"


def test_validate_public_http_url_rejects_non_http_scheme() -> None:
    with pytest.raises(UrlValidationError, match="http or https"):
        validate_public_http_url("file:///tmp/plate.jpg")


def test_download_image_url_rejects_non_image_content(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = {"Content-Type": "text/html", "Content-Length": "12"}
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b"<html></html>"

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr("app.backend.media_download.urlopen", lambda request, timeout: Response())

    with pytest.raises(DownloadError, match="image"):
        download_image_url("https://example.com/page")


def test_download_image_url_reads_bounded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_jpeg_bytes()

    class Response:
        headers = {"Content-Type": "image/jpeg", "Content-Length": str(len(payload))}
        status = 200

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            if self.offset >= len(payload):
                return b""
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr(
        "app.backend.media_download.validate_public_http_url",
        lambda url: url,
    )
    monkeypatch.setattr("app.backend.media_download.urlopen", lambda request, timeout: Response())

    assert download_image_url("https://example.com/plate.jpg") == payload
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m pytest tests/backend/test_url_ingestion.py -v
```

Expected: fails with `ModuleNotFoundError: No module named 'app.backend.media_download'` or missing functions.

- [ ] **Step 3: Implement URL validation**

Create `app/backend/url_security.py`:

```python
"""Safety checks for backend URL ingestion."""

from __future__ import annotations

from ipaddress import ip_address
import socket
from urllib.parse import urlparse


class UrlValidationError(ValueError):
    """Raised when a user-provided URL is not safe to fetch."""


def _is_public_address(address: str) -> bool:
    parsed = ip_address(address)
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def validate_public_http_url(url: str) -> str:
    """Return a validated URL or raise a user-readable validation error."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UrlValidationError("Enter a valid http or https media URL.")

    if not parsed.hostname:
        raise UrlValidationError("Enter a valid public media URL.")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UrlValidationError("Enter a public media URL, not a local address.")

    try:
        address_infos = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise UrlValidationError("The media URL host could not be resolved.") from exc

    addresses = {info[4][0] for info in address_infos}
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise UrlValidationError("Enter a public media URL, not a private or local address.")

    return url.strip()
```

- [ ] **Step 4: Implement bounded image download**

Create `app/backend/media_download.py`:

```python
"""Bounded media downloads for URL-based prediction."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .url_security import UrlValidationError, validate_public_http_url


MAX_IMAGE_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 12
MAX_REDIRECTS = 3


class DownloadError(ValueError):
    """Raised when remote media cannot be downloaded safely."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _read_bounded(response, max_bytes: int) -> bytes:  # type: ignore[no-untyped-def]
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise DownloadError("The image URL is too large.")
        chunks.append(chunk)
    return b"".join(chunks)


def download_image_url(url: str) -> bytes:
    """Download a public image URL with redirect, type, timeout, and size limits."""
    current_url = validate_public_http_url(url)
    opener = build_opener(_NoRedirectHandler)

    for _redirect_count in range(MAX_REDIRECTS + 1):
        request = Request(
            current_url,
            headers={"User-Agent": "FoodLens/0.1 URL ingestion"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                content_length = response.headers.get("Content-Length")
                if content_type and not content_type.startswith("image/"):
                    raise DownloadError("The URL did not return an image.")
                if content_length and int(content_length) > MAX_IMAGE_BYTES:
                    raise DownloadError("The image URL is too large.")
                return _read_bounded(response, MAX_IMAGE_BYTES)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                current_url = validate_public_http_url(urljoin(current_url, exc.headers["Location"]))
                continue
            raise DownloadError(f"The image URL returned HTTP {exc.code}.") from exc
        except UrlValidationError:
            raise
        except (OSError, URLError) as exc:
            raise DownloadError("The image URL could not be downloaded.") from exc

    raise DownloadError("The image URL redirected too many times.")
```

- [ ] **Step 5: Run backend URL helper tests**

Run:

```bash
python3 -m pytest tests/backend/test_url_ingestion.py -v
```

Expected: all tests in `test_url_ingestion.py` pass.

- [ ] **Step 6: Commit backend URL helper layer**

Run:

```bash
git add app/backend/url_security.py app/backend/media_download.py tests/backend/test_url_ingestion.py
git commit -m "feat: add safe image url download"
```

---

## Task 2: Backend URL Endpoints And YouTube Ingestion

**Files:**
- Modify: `app/backend/schemas.py`
- Modify: `app/backend/api.py`
- Modify: `app/backend/requirements.txt`
- Create: `app/backend/youtube_ingestion.py`
- Modify: `tests/backend/test_url_ingestion.py`

- [ ] **Step 1: Add failing endpoint and YouTube tests**

Append to `tests/backend/test_url_ingestion.py`:

```python
from fastapi.testclient import TestClient

import app.backend.api as api
import app.backend.inference as inference
from app.backend.api import app
from app.backend.schemas import MultiFoodPredictionResponse


client = TestClient(app)


def test_image_url_endpoint_runs_multi_food_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    image_bytes = make_jpeg_bytes()
    monkeypatch.setattr(api, "download_image_url", lambda url: image_bytes)
    monkeypatch.setattr(
        api,
        "predict_multi_food_image_bytes",
        lambda payload: inference.build_multi_food_mock(fallback_reason="missing_artifacts"),
    )

    response = client.post(
        "/predict/multi-food/image-url",
        json={"url": "https://example.com/plate.jpg"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["model"] == "resnet50_ft_v2"
    assert body["predictions"]


def test_image_url_endpoint_returns_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "download_image_url",
        lambda url: (_ for _ in ()).throw(api.UrlValidationError("Enter a public media URL.")),
    )

    response = client.post(
        "/predict/multi-food/image-url",
        json={"url": "http://127.0.0.1/plate.jpg"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Enter a public media URL."


def test_youtube_url_endpoint_returns_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "predict_multi_food_youtube_url",
        lambda url: (_ for _ in ()).throw(api.MediaDependencyError("YouTube support needs yt-dlp.")),
    )

    response = client.post(
        "/predict/multi-food/youtube-url",
        json={"url": "https://www.youtube.com/watch?v=abc123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "YouTube support needs yt-dlp."
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python3 -m pytest tests/backend/test_url_ingestion.py -v
```

Expected: fails because `UrlPredictionRequest`, endpoints, and YouTube helper do not exist.

- [ ] **Step 3: Add request schema and dependency**

Modify `app/backend/schemas.py`:

```python
class UrlPredictionRequest(BaseModel):
    """URL-based prediction request."""

    url: str = Field(min_length=1, max_length=2048)
```

Modify `app/backend/requirements.txt` by adding:

```text
yt-dlp
```

- [ ] **Step 4: Add YouTube ingestion helper**

Create `app/backend/youtube_ingestion.py`:

```python
"""YouTube URL ingestion for multi-food video analysis."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

from .inference import MODEL_NAME, predict_multi_food_image_bytes
from .schemas import MultiFoodPrediction, MultiFoodPredictionResponse
from .url_security import validate_public_http_url


MAX_YOUTUBE_DURATION_SECONDS = 10 * 60
FRAME_POSITIONS = (0.2, 0.5, 0.8)


class MediaDependencyError(RuntimeError):
    """Raised when an optional media ingestion dependency is missing."""


class MediaIngestionError(ValueError):
    """Raised when a user media URL cannot be processed."""


def _yt_dlp_module():
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaDependencyError("YouTube support needs yt-dlp installed on the backend.") from exc
    return yt_dlp


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise MediaDependencyError("YouTube support needs ffmpeg installed on the backend.")


def _sample_times(duration: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    return [min(max(duration * position, 0.0), max(duration - 0.05, 0.0)) for position in FRAME_POSITIONS]


def _copy_prediction_for_frame(
    prediction: MultiFoodPrediction,
    frame_index: int,
    global_index: int,
) -> MultiFoodPrediction:
    return prediction.copy(
        update={
            "source_id": f"youtube_frame_{frame_index + 1}",
            "detection_index": global_index,
        }
    )


def combine_frame_responses(
    responses: list[MultiFoodPredictionResponse],
) -> MultiFoodPredictionResponse:
    if not responses:
        raise MediaIngestionError("No usable video frames were extracted.")

    first = responses[0]
    predictions: list[MultiFoodPrediction] = []
    for frame_index, response in enumerate(responses):
        for prediction in response.predictions:
            predictions.append(
                _copy_prediction_for_frame(
                    prediction,
                    frame_index=frame_index,
                    global_index=len(predictions),
                )
            )

    fallback_reasons = [
        response.fallback_reason
        for response in responses
        if response.fallback_reason is not None
    ]

    return MultiFoodPredictionResponse(
        model=f"{first.model} · Video review",
        temperature=first.temperature,
        top_k=first.top_k,
        decision_thresholds=first.decision_thresholds,
        detector_status=f"{first.detector_status} · {len(responses)} frames",
        crop_count=len(predictions),
        predictions=predictions,
        artifact_status=first.artifact_status,
        fallback_reason=fallback_reasons[0] if fallback_reasons else None,
    )


def _download_youtube_video(url: str, output_dir: Path) -> tuple[Path, float]:
    yt_dlp = _yt_dlp_module()
    options = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except Exception as exc:
        raise MediaIngestionError("The YouTube URL could not be downloaded.") from exc

    duration = float(info.get("duration") or 0)
    if duration > MAX_YOUTUBE_DURATION_SECONDS:
        raise MediaIngestionError("The YouTube video is too long for analysis.")

    requested_downloads = info.get("requested_downloads") or []
    if requested_downloads and requested_downloads[0].get("filepath"):
        return Path(requested_downloads[0]["filepath"]), duration

    filepath = info.get("filepath")
    if filepath:
        return Path(filepath), duration

    candidates = sorted(output_dir.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise MediaIngestionError("The YouTube video download did not produce a file.")
    return candidates[0], duration


def _extract_frame(video_path: Path, output_path: Path, timestamp: float) -> bytes:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise MediaIngestionError("A frame could not be extracted from the YouTube video.")
    return output_path.read_bytes()


def predict_multi_food_youtube_url(url: str) -> MultiFoodPredictionResponse:
    safe_url = validate_public_http_url(url)
    _ensure_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="foodlens-youtube-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_path, duration = _download_youtube_video(safe_url, temp_dir)
        frame_responses: list[MultiFoodPredictionResponse] = []
        for index, timestamp in enumerate(_sample_times(duration)):
            frame_path = temp_dir / f"frame-{index + 1}.jpg"
            frame_bytes = _extract_frame(video_path, frame_path, timestamp)
            frame_responses.append(predict_multi_food_image_bytes(frame_bytes))
        return combine_frame_responses(frame_responses)
```

- [ ] **Step 5: Add API endpoints**

Modify `app/backend/api.py` imports and routes:

```python
from fastapi import FastAPI, File, HTTPException, UploadFile

from .media_download import DownloadError, download_image_url
from .schemas import MultiFoodPredictionResponse, PredictionResponse, UrlPredictionRequest
from .url_security import UrlValidationError
from .youtube_ingestion import (
    MediaDependencyError,
    MediaIngestionError,
    predict_multi_food_youtube_url,
)
```

Add routes after `/predict/multi-food/image`:

```python
@app.post("/predict/multi-food/image-url", response_model=MultiFoodPredictionResponse)
async def predict_multi_food_image_url(
    request: UrlPredictionRequest,
) -> MultiFoodPredictionResponse:
    """Predict multiple food regions from a public direct image URL."""
    try:
        image_bytes = download_image_url(request.url)
    except (DownloadError, UrlValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return predict_multi_food_image_bytes(image_bytes)


@app.post("/predict/multi-food/youtube-url", response_model=MultiFoodPredictionResponse)
async def predict_multi_food_youtube(
    request: UrlPredictionRequest,
) -> MultiFoodPredictionResponse:
    """Predict multiple food regions from a public YouTube URL."""
    try:
        return predict_multi_food_youtube_url(request.url)
    except UrlValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MediaDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MediaIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

- [ ] **Step 6: Run backend URL endpoint tests**

Run:

```bash
python3 -m pytest tests/backend/test_url_ingestion.py -v
```

Expected: all URL ingestion tests pass.

- [ ] **Step 7: Run full backend contract tests**

Run:

```bash
python3 -m pytest tests/backend -v
```

Expected: all backend tests pass.

- [ ] **Step 8: Commit backend URL endpoints**

Run:

```bash
git add app/backend/api.py app/backend/schemas.py app/backend/requirements.txt app/backend/youtube_ingestion.py tests/backend/test_url_ingestion.py
git commit -m "feat: add backend url ingestion endpoints"
```

---

## Task 3: Frontend API Client And Analyzer State

**Files:**
- Modify: `app/frontend/src/api/foodlensClient.ts`
- Modify: `app/frontend/src/state/useAnalyzer.ts`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`

- [ ] **Step 1: Write failing frontend state tests**

Modify the mock block in `app/frontend/src/components/AnalyzerWorkbench.test.tsx`:

```typescript
import {
  fetchRuntimeStatus,
  predictMultiFoodImage,
  predictMultiFoodImageUrl,
  predictMultiFoodYoutubeUrl,
} from "../api/foodlensClient";

vi.mock("../api/foodlensClient", () => ({
  combineFrameResults: (results: AnalyzerResult[]) => results[0] ?? createResult("fallback", 0.1),
  fetchRuntimeStatus: vi.fn(async () => ({
    ready: true,
    title: "System ready",
    classifierLabel: "Classifier ready",
    detectorLabel: "Detector ready",
    modeLabel: "Live detector + classifier",
  })),
  predictMultiFoodImage: vi.fn(),
  predictMultiFoodImageUrl: vi.fn(),
  predictMultiFoodYoutubeUrl: vi.fn(),
  toLocalDemoResult: () => createResult("ravioli", 0.972, "local_demo"),
}));
```

Add tests inside `describe("AnalyzerWorkbench", ...)`:

```typescript
it("analyzes a direct image URL", async () => {
  const user = userEvent.setup();
  vi.mocked(predictMultiFoodImageUrl).mockResolvedValue(createResult("pizza", 0.94));
  render(<AnalyzerWorkbench />);

  await user.type(
    screen.getByLabelText("Image URL"),
    "https://images.example.com/pizza.jpg",
  );
  await user.click(screen.getByRole("button", { name: "Analyze URL" }));

  expect(predictMultiFoodImageUrl).toHaveBeenCalledWith(
    "https://images.example.com/pizza.jpg",
  );
  expect(await screen.findByText("Image URL analysis complete")).toBeInTheDocument();
  expect(screen.getByText("pizza")).toBeInTheDocument();
});

it("analyzes a YouTube URL in video mode", async () => {
  const user = userEvent.setup();
  vi.mocked(predictMultiFoodYoutubeUrl).mockResolvedValue(createResult("hamburger", 0.91));
  render(<AnalyzerWorkbench />);

  await user.click(screen.getByRole("button", { name: "Video" }));
  await user.type(
    screen.getByLabelText("YouTube URL"),
    "https://www.youtube.com/watch?v=abc123",
  );
  await user.click(screen.getByRole("button", { name: "Analyze URL" }));

  expect(predictMultiFoodYoutubeUrl).toHaveBeenCalledWith(
    "https://www.youtube.com/watch?v=abc123",
  );
  expect(await screen.findByText("Video URL review complete")).toBeInTheDocument();
  expect(screen.getByText("hamburger")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: fails because URL API functions and analyzer methods do not exist.

- [ ] **Step 3: Add API client URL calls and input error type**

Modify `app/frontend/src/api/foodlensClient.ts`:

```typescript
export class FoodLensApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "FoodLensApiError";
  }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as unknown;
    if (isRecord(body) && isString(body.detail)) {
      return body.detail;
    }
  } catch {
    return `FoodLens API returned ${response.status}`;
  }

  return `FoodLens API returned ${response.status}`;
}

async function postUrlPrediction(
  endpoint: string,
  url: string,
): Promise<AnalyzerResult> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) {
    throw new FoodLensApiError(await parseErrorMessage(response), response.status);
  }

  const body = (await response.json()) as unknown;
  if (!isBackendMultiFoodResponse(body)) {
    throw new Error("FoodLens API returned an invalid multi-food response.");
  }

  return normalizeMultiFoodResponse(body);
}

export function isUserInputApiError(error: unknown): error is FoodLensApiError {
  return error instanceof FoodLensApiError && error.status >= 400 && error.status < 500;
}

export function predictMultiFoodImageUrl(url: string): Promise<AnalyzerResult> {
  return postUrlPrediction("/predict/multi-food/image-url", url);
}

export function predictMultiFoodYoutubeUrl(url: string): Promise<AnalyzerResult> {
  return postUrlPrediction("/predict/multi-food/youtube-url", url);
}
```

- [ ] **Step 4: Add analyzer URL methods**

Modify imports in `app/frontend/src/state/useAnalyzer.ts`:

```typescript
import {
  combineFrameResults,
  isUserInputApiError,
  predictMultiFoodImage,
  predictMultiFoodImageUrl,
  predictMultiFoodYoutubeUrl,
  toLocalDemoResult,
} from "../api/foodlensClient";
```

Extend `AnalyzerState`:

```typescript
analyzeImageUrl: (url: string) => Promise<void>;
analyzeYoutubeUrl: (url: string) => Promise<void>;
```

Add these callbacks before `loadSample`:

```typescript
const analyzeImageUrl = useCallback(
  async (url: string) => {
    requestSequenceRef.current += 1;
    const requestSequence = requestSequenceRef.current;
    replacePreview(url);
    setStatus("loading");
    setResult(null);
    setMessage("Analyzing image URL");

    try {
      const nextResult = await predictMultiFoodImageUrl(url);
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      setResult(nextResult);
      setStatus("ready");
      setMessage("Image URL analysis complete");
    } catch (error) {
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      if (isUserInputApiError(error)) {
        setStatus("error");
        setResult(null);
        setMessage(error.message);
        return;
      }
      setResult(toLocalDemoResult());
      setStatus("ready");
      setMessage(
        error instanceof Error
          ? `Using local demo fallback: ${error.message}`
          : "Using local demo fallback",
      );
    }
  },
  [replacePreview],
);

const analyzeYoutubeUrl = useCallback(
  async (url: string) => {
    requestSequenceRef.current += 1;
    const requestSequence = requestSequenceRef.current;
    replacePreview(null);
    setStatus("loading");
    setResult(null);
    setMessage("Downloading YouTube video");

    try {
      const nextResult = await predictMultiFoodYoutubeUrl(url);
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      setResult(nextResult);
      setStatus("ready");
      setMessage("Video URL review complete");
    } catch (error) {
      if (requestSequence !== requestSequenceRef.current) {
        return;
      }
      if (isUserInputApiError(error)) {
        setStatus("error");
        setResult(null);
        setMessage(error.message);
        return;
      }
      setResult(toLocalDemoResult());
      setStatus("ready");
      setMessage(
        error instanceof Error
          ? `Using local demo fallback: ${error.message}`
          : "Using local demo fallback",
      );
    }
  },
  [replacePreview],
);
```

Return the methods:

```typescript
analyzeImageUrl,
analyzeYoutubeUrl,
```

- [ ] **Step 5: Run frontend state tests**

Run:

```bash
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: URL tests still fail until controls are implemented; existing tests pass.

- [ ] **Step 6: Commit API client and analyzer state**

Run:

```bash
git add app/frontend/src/api/foodlensClient.ts app/frontend/src/state/useAnalyzer.ts app/frontend/src/components/AnalyzerWorkbench.test.tsx
git commit -m "feat: add frontend url analysis state"
```

---

## Task 4: URL Controls UI

**Files:**
- Modify: `app/frontend/src/components/UploadControls.tsx`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.tsx`
- Modify: `app/frontend/src/styles.css`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`

- [ ] **Step 1: Add failing UploadControls tests**

Add to `describe("UploadControls", ...)`:

```typescript
it("submits image URLs from image mode", async () => {
  const user = userEvent.setup();
  const onUrlSubmit = vi.fn();

  render(
    <UploadControls
      mode="image"
      status="idle"
      onModeChange={vi.fn()}
      onUploadImage={vi.fn()}
      onVideoSelected={vi.fn()}
      onUrlSubmit={onUrlSubmit}
      onSample={vi.fn()}
      onClear={vi.fn()}
    />,
  );

  await user.type(screen.getByLabelText("Image URL"), "https://example.com/plate.jpg");
  await user.click(screen.getByRole("button", { name: "Analyze URL" }));

  expect(onUrlSubmit).toHaveBeenCalledWith("https://example.com/plate.jpg");
});

it("submits YouTube URLs from video mode", async () => {
  const user = userEvent.setup();
  const onUrlSubmit = vi.fn();

  render(
    <UploadControls
      mode="video"
      status="idle"
      onModeChange={vi.fn()}
      onUploadImage={vi.fn()}
      onVideoSelected={vi.fn()}
      onUrlSubmit={onUrlSubmit}
      onSample={vi.fn()}
      onClear={vi.fn()}
    />,
  );

  expect(screen.getByPlaceholderText("Paste a YouTube URL")).toBeInTheDocument();

  await user.type(screen.getByLabelText("YouTube URL"), "https://www.youtube.com/watch?v=abc123");
  await user.click(screen.getByRole("button", { name: "Analyze URL" }));

  expect(onUrlSubmit).toHaveBeenCalledWith("https://www.youtube.com/watch?v=abc123");
});
```

Update existing `UploadControls` renders to pass `onUrlSubmit={vi.fn()}`.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: fails because `onUrlSubmit` prop and URL form do not exist.

- [ ] **Step 3: Implement URL form controls**

Modify `app/frontend/src/components/UploadControls.tsx`:

```typescript
import { Image, Link, RotateCcw, Sparkles, Upload, Video } from "lucide-react";
import type { ChangeEvent, FormEvent } from "react";
import { useState } from "react";
```

Add prop:

```typescript
onUrlSubmit: (url: string) => void;
```

Inside component:

```typescript
const [urlValue, setUrlValue] = useState("");
const urlLabel = mode === "video" ? "YouTube URL" : "Image URL";
const urlPlaceholder = mode === "video" ? "Paste a YouTube URL" : "Paste a direct image URL";

function handleUrlSubmit(event: FormEvent<HTMLFormElement>) {
  event.preventDefault();
  const trimmedUrl = urlValue.trim();
  if (!trimmedUrl) {
    return;
  }
  onUrlSubmit(trimmedUrl);
  setUrlValue("");
}
```

Add form between the segmented control and control row:

```tsx
<form className="url-control" aria-label="URL analysis" onSubmit={handleUrlSubmit}>
  <label htmlFor="foodlens-url-input">{urlLabel}</label>
  <div className="url-control__field">
    <Link size={16} aria-hidden="true" />
    <input
      id="foodlens-url-input"
      type="url"
      value={urlValue}
      placeholder={urlPlaceholder}
      disabled={disabled}
      onChange={(event) => setUrlValue(event.target.value)}
    />
    <button type="submit" disabled={disabled || !urlValue.trim()}>
      Analyze URL
    </button>
  </div>
</form>
```

- [ ] **Step 4: Wire URL submit in AnalyzerWorkbench**

Modify the `UploadControls` render in `AnalyzerWorkbench.tsx`:

```tsx
onUrlSubmit={(url) => {
  if (analyzer.mode === "video") {
    void analyzer.analyzeYoutubeUrl(url);
  } else {
    void analyzer.analyzeImageUrl(url);
  }
}}
```

- [ ] **Step 5: Add CSS**

Add to `app/frontend/src/styles.css` near upload controls:

```css
.url-control {
  display: grid;
  flex: 1 1 340px;
  gap: 6px;
  min-width: min(100%, 280px);
}

.url-control label {
  color: var(--lab-muted);
  font-family: var(--font-display);
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.url-control__field {
  display: grid;
  grid-template-columns: auto minmax(160px, 1fr) auto;
  gap: 8px;
  align-items: center;
  padding: 4px;
  border: 1px solid var(--lab-border);
  border-radius: var(--lab-radius);
  background: var(--lab-surface-soft);
}

.url-control__field svg {
  margin-left: 8px;
  color: var(--lab-muted);
}

.url-control__field input {
  width: 100%;
  min-height: 34px;
  border: 0;
  color: var(--lab-ink);
  background: transparent;
  outline: none;
}

.url-control__field button {
  min-height: 34px;
  white-space: nowrap;
}
```

Add responsive rule in the `max-width: 640px` block:

```css
.url-control,
.url-control__field {
  width: 100%;
}

.url-control__field {
  grid-template-columns: auto minmax(0, 1fr);
}

.url-control__field button {
  grid-column: 1 / -1;
}
```

- [ ] **Step 6: Run frontend tests**

Run:

```bash
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: all analyzer and upload control tests pass.

- [ ] **Step 7: Commit URL controls UI**

Run:

```bash
git add app/frontend/src/components/UploadControls.tsx app/frontend/src/components/AnalyzerWorkbench.tsx app/frontend/src/styles.css app/frontend/src/components/AnalyzerWorkbench.test.tsx
git commit -m "feat: add url analyzer controls"
```

---

## Task 5: Verification, Dependency Install, And Browser Audit

**Files:**
- No source changes expected unless verification finds a bug.

- [ ] **Step 1: Install backend dependency if missing**

Run:

```bash
python3 -m pip show yt-dlp
```

If missing, run:

```bash
python3 -m pip install -r app/backend/requirements.txt
```

Expected: `yt-dlp` is installed or pip installs it successfully. If network is unavailable, record that YouTube runtime verification is blocked but tests can still pass with mocks.

- [ ] **Step 2: Run backend tests**

Run:

```bash
python3 -m pytest tests/backend -v
```

Expected: all backend tests pass.

- [ ] **Step 3: Run frontend tests, typecheck, and build**

Run:

```bash
npm test
npm run typecheck
npm run build
```

Expected: all tests pass, TypeScript emits no errors, and Vite builds successfully.

- [ ] **Step 4: Run diff hygiene check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Manual runtime check**

Start the backend and frontend:

```bash
python3 -m uvicorn app.backend.api:app --host 127.0.0.1 --port 8000
npm run dev
```

Check:

- Image URL field appears in image mode.
- YouTube URL field appears in video mode.
- Invalid `http://127.0.0.1/test.jpg` returns a visible “public media URL” error.
- Existing Upload, Sample, Clear, Review, and Models buttons still work.
- If `yt-dlp` and `ffmpeg` are installed, a short public YouTube URL reaches `Video URL review complete`.

- [ ] **Step 6: Commit any verification fixes**

If verification required fixes:

```bash
git add app/backend app/frontend/src tests/backend
git commit -m "fix: stabilize url ingestion workflow"
```

If no fixes were required, do not create an empty commit.
