# URL Ingestion Design

## Goal

FoodLens should accept pasted URLs as an alternative to local file upload:

- Image mode accepts a direct image URL.
- Video mode accepts a YouTube URL.
- Results use the same analysis, review, crop, and model metadata UI that file uploads already use.

## Scope

This design implements backend URL ingestion. The browser will not fetch remote media directly because image hosts and YouTube commonly block browser-side reads with CORS, and YouTube stream extraction is not practical in the frontend.

In scope:

- URL input in the existing analyzer controls.
- Backend image URL endpoint.
- Backend YouTube URL endpoint using `yt-dlp`.
- Video frame sampling on the backend.
- Shared result contract with existing multi-food image analysis.
- Clear loading and error/fallback messages.
- Tests for backend validation and frontend controls.

Out of scope:

- Arbitrary webpage scraping for embedded images.
- Login-gated/private YouTube videos.
- YouTube playlists and channels.
- Long-running background jobs.
- Persisting downloaded media.

## User Experience

The existing Upload/Sample/Clear controls gain a compact URL form. The form label and placeholder change with mode:

- Image mode: direct image URL, for example `https://example.com/plate.jpg`.
- Video mode: YouTube URL, for example `https://www.youtube.com/watch?v=...`.

Submitting the URL starts analysis immediately. The status notice should show mode-specific progress:

- `Analyzing image URL`
- `Downloading YouTube video`
- `Sampling video frames`
- `Video URL review complete`

If the URL cannot be fetched, is unsupported, is too large, or cannot be decoded as image/video, the app should keep the existing fallback behavior where appropriate and show a readable message. Raw backend tokens should not leak into the UI.

## Backend Architecture

Add URL-specific request models and endpoints:

- `POST /predict/multi-food/image-url`
  - Body: `{ "url": "https://..." }`
  - Downloads the image bytes with validation.
  - Calls `predict_multi_food_image_bytes`.
  - Returns `MultiFoodPredictionResponse`.

- `POST /predict/multi-food/youtube-url`
  - Body: `{ "url": "https://www.youtube.com/watch?v=..." }`
  - Downloads a temporary video with `yt-dlp`.
  - Samples a small fixed number of frames.
  - Calls `predict_multi_food_image_bytes` for each sampled frame.
  - Combines frame-level responses into one `MultiFoodPredictionResponse`.

The backend should keep helper boundaries small:

- `url_security.py`: URL parsing and network safety checks.
- `media_download.py`: bounded HTTP download for direct image URLs.
- `youtube_ingestion.py`: `yt-dlp` download and frame extraction.
- API routes stay thin and delegate to helpers.

## URL Safety

Backend URL ingestion must avoid server-side request forgery and runaway downloads:

- Allow only `http` and `https`.
- Reject localhost, loopback, link-local, private, and multicast IP targets.
- Resolve hostnames before fetching and reject unsafe resolved addresses.
- Set request timeout and maximum bytes for direct image download.
- Validate content type when available, while still allowing image decoding to be the final authority.
- Use temporary files for YouTube downloads and remove them after analysis.
- Limit YouTube download format and duration where feasible.

## Video Sampling

The backend should mirror the frontend video behavior:

- Sample three frames around 20%, 50%, and 80% of duration.
- Analyze each frame through the existing multi-food image pipeline.
- Combine frame predictions into a single response, preserving region labels and using a video-specific model suffix.

If frame extraction fails, return a controlled error or fallback response with a user-readable reason.

## Frontend Architecture

Extend the existing analyzer state rather than creating a separate URL workflow:

- Add `analyzeImageUrl(url: string)`.
- Add `analyzeYoutubeUrl(url: string)`.
- Keep file upload methods unchanged.
- Add URL form props to `UploadControls`.
- Disable URL submit while loading.
- Clear the URL field after a successful submit or clear action.

The preview stage can show the pasted image URL for direct images. For YouTube videos, preview can initially remain empty or show a neutral video URL state; analysis results and crop cards remain the primary feedback.

## API Client

Add frontend client methods:

- `predictMultiFoodImageUrl(url: string): Promise<AnalyzerResult>`
- `predictMultiFoodYoutubeUrl(url: string): Promise<AnalyzerResult>`

Both normalize the backend response with the same `normalizeMultiFoodResponse` function used by file uploads.

## Errors

Frontend messages should distinguish likely user actions:

- Invalid URL: ask for a valid `http` or `https` URL.
- Unsupported host or private URL: say the app only accepts public media URLs.
- YouTube download unavailable: say YouTube support needs backend downloader support.
- Decode failure: ask for another image/video URL.
- Backend failure: use existing local demo fallback only when the response is unavailable or invalid, not when validation correctly rejects input.

## Testing

Backend tests:

- Reject invalid schemes and private/local URLs.
- Image URL endpoint calls the image pipeline for a mocked public image download.
- Image URL endpoint rejects oversized or non-image content.
- YouTube endpoint handles downloader failure with a controlled error.
- YouTube helper samples frames and combines mocked frame results.

Frontend tests:

- URL input appears in image and video mode with correct placeholder.
- Submitting an image URL calls `analyzeImageUrl`.
- Submitting a YouTube URL calls `analyzeYoutubeUrl`.
- Loading state disables URL submit.
- Existing Upload, Sample, Clear, Review, and Models flows still work.

Manual verification:

- Direct image URL analysis.
- YouTube URL analysis with a short public video.
- Browser button audit including URL submit.
- Backend contract tests and frontend build.

## Dependencies

Add backend dependency:

- `yt-dlp` for YouTube download/extraction.

Use existing Python libraries where possible for HTTP requests and image decoding. If frame extraction needs a binary tool such as `ffmpeg`, the implementation must detect its absence and return a clear error instead of crashing.
