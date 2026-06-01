import { Image, Link, RotateCcw, Sparkles, Upload, Video } from "lucide-react";
import type { ChangeEvent, FormEvent } from "react";
import { useState } from "react";

import type { AnalyzerMode, AnalyzerStatus } from "../state/useAnalyzer";

type UploadControlsProps = {
  mode: AnalyzerMode;
  status: AnalyzerStatus;
  onModeChange: (mode: AnalyzerMode) => void;
  onUploadImage: (file: File) => void;
  onVideoSelected: (file: File) => void;
  onUrlSubmit: (url: string) => void;
  onSample: () => void;
  onClear: () => void;
};

export function UploadControls({
  mode,
  status,
  onModeChange,
  onUploadImage,
  onVideoSelected,
  onUrlSubmit,
  onSample,
  onClear,
}: UploadControlsProps) {
  const [urlValue, setUrlValue] = useState("");
  const isLoading = status === "loading";
  const disabled = isLoading;
  const uploadAccept = mode === "video" ? "video/*" : "image/*";
  const urlLabel = mode === "video" ? "YouTube URL" : "Image URL";
  const urlPlaceholder = mode === "video" ? "Paste a YouTube URL" : "Paste a direct image URL";

  function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file && mode === "video") {
      onVideoSelected(file);
    } else if (file) {
      onUploadImage(file);
    }
    event.target.value = "";
  }

  function handleUrlSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedUrl = urlValue.trim();
    if (!trimmedUrl) {
      return;
    }
    onUrlSubmit(trimmedUrl);
    setUrlValue("");
  }

  function handleClear() {
    setUrlValue("");
    onClear();
  }

  return (
    <section className="upload-controls" aria-label="Analyzer controls">
      <div className="upload-controls__input-row">
        <div className="segmented-control" role="group" aria-label="Input mode">
          <button
            type="button"
            className={mode === "image" ? "is-active" : ""}
            disabled={disabled}
            onClick={() => onModeChange("image")}
          >
            <Image size={16} aria-hidden="true" />
            Image
          </button>
          <button
            type="button"
            className={mode === "video" ? "is-active" : ""}
            disabled={disabled}
            onClick={() => onModeChange("video")}
          >
            <Video size={16} aria-hidden="true" />
            Video
          </button>
        </div>
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
      </div>
      <div className="control-row upload-controls__action-row">
        <label className={disabled ? "upload-button is-disabled" : "upload-button"}>
          <Upload size={16} aria-hidden="true" />
          Upload
          <input
            type="file"
            accept={uploadAccept}
            disabled={disabled}
            onChange={handleUpload}
          />
        </label>
        <button type="button" onClick={onSample} disabled={disabled}>
          <Sparkles size={16} aria-hidden="true" />
          Sample
        </button>
        <button type="button" onClick={handleClear} disabled={disabled}>
          <RotateCcw size={16} aria-hidden="true" />
          Clear
        </button>
      </div>
    </section>
  );
}
