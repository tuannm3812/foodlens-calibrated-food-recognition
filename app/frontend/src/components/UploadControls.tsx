import { Image, RotateCcw, Sparkles, Upload, Video } from "lucide-react";
import type { ChangeEvent } from "react";

import type { AnalyzerMode, AnalyzerStatus } from "../state/useAnalyzer";

type UploadControlsProps = {
  mode: AnalyzerMode;
  status: AnalyzerStatus;
  onModeChange: (mode: AnalyzerMode) => void;
  onUploadImage: (file: File) => void;
  onSample: () => void;
  onClear: () => void;
};

export function UploadControls({
  mode,
  status,
  onModeChange,
  onUploadImage,
  onSample,
  onClear,
}: UploadControlsProps) {
  const isLoading = status === "loading";
  const disabled = isLoading;

  function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      onUploadImage(file);
    }
    event.target.value = "";
  }

  return (
    <section className="upload-controls" aria-label="Analyzer controls">
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
      <div className="control-row">
        <label className={`upload-button ${mode === "video" || disabled ? "is-disabled" : ""}`}>
          <Upload size={16} aria-hidden="true" />
          Upload
          <input
            type="file"
            accept="image/*"
            disabled={mode === "video" || disabled}
            onChange={handleUpload}
          />
        </label>
        <button type="button" onClick={onSample} disabled={disabled}>
          <Sparkles size={16} aria-hidden="true" />
          Sample
        </button>
        <button type="button" onClick={onClear} disabled={disabled}>
          <RotateCcw size={16} aria-hidden="true" />
          Clear
        </button>
      </div>
      {mode === "video" ? (
        <p className="control-note">Video upload arrives in Task 6.</p>
      ) : null}
    </section>
  );
}
