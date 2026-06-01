import { Image, RotateCcw, Sparkles, Upload, Video } from "lucide-react";
import type { ChangeEvent } from "react";

import type { AnalyzerMode, AnalyzerStatus } from "../state/useAnalyzer";

type UploadControlsProps = {
  mode: AnalyzerMode;
  status: AnalyzerStatus;
  onModeChange: (mode: AnalyzerMode) => void;
  onUploadImage: (file: File) => void;
  onVideoSelected: (file: File) => void;
  onSample: () => void;
  onClear: () => void;
};

export function UploadControls({
  mode,
  status,
  onModeChange,
  onUploadImage,
  onVideoSelected,
  onSample,
  onClear,
}: UploadControlsProps) {
  const isLoading = status === "loading";
  const disabled = isLoading;
  const uploadAccept = mode === "video" ? "video/*" : "image/*";

  function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file && mode === "video") {
      onVideoSelected(file);
    } else if (file) {
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
        <button type="button" onClick={onClear} disabled={disabled}>
          <RotateCcw size={16} aria-hidden="true" />
          Clear
        </button>
      </div>
    </section>
  );
}
