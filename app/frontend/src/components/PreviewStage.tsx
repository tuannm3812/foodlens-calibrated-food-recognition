import type { AnalyzerResult, UiRegionPrediction } from "../api/types";
import type { AnalyzerMode } from "../state/useAnalyzer";
import type { CSSProperties } from "react";

type PreviewStageProps = {
  mode: AnalyzerMode;
  previewUrl: string | null;
  result: AnalyzerResult | null;
  selectedRegionKey?: string | null;
};

function regionStyle(region: UiRegionPrediction): CSSProperties {
  const box = region.bbox;
  if (!box) {
    return {};
  }

  const width = Math.max(0, ((box.x2 - box.x1) / box.source_width) * 100);
  const height = Math.max(0, ((box.y2 - box.y1) / box.source_height) * 100);

  return {
    left: `${(box.x1 / box.source_width) * 100}%`,
    top: `${(box.y1 / box.source_height) * 100}%`,
    width: `${width}%`,
    height: `${height}%`,
  };
}

function layerStyle(regions: UiRegionPrediction[]): CSSProperties | undefined {
  const sourceBox = regions.find((region) => region.bbox)?.bbox;
  if (!sourceBox) {
    return undefined;
  }

  const sourceAspectRatio = sourceBox.source_width / sourceBox.source_height;
  const frameAspectRatio = 16 / 10;

  return {
    aspectRatio: `${sourceBox.source_width} / ${sourceBox.source_height}`,
    width: sourceAspectRatio >= frameAspectRatio ? "100%" : "auto",
    height: sourceAspectRatio >= frameAspectRatio ? "auto" : "100%",
  };
}

function regionKey(region: UiRegionPrediction): string {
  return `${region.source_id}-${region.detection_index}`;
}

function formatLabel(label: string): string {
  return label.replace(/_/g, " ");
}

export function PreviewStage({
  mode,
  previewUrl,
  result,
  selectedRegionKey,
}: PreviewStageProps) {
  const regions = previewUrl && mode === "image"
    ? (result?.regions ?? []).filter((region) => region.bbox).slice(0, 8)
    : [];

  return (
    <section className="preview-stage" aria-label={`${mode === "video" ? "Video" : "Image"} preview`}>
      <div className="preview-stage__frame">
        {previewUrl ? (
          <div className="preview-image-layer" style={layerStyle(regions)}>
            {mode === "video" ? (
              <video
                className="preview-stage__image"
                src={previewUrl}
                aria-label="Selected food video"
                controls
                muted
                playsInline
              />
            ) : (
              <>
                <img className="preview-stage__image" src={previewUrl} alt="Selected food input" />
                <div className="preview-overlay-layer" aria-hidden={regions.length === 0}>
                  {regions.map((region) => {
                    const label = formatLabel(region.foodlens.top_label);

                    return (
                      <span
                        key={`${region.source_id}-${region.detection_index}`}
                        className={`bbox-overlay${
                          regionKey(region) === selectedRegionKey
                            ? " bbox-overlay--selected"
                            : ""
                        }`}
                        style={regionStyle(region)}
                        role="img"
                        aria-label={`Region ${region.displayIndex}: ${label}`}
                      >
                        {region.displayIndex} {label}
                      </span>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="preview-stage__empty">
            <span className="preview-stage__empty-icon" aria-hidden="true">
              +
            </span>
            <strong>No subject detected</strong>
            <span>Upload a food {mode} or load the sample to begin analysis.</span>
          </div>
        )}
      </div>
    </section>
  );
}
