import type { AnalyzerResult, UiRegionPrediction } from "../api/types";
import type { CSSProperties } from "react";

type PreviewStageProps = {
  previewUrl: string | null;
  result: AnalyzerResult | null;
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

  return {
    aspectRatio: `${sourceBox.source_width} / ${sourceBox.source_height}`,
  };
}

export function PreviewStage({ previewUrl, result }: PreviewStageProps) {
  const regions = previewUrl
    ? (result?.regions ?? []).filter((region) => region.bbox).slice(0, 8)
    : [];

  return (
    <section className="preview-stage" aria-label="Image preview">
      <div className="preview-stage__frame">
        {previewUrl ? (
          <div className="preview-image-layer" style={layerStyle(regions)}>
            <img className="preview-stage__image" src={previewUrl} alt="Selected food input" />
            <div className="preview-overlay-layer" aria-hidden={regions.length === 0}>
              {regions.map((region) => (
                <span
                  key={`${region.source_id}-${region.detection_index}`}
                  className="bbox-overlay"
                  style={regionStyle(region)}
                  aria-label={`Detected region ${region.displayIndex}`}
                >
                  {region.displayIndex}
                </span>
              ))}
            </div>
          </div>
        ) : (
          <div className="preview-stage__empty">
            <span>Awaiting image input</span>
          </div>
        )}
      </div>
    </section>
  );
}
